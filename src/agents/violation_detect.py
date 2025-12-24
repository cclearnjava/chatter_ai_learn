import os
import json
import httpx
import pandas as pd
import re
from typing import List, Dict, Optional, Any, Tuple, Union
from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI, OpenAIError, TimeoutError
from src.agents import Agent
from settings.settings import SettingsManager
from src.utils import PROJECT_ROOT_PATH
from src.graph.state import AgentState  # LangGraph标准状态类
import logging
import asyncio

# 标准化日志（带LangGraph节点标识）
logger = logging.getLogger("violation_detection_agent")

# 违规检测结果类型定义（类型安全）
class ViolationDetectionResult(BaseModel):
    is_violation: bool = False  # 是否违规
    violation_type: Optional[str] = None  # 违规类型：vocab（规则式）/lm（语义式）/both（两者均违规）
    violating_words: Optional[List[str]] = None  # 匹配到的违规词
    lm_result: Optional[Dict[str, Any]] = None  # LLM检测原始结果
    error_msg: Optional[str] = None  # 异常信息

# 检测配置常量（便于维护）
DEFAULT_CHAT_HISTORY_LIMIT = 2  # 默认取最近2条聊天历史
VIOLATION_VOCAB_DEFAULT_PATH = "config/violation_vocab.csv"  # 词表默认路径

class ViolationDetectionAgent(Agent):
    """
    违规检测Agent（LangGraph适配版）
    核心能力：规则式（违规词表）+ 语义式（LLM）双层检测，输出标准化违规结果
    """
    # LangGraph节点元信息
    node_name: str = "violation_detection"
    description: str = "通过规则式词表匹配和语义式LLM判断，检测文本是否存在违规内容"

    def __init__(
        self,
        settings: Optional[SettingsManager] = None,
        prompt_engine=None,
        llm_model=None  # 兼容Agent基类参数
    ) -> None:
        super().__init__(settings=settings, prompt_engine=prompt_engine, llm_model=llm_model)
        # 1. 加载核心配置
        self.conf = self.settings.violation if self.settings else None
        self.server_conf = self.settings.server if self.settings else None

        # 2. 初始化LLM客户端（带资源管理）
        self.client = self._init_llm_client()

        # 3. LLM生成参数（统一配置管理）
        self.llm_config = self._load_llm_config()

        # 4. 加载违规词表（带兜底）
        self.violating_words_regx, self.violating_word_list = self._load_violating_words_with_fallback()

        # 5. 检测策略配置
        self.detection_strategy = {
            "vocab_first": True,  # 规则式优先（规则判定违规则不调用LLM）
            "chat_history_limit": DEFAULT_CHAT_HISTORY_LIMIT
        }

        logger.info(f"[{self.node_name}] 违规检测Agent初始化完成")

    def _init_llm_client(self) -> Optional[AsyncOpenAI]:
        """初始化LLM客户端（带资源管理，异常兜底）"""
        if not self.conf:
            logger.warning(f"[{self.node_name}] 无违规检测配置，跳过LLM客户端初始化")
            return None

        try:
            # 构建HTTP客户端
            http_client = None
            if self.conf.use_proxy and self.server_conf:
                http_client = httpx.AsyncClient(
                    proxies=self.server_conf.proxy,
                    timeout=httpx.Timeout(10.0),
                    limits=httpx.Limits(max_connections=10)  # 限制连接数
                )

            # 构建默认请求头
            default_headers = None
            if self.conf.authorization:
                default_headers = {
                    "Content-Type": "application/json",
                    "Authorization": self.conf.authorization
                }

            # 初始化AsyncOpenAI客户端
            return AsyncOpenAI(
                base_url=None if self.conf.infer_backend == "openai" else self.conf.url,
                api_key=self.conf.api_key,
                http_client=http_client,
                default_headers=default_headers
            )
        except Exception as e:
            logger.error(f"[{self.node_name}] LLM客户端初始化失败：{e}", exc_info=True)
            return None

    def _load_llm_config(self) -> Dict[str, Any]:
        """加载LLM生成配置（统一管理，带默认值）"""
        if not self.conf:
            return {
                "model_id": "gpt-3.5-turbo",
                "top_p": 0.9,
                "temperature": 0.1,
                "presence_penalty": 0.0,
                "timeout": 10.0
            }

        return {
            "model_id": self.conf.model_id,
            "top_p": self.conf.top_p,
            "temperature": self.conf.temperature,
            "presence_penalty": self.conf.presence_penalty,
            "timeout": 10.0
        }

    def _load_violating_words_with_fallback(self) -> Tuple[Optional[re.Pattern], List[str]]:
        """加载违规词表（带兜底，避免初始化失败）"""
        if not self.conf or not self.conf.violation_vocab:
            vocab_path = os.path.join(PROJECT_ROOT_PATH.absolute(), VIOLATION_VOCAB_DEFAULT_PATH)
            logger.warning(f"[{self.node_name}] 未配置违规词表路径，使用默认路径：{vocab_path}")
        else:
            vocab_path = os.path.join(PROJECT_ROOT_PATH.absolute(), self.conf.violation_vocab)

        # 校验文件
        if not os.path.exists(vocab_path) or os.path.getsize(vocab_path) == 0:
            logger.error(f"[{self.node_name}] 违规词表文件不存在或为空：{vocab_path}，跳过规则式检测")
            return None, []

        try:
            # 读取词表并去重
            df = pd.read_csv(vocab_path)
            if "word" not in df.columns:
                logger.error(f"[{self.node_name}] 违规词表缺少'word'列：{vocab_path}")
                return None, []

            violating_words = df["word"].drop_duplicates().astype(str).values.tolist()
            if not violating_words:
                logger.warning(f"[{self.node_name}] 违规词表无有效数据：{vocab_path}")
                return None, []

            # 构建正则（忽略大小写，支持整词匹配）
            escaped_words = [re.escape(word) for word in violating_words]
            regx_pattern = r"\b({})\b".format("|".join(escaped_words))
            violating_words_regx = re.compile(regx_pattern, re.IGNORECASE)  # 忽略大小写

            logger.info(f"[{self.node_name}] 成功加载违规词表，共{len(violating_words)}个违规词")
            return violating_words_regx, violating_words
        except Exception as e:
            logger.error(f"[{self.node_name}] 加载违规词表失败：{e}", exc_info=True)
            return None, []

    def _check_vocab_violation(self, text: str) -> Tuple[bool, List[str]]:
        """规则式违规检测（优化匹配逻辑，返回详情）"""
        if not self.violating_words_regx or not text:
            return False, []

        # 查找所有匹配项（去重）
        matches = list(set(self.violating_words_regx.findall(text)))
        is_violation = len(matches) > 0
        if is_violation:
            logger.info(f"[{self.node_name}] 规则式检测到违规词：{matches}")
        return is_violation, matches

    async def _check_lm_violation(self, state: AgentState, text: str) -> ViolationDetectionResult:
        """语义式违规检测（细分异常，返回标准化结果）"""
        result = ViolationDetectionResult()

        # 前置校验
        if not self.client or not self.prompt_engine:
            result.error_msg = "LLM客户端或Prompt引擎未初始化"
            logger.error(f"[{self.node_name}] {result.error_msg}")
            return result

        if not text or not hasattr(state, "chat_history"):
            result.error_msg = "待检测文本或聊天历史为空"
            logger.warning(f"[{self.node_name}] {result.error_msg}")
            return result

        # 1. 构建Prompt
        try:
            # 获取最近N条聊天历史
            flatten_history = state.chat_history[-self.detection_strategy["chat_history_limit"]:]
            system_prompt, user_content = self.prompt_engine.get_violation_detect_prompt(
                state.message.content if hasattr(state, "message") else "",
                text,
                flatten_history
            )
        except Exception as e:
            result.error_msg = f"构建Prompt失败：{str(e)}"
            logger.error(f"[{self.node_name}] {result.error_msg}")
            return result

        # 2. LLM调用
        try:
            chat_completion = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                model=self.llm_config["model_id"],
                timeout=self.llm_config["timeout"],
                temperature=self.llm_config["temperature"],
                top_p=self.llm_config["top_p"],
                presence_penalty=self.llm_config["presence_penalty"],
                response_format={"type": "json_object"}
            )
        except TimeoutError:
            result.error_msg = "LLM调用超时"
            logger.warning(f"[{self.node_name}] {result.error_msg}")
            return result
        except OpenAIError as e:
            result.error_msg = f"LLM接口错误：{str(e)}"
            logger.error(f"[{self.node_name}] {result.error_msg}")
            return result
        except Exception as e:
            result.error_msg = f"LLM调用异常：{str(e)}"
            logger.error(f"[{self.node_name}] {result.error_msg}", exc_info=True)
            return result

        # 3. 结果解析
        try:
            raw_result = chat_completion.choices[0].message.content.strip()
            # 清理代码块格式
            clean_result = raw_result.replace("```json\n", "").replace("\n```", "").strip()
            lm_result_dict = json.loads(clean_result)
            result.lm_result = lm_result_dict

            # 解析label字段（支持int/str类型）
            label = lm_result_dict.get("label", 0)
            label = int(label) if str(label).isdigit() else 0
            result.is_violation = label != 0
            result.violation_type = "lm"

            logger.info(f"[{self.node_name}] 语义式检测结果：{'违规' if result.is_violation else '合规'} | 原始结果：{raw_result}")
        except json.JSONDecodeError:
            result.error_msg = "LLM返回结果不是有效JSON"
            logger.error(f"[{self.node_name}] {result.error_msg} | 原始结果：{raw_result[:100]}")
        except Exception as e:
            result.error_msg = f"解析LLM结果失败：{str(e)}"
            logger.error(f"[{self.node_name}] {result.error_msg}", exc_info=True)

        return result

    async def _run_detection(self, state: AgentState, text: str) -> ViolationDetectionResult:
        """统一检测入口（整合规则式+语义式，支持策略配置）"""
        final_result = ViolationDetectionResult()

        # 1. 规则式检测（优先）
        vocab_violation, vocab_matches = self._check_vocab_violation(text)
        if vocab_violation:
            final_result.is_violation = True
            final_result.violation_type = "vocab"
            final_result.violating_words = vocab_matches

            # 规则式优先：直接返回，不调用LLM
            if self.detection_strategy["vocab_first"]:
                logger.info(f"[{self.node_name}] 规则式检测到违规，跳过语义式检测")
                return final_result

        # 2. 语义式检测（规则式未违规或不启用优先策略）
        lm_result = await self._check_lm_violation(state, text)
        if lm_result.is_violation:
            final_result.is_violation = True
            final_result.lm_result = lm_result.lm_result
            final_result.error_msg = lm_result.error_msg

            # 标记违规类型
            if vocab_violation:
                final_result.violation_type = "both"
            else:
                final_result.violation_type = "lm"
        elif vocab_violation:
            final_result.is_violation = True
            final_result.violation_type = "vocab"
            final_result.violating_words = vocab_matches

        return final_result

    # ------------------------------ LangGraph生命周期钩子 ------------------------------
    def before_run(self, state: AgentState) -> AgentState:
        """执行前：校验字段、初始化日志"""
        logger.info(f"[{self.node_name}] 开始执行违规检测")

        # 校验核心字段
        if not hasattr(state, "to_detect_text"):
            # 兼容：从message.content获取待检测文本
            if hasattr(state, "message") and hasattr(state.message, "content"):
                state.to_detect_text = state.message.content
            else:
                logger.warning(f"[{self.node_name}] AgentState缺失待检测文本（to_detect_text）")
                state.to_detect_text = ""

        if not hasattr(state, "chat_history"):
            state.chat_history = []

        if not hasattr(state, "dialog_state"):
            state.dialog_state = {}

        return state

    async def _run_logic(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph核心逻辑：执行检测并更新状态"""
        # 提取待检测文本
        text = state.to_detect_text
        if not text:
            logger.warning(f"[{self.node_name}] 待检测文本为空，返回合规结果")
            detection_result = ViolationDetectionResult()
            return {
                "dialog_state": {
                    "violation_detection_result": detection_result.dict(),
                    "node_name": self.node_name,
                    "node_status": "success",
                    "is_content_safe": not detection_result.is_violation
                }
            }

        # 执行检测
        detection_result = await self._run_detection(state, text)

        # 构造状态更新字典
        return {
            "dialog_state": {
                "violation_detection_result": detection_result.dict(),
                "node_name": self.node_name,
                "node_status": "success" if not detection_result.error_msg else "failed",
                "is_content_safe": not detection_result.is_violation
            }
        }

    def after_run(self, state: AgentState, result: Dict[str, Any]) -> Dict[str, Any]:
        """执行后：资源清理、补充日志"""
        dialog_state = result.get("dialog_state", {})
        detection_result = dialog_state.get("violation_detection_result", {})
        logger.info(
            f"[{self.node_name}] 违规检测完成 | 状态：{dialog_state.get('node_status')} | "
            f"是否合规：{dialog_state.get('is_content_safe')} | 违规类型：{detection_result.get('violation_type')}"
        )

        # 关闭HTTP客户端，避免连接泄漏
        if self.client and self.client.http_client:
            async def close_client():
                await self.client.http_client.aclose()
            asyncio.create_task(close_client())

        return result

    # ------------------------------ LangGraph标准入口 ------------------------------
    async def run(self, state: AgentState) -> Dict[str, Any]:
        """
        LangGraph节点标准入口
        :param state: LangGraph AgentState（需包含to_detect_text/chat_history字段）
        :return: 状态更新字典（含违规检测结果）
        """
        try:
            # 执行前钩子
            state = self.before_run(state)
            # 核心逻辑
            update_dict = await self._run_logic(state)
            # 执行后钩子
            update_dict = self.after_run(state, update_dict)
            return update_dict
        except Exception as e:
            error_msg = f"违规检测流程异常：{str(e)}"
            logger.error(f"[{self.node_name}] {error_msg}", exc_info=True)
            # 异常兜底
            return {
                "dialog_state": {
                    "violation_detection_result": ViolationDetectionResult(error_msg=error_msg).dict(),
                    "node_name": self.node_name,
                    "node_status": "failed",
                    "is_content_safe": False  # 异常时默认判定不合规，提升安全性
                }
            }

    # ------------------------------ 兼容旧逻辑的入口（可选） ------------------------------
    async def run_legacy(self, context, response: str) -> bool:
        """兼容旧Context入口：返回布尔值检测结果"""
        # 转换旧Context为AgentState
        state = AgentState()
        state.to_detect_text = response
        state.chat_history = context.chat_history.get_flatten_history()
        state.message = context.message
        state.dialog_state = context.dialog_state.__dict__

        # 执行检测
        update_dict = await self.run(state)
        detection_result = ViolationDetectionResult(** update_dict["dialog_state"]["violation_detection_result"])
        return detection_result.is_violation

    # 兼容旧方法名
    check_violating_with_vocab = _check_vocab_violation
    check_violating_with_lm = _check_lm_violation