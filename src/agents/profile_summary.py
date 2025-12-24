import json
from typing import List, Dict, Optional, Any, TypedDict
from pydantic import BaseModel, ValidationError
import logging
from src.agents import Agent
from src.agents.prompt_buckets import PromptBuckets
from src.agents.llm_model import LLMModel
from settings.settings import SettingsManager
from src.graph.state import AgentState  # LangGraph标准状态类
from src.utils import extract_json_from_string
from utils.constant import LLMLogitProcessor, ChatRole
from llm.template import Template, TEMPLATES
from utils.schema import Context, ChatMessage

# 标准化日志（LangGraph节点命名）
logger = logging.getLogger("profile_summary_agent")


# 画像结构类型定义（类型安全）
class FanProfile(BaseModel):
    name: Optional[str] = None
    age: Optional[str] = None
    birthday: Optional[str] = None
    zodiac_sign: Optional[str] = None
    height: Optional[str] = None
    bra_size: Optional[str] = None
    from_: Optional[str] = None  # 避免关键字冲突
    location: Optional[str] = None
    hobbies: Optional[str] = None
    personality: Optional[str] = None
    favorite_expressions: Optional[str] = None
    sexual_favorites: Optional[str] = None

    class Config:
        alias_generator = lambda x: x.replace("_", "")  # 兼容from→from_
        populate_by_name = True


class ProfileSummaryAgent(Agent):
    """
    粉丝画像总结Agent（LangGraph适配版）
    核心能力：基于粉丝聊天历史，调用LLM生成结构化粉丝画像，内置重试/容错/类型校验
    """
    # LangGraph节点元信息
    node_name: str = "fan_profile_summary"
    description: str = "基于粉丝聊天历史，生成结构化的粉丝画像（姓名/年龄/爱好等）"

    def __init__(
            self,
            settings: Optional[SettingsManager] = None,
            llm_model: Optional[LLMModel] = None,
            prompt_engine: Optional[PromptBuckets] = None
    ) -> None:
        super().__init__(settings=settings, llm_model=llm_model, prompt_engine=prompt_engine)
        # 1. 加载模板（带兜底）
        self.template = self._load_template_with_fallback()
        # 2. 加载LLM配置（统一管理）
        self.llm_config = self._load_llm_config()
        # 3. 默认画像（配置化）
        self.default_profile = FanProfile()

    def _load_template_with_fallback(self) -> Template:
        """加载模板（失败返回默认模板，不抛异常）"""
        template_name = self.settings.profile_summary.template_name
        if not template_name:
            logger.warning(f"[{self.node_name}] 模板名称为空，使用默认模板")
            template_name = "default_profile_summary"

        template = TEMPLATES.get(template_name)
        if not template:
            logger.error(f"[{self.node_name}] 模板{template_name}不存在，使用默认模板")
            # 兜底默认模板（保证流程不中断）
            template = Template(
                system_prompt="总结粉丝画像，返回JSON格式，包含name/age/hobbies等字段",
                user_prompt="基于以下聊天历史总结粉丝画像：{{history}}"
            )
        return template

    def _load_llm_config(self) -> Dict[str, Any]:
        """加载LLM调用配置（统一管理）"""
        conf = self.settings.profile_summary
        return {
            "temperature": conf.temperature,
            "top_p": conf.top_p,
            "top_k": conf.top_k,
            "num_return_sequences": conf.num_return_sequences,
            "repetition_penalty": conf.repetition_penalty,
            "max_new_tokens": conf.max_new_tokens,
            "logit_processors": [LLMLogitProcessor.NO_REPEAT_TOKEN.value],
            "retry_times": 2  # 重试次数配置化
        }

    def _extract_and_validate_profile(self, llm_response: str) -> Optional[FanProfile]:
        """提取并校验画像（类型安全）"""
        try:
            # 提取JSON
            rst_json = extract_json_from_string(llm_response)
            if not rst_json:
                logger.warning(f"[{self.node_name}] LLM返回无有效JSON：{llm_response[:100]}")
                return None
            # 类型校验（兼容from→from_）
            return FanProfile(**rst_json)
        except ValidationError as e:
            logger.error(f"[{self.node_name}] 画像JSON校验失败：{e} | 原始数据：{rst_json}")
            return None
        except Exception as e:
            logger.error(f"[{self.node_name}] 画像JSON提取失败：{e}")
            return None

    async def _generate_profile(self, history: List[Dict[str, str]]) -> FanProfile:
        """核心逻辑：生成画像（带重试/兜底）"""
        # 1. 构建Prompt
        try:
            prompts = self.prompt_engine.get_profile_summary_prompt(
                self.template,
                ChatMessage(role=ChatRole.FAN.value),
                history=history
            )
            logger.debug(f"[{self.node_name}] 画像Prompt：\n{prompts}")
        except Exception as e:
            logger.error(f"[{self.node_name}] 构建Prompt失败：{e}")
            return self.default_profile

        # 2. LLM调用（带重试）
        for retry_idx in range(self.llm_config["retry_times"]):
            try:
                responses = await self.llm_model.run(
                    prompts,
                    temperature=self.llm_config["temperature"],
                    top_p=self.llm_config["top_p"],
                    top_k=self.llm_config["top_k"],
                    num_return_sequences=self.llm_config["num_return_sequences"],
                    repetition_penalty=self.llm_config["repetition_penalty"],
                    max_new_tokens=self.llm_config["max_new_tokens"],
                    logit_processors=self.llm_config["logit_processors"]
                )
                logger.debug(f"[{self.node_name}] LLM返回（重试{retry_idx}）：{responses}")

                # 3. 提取并校验结果
                if responses and responses[0]:
                    profile = self._extract_and_validate_profile(responses[0])
                    if profile:
                        logger.info(f"[{self.node_name}] 画像生成成功（重试{retry_idx}）")
                        return profile
            except Exception as e:
                logger.error(f"[{self.node_name}] LLM调用失败（重试{retry_idx}）：{e}")

        # 重试失败返回默认值
        logger.warning(f"[{self.node_name}] 所有重试失败，返回默认画像")
        return self.default_profile

    # ------------------------------ LangGraph生命周期钩子 ------------------------------
    def before_run(self, state: AgentState) -> AgentState:
        """执行前：校验字段、初始化日志"""
        logger.info(f"[{self.node_name}] 开始生成粉丝画像")
        # 校验核心字段
        if not hasattr(state, "chat_history") or not state.chat_history:
            logger.warning(f"[{self.node_name}] AgentState缺失chat_history字段")
            state.chat_history = []
        return state

    async def _run_logic(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph核心逻辑：生成画像并更新状态"""
        # 1. 提取聊天历史
        history = state.chat_history if isinstance(state.chat_history, list) else []
        # 2. 生成画像
        profile = await self._generate_profile(history)
        # 3. 构造状态更新字典
        return {
            "fan_profile": profile.dict(by_alias=True),  # 转回from（兼容旧逻辑）
            "dialog_state": {
                "node_name": self.node_name,
                "node_status": "success",
                "profile_generated": True
            }
        }

    def after_run(self, state: AgentState, result: Dict[str, Any]) -> Dict[str, Any]:
        """执行后：补充日志、监控信息"""
        logger.info(f"[{self.node_name}] 画像生成完成 | 状态：{result['dialog_state']['node_status']}")
        return result

    # ------------------------------ LangGraph标准入口 ------------------------------
    async def run(self, state: AgentState) -> Dict[str, Any]:
        """
        LangGraph节点标准入口
        :param state: LangGraph AgentState（需包含chat_history字段）
        :return: 状态更新字典（含fan_profile）
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
            logger.error(f"[{self.node_name}] 画像生成异常：{e}", exc_info=True)
            # 异常兜底
            return {
                "fan_profile": self.default_profile.dict(by_alias=True),
                "dialog_state": {
                    "node_name": self.node_name,
                    "node_status": "failed",
                    "profile_generated": False,
                    "error": str(e)
                }
            }

    # ------------------------------ 兼容旧逻辑的入口（可选） ------------------------------
    async def run_direct(self, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """兼容旧逻辑：直接接收历史，返回画像字典"""
        profile = await self._generate_profile(history)
        return profile.dict(by_alias=True)

    async def run_legacy(self, context: Context) -> str:
        """兼容旧Context入口：返回JSON字符串"""
        profile = await self._generate_profile(context.chat_history.get_flatten_history())
        return json.dumps(profile.dict(by_alias=True))