from typing import Dict, Any, Optional, Union
import re
import logging
from httpx import AsyncClient, TimeoutException
from openai import AsyncOpenAI, OpenAIError

from src.agents import Agent  # 导入优化后的Agent基类
from settings.settings import SettingsManager
from utils.schema import Context  # 原上下文类
from src.graph.state import AgentState  # LangGraph状态类
from utils.constant import ChatAction

# 标准化日志（复用基类logger命名规范）
logger = logging.getLogger(__name__)


class IntentionAgent(Agent):
    """
    粉丝意图识别Agent（适配LangGraph）
    核心能力：规则优先+LLM兜底识别粉丝意图，映射为标准化业务动作
    """
    # LangGraph节点元信息（覆盖基类）
    node_name: str = "intention_detection"
    description: str = "识别粉丝消息意图（内容请求/准备购买等），并映射为标准化业务动作"

    def __init__(
            self,
            settings: Optional[SettingsManager] = None,
            prompt_engine=None,
            llm_model=None  # 基类兼容参数（本类暂未使用，保留以适配基类）
    ) -> None:
        # 复用基类初始化逻辑（自动加载配置）
        _settings = settings or SettingsManager.get_instance()
        super().__init__(settings=_settings, prompt_engine=prompt_engine, llm_model=llm_model)

        # 意图识别核心配置
        self._init_openai_client()
        self.model_id = _settings.fan_intention.model_id
        self.intention_action_map = {
            "content request": ChatAction.FAN_INTENTION_CONTENT_REQUEST.action,
            "ready to purchase": ChatAction.FAN_INTENTION_READY_TO_PURCHASE.action,
            "preview request": ChatAction.FAN_INTENTION_PREVIEW_REQUEST.action,
            "negotiation": ChatAction.FAN_INTENTION_NEGOTIATION.action,
            "objection": ChatAction.FAN_INTENTION_OBJECTION.action,
            "ppv inquiry": ChatAction.FAN_INTENTION_PPV_INQUIRY.action,
            "bond": ChatAction.FAN_INTENTION_BOND.action,
            "tease": ChatAction.FAN_INTENTION_TEASE.action,
            "sexting": ChatAction.FAN_INTENTION_SEXTING.action,
            "other": ChatAction.PPV_CHAT.action,
        }

        # LLM生成参数（配置化）
        self.temperature = _settings.fan_intention.temperature
        self.top_p = _settings.fan_intention.top_p
        self.num_return_sequences = _settings.fan_intention.num_return_sequences
        self.max_new_tokens = _settings.fan_intention.max_new_tokens
        self.top_k = _settings.fan_intention.top_k
        self.repetition_penalty = _settings.fan_intention.repetition_penalty

    def _init_openai_client(self) -> None:
        """初始化OpenAI客户端（封装为独立方法，便于维护）"""
        settings = self.settings.fan_intention
        # 构建HTTP客户端（支持代理/超时）
        http_client = None
        if settings.use_proxy:
            http_client = AsyncClient(
                proxies=self.settings.server.proxy,
                timeout=Timeout(10.0)
            )

        # 构建请求头（支持自定义Authorization）
        default_headers = {}
        if settings.authorization:
            default_headers = {
                "Content-Type": "application/json",
                "Authorization": settings.authorization
            }

        # 初始化AsyncOpenAI客户端
        self.client = AsyncOpenAI(
            base_url=None if settings.infer_backend == "openai" else settings.url,
            api_key=settings.api_key,
            http_client=http_client,
            default_headers=default_headers
        )

    def _cr_intention_rule(self, content: str) -> bool:
        """规则1：检测粉丝「内容请求」意图（小写+正则匹配）"""
        if not content:
            return False
        cr_patterns = ["can i see", "i wanna see", "i'd like to see", "let me see", "show me"]
        cr_regex = re.compile("|".join(map(re.escape, cr_patterns)), re.IGNORECASE)
        return bool(cr_regex.search(content.lower()))

    def _rtp_intention_rule(self, pre_content: str, now_content: str) -> bool:
        """规则2：检测粉丝「准备购买」意图（创作者预告+粉丝确认）"""
        if not pre_content or not now_content:
            return False
        # 创作者预告关键词
        c_patterns = ["wanna see", "want to see", "show you", "are you ready", "are u ready"]
        c_regex = re.compile("|".join(map(re.escape, c_patterns)), re.IGNORECASE)
        # 粉丝确认关键词
        f_patterns = [
            "please", "yes", "ok", "okay", "yeah", "ready", "si", "sure", "definitely",
            "show me", "i would love to", "i do", "alright", "let's see", "let me see",
            "send me", "of course", "go on"
        ]
        f_regex = re.compile("|".join(map(re.escape, f_patterns)), re.IGNORECASE)
        return bool(c_regex.search(pre_content.lower())) and bool(f_regex.search(now_content.lower()))

    def _intention_to_action(self, intention: str) -> str:
        """意图→标准化动作映射（增强容错）"""
        clean_intention = intention.strip().lower()
        return self.intention_action_map.get(clean_intention, self.intention_action_map["other"])

    async def _call_llm_for_intention(self, system_prompt: str, user_content: str) -> str:
        """调用LLM识别意图（封装为独立方法，便于异常处理）"""
        try:
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                model=self.model_id,
                temperature=self.temperature,
                top_p=self.top_p,
                n=self.num_return_sequences,
                max_completion_tokens=self.max_new_tokens,
                timeout=10.0,
                extra_body={
                    "top_k": self.top_k,
                    "repetition_penalty": self.repetition_penalty,
                }
            )
            # 解析LLM返回结果（兼容「｜」分割和纯文本）
            raw_result = response.choices[0].message.content.strip().lower()
            content_parts = raw_result.split('｜')
            intention = content_parts[1] if len(content_parts) > 1 else content_parts[0]
            self.logger.info(f"[IntentionAgent] LLM识别意图：{intention}")
            return intention
        except TimeoutException:
            self.logger.error("[IntentionAgent] LLM调用超时")
            return "other"
        except OpenAIError as e:
            self.logger.error(f"[IntentionAgent] OpenAI接口错误：{str(e)}")
            return "other"
        except Exception as e:
            self.logger.error(f"[IntentionAgent] LLM调用异常：{str(e)}", exc_info=True)
            return "other"

    def before_run(self, state: Union[Context, AgentState]) -> Union[Context, AgentState]:
        """执行前钩子：校验必要字段"""
        self.logger.info(f"[IntentionAgent] 开始意图识别 | 节点：{self.node_name}")
        # 校验核心字段
        if isinstance(state, Context):
            if not state.message or not state.chat_history:
                self.logger.warning("[IntentionAgent] 上下文缺失消息/历史对话")
        else:  # AgentState
            if not state.message or not getattr(state, "flatten_history", None):
                self.logger.warning("[IntentionAgent] AgentState缺失消息/扁平化历史")
        return state

    async def _run_logic(self, state: Union[Context, AgentState]) -> Dict[str, Any]:
        """
        核心业务逻辑（实现基类抽象方法）
        :param state: 兼容原Context和LangGraph的AgentState
        :return: LangGraph标准的状态更新字典
        """
        # 1. 状态字段适配（统一获取消息/历史）
        if isinstance(state, Context):
            message = state.message
            history = state.chat_history.get_flatten_history()[-1:]  # 取最后1条历史
        else:
            message = state.message
            history = state.flatten_history[-1:] if hasattr(state, "flatten_history") else []

        # 2. 构建Prompt（复用prompt_engine）
        system_prompt, user_content = self.prompt_engine.get_fan_intention_prompt(state)

        # 3. 规则优先识别意图
        intention = "other"
        if message.role == "fan":
            # 规则1：内容请求
            if self._cr_intention_rule(str(message.content)):
                intention = "content request"
                self.logger.info(f"[IntentionAgent] 规则匹配意图：{intention}")
            # 规则2：准备购买（需历史有创作者消息）
            elif history and history[0].get("role") == "creator":
                pre_content = history[0].get("content", "")
                now_content = str(message.content)
                if self._rtp_intention_rule(pre_content, now_content):
                    intention = "ready to purchase"
                    self.logger.info(f"[IntentionAgent] 规则匹配意图：{intention}")

        # 4. 规则未命中→调用LLM兜底
        if intention == "other":
            intention = await self._call_llm_for_intention(system_prompt, user_content)

        # 5. 意图映射为动作
        action = self._intention_to_action(intention)

        # 6. 返回LangGraph标准状态更新字典
        return {
            "dialog_state": {
                "intention": intention,
                "action": action,
                "intention_detected": True  # 新增标识，便于后续流程判断
            }
        }

    def after_run(self, state: Union[Context, AgentState], result: Dict[str, Any]) -> Dict[str, Any]:
        """执行后钩子：增强日志"""
        intention = result["dialog_state"]["intention"]
        action = result["dialog_state"]["action"]
        self.logger.info(f"[IntentionAgent] 意图识别完成 | 意图：{intention} | 动作：{action}")
        return result

    # 兼容原run方法（平滑迁移旧代码）
    async def run(self, state: Union[Context, AgentState]) -> Union[Context, AgentState, Dict[str, Any]]:
        try:
            # 执行前钩子
            state = self.before_run(state)
            # 核心逻辑
            update_dict = await self._run_logic(state)
            # 执行后钩子
            update_dict = self.after_run(state, update_dict)

            # 兼容逻辑：如果是原Context，直接更新；如果是AgentState，返回更新字典
            if isinstance(state, Context):
                state.dialog_state.update(**update_dict["dialog_state"])
                return state
            else:
                return update_dict
        except Exception as e:
            self.logger.error(f"[IntentionAgent] 执行失败：{str(e)}", exc_info=True)
            # 异常兜底返回
            fallback_dict = {
                "dialog_state": {
                    "intention": "other",
                    "action": ChatAction.PPV_CHAT.action,
                    "intention_detected": False,
                    "error": str(e)
                }
            }
            if isinstance(state, Context):
                state.dialog_state.update(**fallback_dict["dialog_state"])
                return state
            else:
                return fallback_dict