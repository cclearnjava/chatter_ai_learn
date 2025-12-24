from typing import Dict, Any, Optional, Type, TypeVar
from abc import ABC, abstractmethod  # 新增抽象类约束
import logging

from llm.prompt_buckets import PromptBuckets
from llm.llm_model import LLMModel
from llm.template import Template, TEMPLATES

from settings.settings import SettingsManager
# 兼容原Context和LangGraph的AgentState
from src.graph.state import AgentState  # LangGraph状态类
from utils.schema import Context  # 原上下文类

# 定义状态类型泛型，兼容两种上下文
StateType = TypeVar("StateType", Context, AgentState)

logger = logging.getLogger(__name__)


class Agent(ABC):
    """
    适配LangGraph的Agent基类（抽象基类）
    核心改进：
    1. 标准化LangGraph节点接口
    2. 统一异常处理+生命周期钩子
    3. 自动初始化依赖（减少外部耦合）
    4. 兼容原Context和新AgentState
    """
    # 节点元信息（子类可覆盖）
    node_name: str = "base_agent"
    description: str = "Base agent for LangGraph nodes"

    def __init__(
            self,
            settings: SettingsManager,
            llm_model: Optional[LLMModel] = None,
            prompt_engine: Optional[PromptBuckets] = None
    ) -> None:
        self.settings = settings
        # 自动初始化LLM模型（无外部传入时）
        self.llm_model = llm_model or self._init_llm_model()
        # 自动初始化提示词引擎（无外部传入时）
        self.prompt_engine = prompt_engine or self._init_prompt_engine()

        # 日志初始化
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{self.node_name}]")

    def _init_llm_model(self) -> LLMModel:
        """自动初始化LLM模型（从配置加载）"""
        try:
            from llm.llm_model import LLMModel  # 延迟导入避免循环依赖
            return LLMModel(settings=self.settings)
        except Exception as e:
            self.logger.warning(f"Failed to auto-init LLM model: {e}")
            return None

    def _init_prompt_engine(self) -> PromptBuckets:
        """自动初始化提示词引擎（从配置加载）"""
        try:
            return PromptBuckets(settings=self.settings)
        except Exception as e:
            self.logger.warning(f"Failed to auto-init PromptEngine: {e}")
            return None

    @staticmethod
    def _load_template(template_name: str) -> Template:
        """通用模板加载（保留原逻辑，增强日志）"""
        if not template_name:
            raise ValueError("Template name is invalid!")
        template = TEMPLATES.get(template_name, None)
        if template is None:
            raise ValueError(f"Template {template_name} does not exist.")
        logger.info(f"Loaded template: {template_name}")
        return template

    def before_run(self, state: StateType) -> StateType:
        """执行前钩子（子类可覆盖）：日志/埋点/状态预处理"""
        self.logger.info(f"Starting agent execution | node: {self.node_name}")
        # 统一状态校验（示例：检查必要字段）
        if isinstance(state, AgentState) and not hasattr(state, "dialog_state"):
            self.logger.warning("AgentState missing 'dialog_state' field")
        return state

    def after_run(self, state: StateType, result: Dict[str, Any]) -> Dict[str, Any]:
        """执行后钩子（子类可覆盖）：结果处理/日志/埋点"""
        self.logger.info(f"Finished agent execution | node: {self.node_name} | updated fields: {list(result.keys())}")
        return result

    @abstractmethod  # 强制子类实现
    async def _run_logic(self, state: StateType) -> Dict[str, Any]:
        """核心业务逻辑（子类必须实现）：返回需更新的状态字典"""
        pass

    async def run(self, state: StateType) -> Dict[str, Any]:
        """
        LangGraph标准化执行入口（统一接口）
        :param state: LangGraph状态（AgentState）或原上下文（Context）
        :return: 需更新到状态的字典（符合LangGraph节点要求）
        """
        try:
            # 1. 执行前钩子
            state = self.before_run(state)

            # 2. 执行核心业务逻辑
            result = await self._run_logic(state)

            # 3. 执行后钩子
            result = self.after_run(state, result)

            return result

        except Exception as e:
            # 统一异常捕获+兜底（避免LangGraph工作流中断）
            self.logger.error(f"Agent execution failed | node: {self.node_name} | error: {str(e)}", exc_info=True)
            # 兜底返回（保证工作流不中断）
            return {
                "error": str(e),
                "node_failed": self.node_name,
                "should_regenerate": False  # 适配质检重试逻辑
            }