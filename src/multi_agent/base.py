# src/multi_agent/base.py
"""
Multi-Agent 基础定义

包含:
- AgentRole: Agent 角色枚举
- AgentMessage: Agent 间通信消息
- TaskAssignment: 任务分配结构
- ExpertAgent: 专家 Agent 基类
"""

from typing import Dict, Any, List, Optional, Callable, Union
from pydantic import BaseModel, Field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)


# ======================== 角色定义 ========================

class AgentRole(str, Enum):
    """Agent 角色定义"""
    
    # 核心角色
    SUPERVISOR = "supervisor"           # 决策中枢
    
    # 专家角色
    ANALYST = "analyst"                 # 意图分析师
    RECOMMENDER = "recommender"         # 内容推荐师
    GENERATOR = "generator"             # 对话生成师
    REVIEWER = "reviewer"               # 质量审核师
    SAFETY_GUARD = "safety_guard"       # 安全守卫
    FAQ_EXPERT = "faq_expert"           # FAQ 专家
    PROFILE_ANALYST = "profile_analyst" # 画像分析师
    
    @property
    def description(self) -> str:
        """角色描述"""
        descriptions = {
            self.SUPERVISOR: "决策中枢，负责任务分析、分派和结果汇总",
            self.ANALYST: "分析用户意图，识别需求类型",
            self.RECOMMENDER: "基于用户画像和意图推荐合适内容",
            self.GENERATOR: "生成自然、个性化的对话回复",
            self.REVIEWER: "审核回复质量，确保符合标准",
            self.SAFETY_GUARD: "检测输入输出安全，防止违规内容",
            self.FAQ_EXPERT: "匹配FAQ知识库，快速解答常见问题",
            self.PROFILE_ANALYST: "分析用户画像，提供个性化建议"
        }
        return descriptions.get(self, "未知角色")


class MessageType(str, Enum):
    """消息类型"""
    TASK_ASSIGN = "task_assign"         # 任务分配
    TASK_RESULT = "task_result"         # 任务结果
    QUERY = "query"                     # 查询请求
    RESPONSE = "response"               # 查询响应
    FEEDBACK = "feedback"               # 反馈
    BROADCAST = "broadcast"             # 广播
    HANDOFF = "handoff"                 # 任务交接


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """任务优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# ======================== 消息结构 ========================

class AgentMessage(BaseModel):
    """Agent 间通信消息"""
    
    message_id: str = Field(default_factory=lambda: f"msg_{datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    message_type: MessageType
    sender: AgentRole
    receiver: AgentRole
    content: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    reply_to: Optional[str] = None  # 回复的消息ID
    
    class Config:
        use_enum_values = True


class TaskAssignment(BaseModel):
    """任务分配结构"""
    
    task_id: str = Field(default_factory=lambda: f"task_{datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    assigned_to: AgentRole
    task_type: str
    instruction: str                    # 任务指令
    context: Dict[str, Any] = Field(default_factory=dict)  # 任务上下文
    priority: TaskPriority = TaskPriority.NORMAL
    deadline: Optional[datetime] = None
    dependencies: List[str] = Field(default_factory=list)  # 依赖的任务ID
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    class Config:
        use_enum_values = True


class TaskResult(BaseModel):
    """任务执行结果"""
    
    task_id: str
    success: bool
    output: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0  # 执行耗时（秒）
    confidence: float = 1.0
    quality_score: Optional[float] = None
    retryable: bool = False
    handoff_required: bool = False
    risk_level: str = "low"
    reason: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ======================== 专家 Agent 基类 ========================

class ExpertAgent(ABC):
    """
    专家 Agent 基类
    
    所有专家 Agent 继承此类，实现特定领域的能力
    """
    
    # 子类必须定义的属性
    role: AgentRole = AgentRole.ANALYST
    name: str = "base_expert"
    description: str = "专家 Agent 基类"
    
    # 能力声明（子类定义）
    capabilities: List[str] = []
    
    def __init__(self, settings=None):
        self.settings = settings
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{self.role.value}]")
        self._message_handlers: Dict[MessageType, Callable] = {}
        self._setup_handlers()
    
    def _setup_handlers(self):
        """设置消息处理器（子类可覆盖）"""
        self._message_handlers = {
            MessageType.TASK_ASSIGN: self._handle_task_assign,
            MessageType.QUERY: self._handle_query,
            MessageType.FEEDBACK: self._handle_feedback
        }
    
    # =============== 生命周期方法 ===============
    
    def before_execute(self, task: TaskAssignment) -> TaskAssignment:
        """执行前钩子"""
        self.logger.info(f"[{self.role.value}] 开始执行任务: {task.task_id}")
        task.status = TaskStatus.IN_PROGRESS
        return task
    
    def after_execute(self, task: TaskAssignment, result: TaskResult) -> TaskResult:
        """执行后钩子"""
        self.logger.info(
            f"[{self.role.value}] 任务完成: {task.task_id} | "
            f"成功: {result.success} | 耗时: {result.execution_time:.2f}s"
        )
        return result
    
    @abstractmethod
    async def execute(self, task: TaskAssignment) -> TaskResult:
        """
        执行任务（子类必须实现）
        
        Args:
            task: 任务分配
            
        Returns:
            TaskResult: 任务执行结果
        """
        pass
    
    # =============== 消息处理 ===============
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理接收到的消息"""
        handler = self._message_handlers.get(message.message_type)
        if handler:
            return await handler(message)
        else:
            self.logger.warning(f"未知消息类型: {message.message_type}")
            return None
    
    async def _handle_task_assign(self, message: AgentMessage) -> AgentMessage:
        """处理任务分配消息"""
        task_data = message.content.get("task")
        if not task_data:
            return self._create_error_response(message, "任务数据为空")
        
        task = TaskAssignment(**task_data)
        task = self.before_execute(task)
        
        try:
            result = await self.execute(task)
            result = self.after_execute(task, result)
            
            return AgentMessage(
                message_type=MessageType.TASK_RESULT,
                sender=self.role,
                receiver=message.sender,
                content={"result": result.dict()},
                reply_to=message.message_id
            )
        except Exception as e:
            self.logger.error(f"任务执行失败: {e}", exc_info=True)
            return self._create_error_response(message, str(e))
    
    async def _handle_query(self, message: AgentMessage) -> AgentMessage:
        """处理查询请求（子类可覆盖）"""
        return AgentMessage(
            message_type=MessageType.RESPONSE,
            sender=self.role,
            receiver=message.sender,
            content={"capabilities": self.capabilities, "status": "ready"},
            reply_to=message.message_id
        )
    
    async def _handle_feedback(self, message: AgentMessage) -> Optional[AgentMessage]:
        """处理反馈（子类可覆盖）"""
        self.logger.info(f"收到反馈: {message.content}")
        return None
    
    def _create_error_response(self, original: AgentMessage, error: str) -> AgentMessage:
        """创建错误响应"""
        return AgentMessage(
            message_type=MessageType.TASK_RESULT,
            sender=self.role,
            receiver=original.sender,
            content={"result": TaskResult(
                task_id=original.content.get("task", {}).get("task_id", "unknown"),
                success=False,
                error=error
            ).dict()},
            reply_to=original.message_id
        )
    
    # =============== 工具方法 ===============
    
    def can_handle(self, task_type: str) -> bool:
        """检查是否能处理某类任务"""
        return task_type in self.capabilities
    
    def get_status(self) -> Dict[str, Any]:
        """获取 Agent 状态"""
        return {
            "role": self.role.value,
            "name": self.name,
            "capabilities": self.capabilities,
            "status": "ready"
        }


# ======================== 协作上下文 ========================

class CollaborationContext(BaseModel):
    """协作上下文 - 在多个 Agent 间共享的状态"""
    
    session_id: str
    original_input: str                           # 原始用户输入
    current_stage: str = "init"                   # 当前阶段
    completed_tasks: List[str] = Field(default_factory=list)  # 已完成的任务
    task_results: Dict[str, TaskResult] = Field(default_factory=dict)  # 任务结果
    shared_context: Dict[str, Any] = Field(default_factory=dict)  # 共享上下文
    message_history: List[AgentMessage] = Field(default_factory=list)  # 消息历史
    final_output: Optional[str] = None            # 最终输出
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_task_result(self, task_id: str, result: TaskResult):
        """添加任务结果"""
        self.task_results[task_id] = result
        if result.success:
            self.completed_tasks.append(task_id)
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """获取任务结果"""
        return self.task_results.get(task_id)
    
    def update_shared_context(self, key: str, value: Any):
        """更新共享上下文"""
        self.shared_context[key] = value
    
    def get_shared_context(self, key: str, default: Any = None) -> Any:
        """获取共享上下文"""
        return self.shared_context.get(key, default)
