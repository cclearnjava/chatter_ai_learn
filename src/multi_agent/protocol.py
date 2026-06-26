# src/multi_agent/protocol.py
"""
Agent 通信协议

包含:
- AgentProtocol: 通信协议定义
- MessageBus: 消息总线（Agent 间通信中枢）
"""

from typing import Dict, Any, List, Optional, Callable, Awaitable
from collections import defaultdict
import asyncio
import logging
from datetime import datetime

from src.multi_agent.base import (
    AgentRole,
    AgentMessage,
    MessageType,
    ExpertAgent
)

logger = logging.getLogger(__name__)


class AgentProtocol:
    """
    Agent 通信协议
    
    定义 Agent 间通信的标准接口和规范
    """
    
    # 协议版本
    VERSION = "1.0"
    
    # 超时设置（秒）
    DEFAULT_TIMEOUT = 30
    TASK_TIMEOUT = 60
    
    @staticmethod
    def create_task_message(
        sender: AgentRole,
        receiver: AgentRole,
        task_type: str,
        instruction: str,
        context: Dict[str, Any] = None,
        priority: str = "normal"
    ) -> AgentMessage:
        """创建任务分配消息"""
        from src.multi_agent.base import TaskAssignment, TaskPriority
        
        task = TaskAssignment(
            assigned_to=receiver,
            task_type=task_type,
            instruction=instruction,
            context=context or {},
            priority=TaskPriority(priority)
        )
        
        return AgentMessage(
            message_type=MessageType.TASK_ASSIGN,
            sender=sender,
            receiver=receiver,
            content={"task": task.dict()}
        )
    
    @staticmethod
    def create_query_message(
        sender: AgentRole,
        receiver: AgentRole,
        query: str,
        params: Dict[str, Any] = None
    ) -> AgentMessage:
        """创建查询消息"""
        return AgentMessage(
            message_type=MessageType.QUERY,
            sender=sender,
            receiver=receiver,
            content={"query": query, "params": params or {}}
        )
    
    @staticmethod
    def create_broadcast_message(
        sender: AgentRole,
        content: Dict[str, Any]
    ) -> AgentMessage:
        """创建广播消息"""
        return AgentMessage(
            message_type=MessageType.BROADCAST,
            sender=sender,
            receiver=AgentRole.SUPERVISOR,  # 广播由 Supervisor 转发
            content=content
        )
    
    @staticmethod
    def create_handoff_message(
        sender: AgentRole,
        receiver: AgentRole,
        reason: str,
        context: Dict[str, Any]
    ) -> AgentMessage:
        """创建任务交接消息"""
        return AgentMessage(
            message_type=MessageType.HANDOFF,
            sender=sender,
            receiver=receiver,
            content={"reason": reason, "context": context}
        )


class MessageBus:
    """
    消息总线
    
    负责 Agent 间的消息路由和传递
    """
    
    def __init__(self):
        self._agents: Dict[AgentRole, ExpertAgent] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._subscribers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self._message_history: List[AgentMessage] = []
        self._running = False
        self._logger = logging.getLogger("MessageBus")
    
    # =============== Agent 注册 ===============
    
    def register_agent(self, agent: ExpertAgent):
        """注册 Agent 到消息总线"""
        self._agents[agent.role] = agent
        self._logger.info(f"Agent 已注册: {agent.role.value} ({agent.name})")
    
    def unregister_agent(self, role: AgentRole):
        """注销 Agent"""
        if role in self._agents:
            del self._agents[role]
            self._logger.info(f"Agent 已注销: {role.value}")
    
    def get_agent(self, role: AgentRole) -> Optional[ExpertAgent]:
        """获取已注册的 Agent"""
        return self._agents.get(role)
    
    def get_all_agents(self) -> Dict[AgentRole, ExpertAgent]:
        """获取所有已注册的 Agent"""
        return self._agents.copy()
    
    # =============== 消息发送 ===============
    
    async def send(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        发送消息并等待响应
        
        Args:
            message: 要发送的消息
            
        Returns:
            响应消息（如果有）
        """
        self._message_history.append(message)
        self._logger.debug(
            f"消息发送: {message.sender.value} -> {message.receiver.value} "
            f"[{message.message_type.value}]"
        )
        
        # 广播消息
        if message.message_type == MessageType.BROADCAST:
            await self._broadcast(message)
            return None
        
        # 点对点消息
        receiver = self._agents.get(message.receiver)
        if not receiver:
            self._logger.error(f"目标 Agent 未注册: {message.receiver.value}")
            return None
        
        try:
            response = await asyncio.wait_for(
                receiver.handle_message(message),
                timeout=AgentProtocol.DEFAULT_TIMEOUT
            )
            if response:
                self._message_history.append(response)
            return response
        except asyncio.TimeoutError:
            self._logger.error(f"消息处理超时: {message.message_id}")
            return None
        except Exception as e:
            self._logger.error(f"消息处理失败: {e}", exc_info=True)
            return None
    
    async def send_and_wait(
        self, 
        message: AgentMessage, 
        timeout: float = None
    ) -> Optional[AgentMessage]:
        """发送消息并等待响应（带超时）"""
        timeout = timeout or AgentProtocol.TASK_TIMEOUT
        try:
            return await asyncio.wait_for(
                self.send(message),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self._logger.error(f"等待响应超时: {message.message_id}")
            return None
    
    async def _broadcast(self, message: AgentMessage):
        """广播消息给所有 Agent"""
        tasks = []
        for role, agent in self._agents.items():
            if role != message.sender:  # 不发给自己
                broadcast_msg = AgentMessage(
                    message_type=message.message_type,
                    sender=message.sender,
                    receiver=role,
                    content=message.content,
                    metadata=message.metadata
                )
                tasks.append(agent.handle_message(broadcast_msg))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    # =============== 订阅机制 ===============
    
    def subscribe(self, message_type: MessageType, callback: Callable):
        """订阅特定类型的消息"""
        self._subscribers[message_type].append(callback)
    
    def unsubscribe(self, message_type: MessageType, callback: Callable):
        """取消订阅"""
        if callback in self._subscribers[message_type]:
            self._subscribers[message_type].remove(callback)
    
    # =============== 工具方法 ===============
    
    def get_message_history(self) -> List[AgentMessage]:
        """获取消息历史"""
        return self._message_history.copy()
    
    def clear_history(self):
        """清空消息历史"""
        self._message_history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取消息总线统计"""
        return {
            "registered_agents": [
                role.value if hasattr(role, "value") else str(role)
                for role in self._agents.keys()
            ],
            "message_count": len(self._message_history),
            "agent_count": len(self._agents)
        }


class AgentOrchestrator:
    """
    Agent 编排器
    
    管理 Agent 的启动、停止和生命周期
    """
    
    def __init__(self, message_bus: MessageBus = None):
        self.message_bus = message_bus or MessageBus()
        self._logger = logging.getLogger("AgentOrchestrator")
    
    def add_agent(self, agent: ExpertAgent):
        """添加 Agent"""
        self.message_bus.register_agent(agent)
    
    def remove_agent(self, role: AgentRole):
        """移除 Agent"""
        self.message_bus.unregister_agent(role)
    
    async def execute_task_chain(
        self, 
        tasks: List[Dict[str, Any]],
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        执行任务链（顺序执行）
        
        Args:
            tasks: 任务列表 [{"agent": AgentRole, "task_type": str, "instruction": str}]
            context: 共享上下文
            
        Returns:
            任务结果列表
        """
        results = []
        shared_context = context or {}
        
        for task_config in tasks:
            agent_role = task_config["agent"]
            task_type = task_config["task_type"]
            instruction = task_config["instruction"]
            
            # 创建任务消息
            message = AgentProtocol.create_task_message(
                sender=AgentRole.SUPERVISOR,
                receiver=agent_role,
                task_type=task_type,
                instruction=instruction,
                context=shared_context
            )
            
            # 发送并等待结果
            response = await self.message_bus.send_and_wait(message)
            
            if response and response.content.get("result"):
                result = response.content["result"]
                results.append(result)
                
                # 更新共享上下文
                if result.get("success") and result.get("output"):
                    shared_context.update(result["output"])
        
        return results
    
    async def execute_parallel_tasks(
        self,
        tasks: List[Dict[str, Any]],
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        并行执行任务
        
        Args:
            tasks: 任务列表
            context: 共享上下文
            
        Returns:
            任务结果列表
        """
        shared_context = context or {}
        
        async def execute_single(task_config):
            message = AgentProtocol.create_task_message(
                sender=AgentRole.SUPERVISOR,
                receiver=task_config["agent"],
                task_type=task_config["task_type"],
                instruction=task_config["instruction"],
                context=shared_context
            )
            response = await self.message_bus.send_and_wait(message)
            return response.content.get("result") if response else None
        
        results = await asyncio.gather(
            *[execute_single(task) for task in tasks],
            return_exceptions=True
        )
        
        return [r for r in results if r and not isinstance(r, Exception)]
