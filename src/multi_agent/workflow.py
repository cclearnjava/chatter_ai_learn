# src/multi_agent/workflow.py
"""
Multi-Agent 工作流

整合 Supervisor 和各专家 Agent，提供统一的工作流入口
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime

from src.multi_agent.base import AgentRole, CollaborationContext
from src.multi_agent.protocol import MessageBus, AgentOrchestrator
from src.multi_agent.supervisor import SupervisorAgent
from src.multi_agent.experts import (
    AnalystExpert,
    RecommenderExpert,
    GeneratorExpert,
    ReviewerExpert,
    SafetyGuardExpert,
    FAQExpert,
    ProfileExpert,
)
from src.graph.state import AgentState

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """
    Multi-Agent 工作流
    
    使用方式：
        workflow = MultiAgentWorkflow()
        result = await workflow.run(initial_state)
    """
    
    def __init__(self, settings=None):
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化消息总线
        self.message_bus = MessageBus()
        
        # 初始化编排器
        self.orchestrator = AgentOrchestrator(self.message_bus)
        
        # 初始化 Supervisor
        self.supervisor = SupervisorAgent(self.message_bus, settings)
        
        # 初始化并注册专家 Agent
        self._init_experts()
        
        self.logger.info("Multi-Agent 工作流初始化完成")
    
    def _init_experts(self):
        """初始化并注册所有专家 Agent"""
        experts = [
            AnalystExpert(self.settings),
            RecommenderExpert(self.settings),
            GeneratorExpert(self.settings),
            ReviewerExpert(self.settings),
            SafetyGuardExpert(self.settings),
            FAQExpert(self.settings),
            ProfileExpert(self.settings),
        ]
        
        for expert in experts:
            self.message_bus.register_agent(expert)
            self.logger.info(f"专家已注册: {expert.role.value} - {expert.name}")
    
    async def run(self, initial_state: AgentState) -> AgentState:
        """
        运行 Multi-Agent 工作流
        
        Args:
            initial_state: 初始状态
            
        Returns:
            处理后的状态
        """
        start_time = datetime.now()
        self.logger.info(f"[MultiAgentWorkflow] 开始处理: {initial_state.request_id}")
        
        try:
            # 委托给 Supervisor 处理
            result_state = await self.supervisor.process(initial_state)
            
            # 记录统计
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"[MultiAgentWorkflow] 处理完成: {initial_state.request_id} | "
                f"耗时: {duration:.2f}s | "
                f"状态: {result_state.status}"
            )
            
            return result_state
            
        except Exception as e:
            self.logger.error(f"[MultiAgentWorkflow] 处理失败: {e}", exc_info=True)
            initial_state.error_message = str(e)
            initial_state.status = -1
            return initial_state
    
    def get_stats(self) -> Dict[str, Any]:
        """获取工作流统计信息"""
        return {
            "message_bus": self.message_bus.get_stats(),
            "registered_agents": [
                agent.get_status() 
                for agent in self.message_bus.get_all_agents().values()
            ]
        }


class MultiAgentChatWorkflow:
    """
    Multi-Agent 聊天工作流
    
    与现有 ChatWorkflow 接口兼容，可作为替代方案使用
    """
    
    def __init__(self, settings=None):
        self.multi_agent_workflow = MultiAgentWorkflow(settings)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def run(self, initial_state: AgentState) -> AgentState:
        """运行工作流"""
        return await self.multi_agent_workflow.run(initial_state)
    
    def compile(self):
        """兼容原有接口"""
        return self
    
    async def ainvoke(self, initial_state: AgentState) -> AgentState:
        """兼容 LangGraph 接口"""
        return await self.run(initial_state)


# ======================== 工厂函数 ========================

def create_multi_agent_workflow(settings=None) -> MultiAgentWorkflow:
    """创建 Multi-Agent 工作流实例"""
    return MultiAgentWorkflow(settings)


def create_chat_workflow(
    use_multi_agent: bool = False, 
    settings=None
):
    """
    创建聊天工作流
    
    Args:
        use_multi_agent: 是否使用 Multi-Agent 模式
        settings: 配置
        
    Returns:
        工作流实例
    """
    if use_multi_agent:
        return MultiAgentChatWorkflow(settings)
    else:
        # 使用原有的单一工作流
        from src.graph.workflow import ChatWorkflow
        return ChatWorkflow()
