# src/multi_agent/__init__.py
"""
Multi-Agent 协作系统

架构说明：
- Supervisor Agent: 决策中枢，负责任务分析、分派、汇总
- Expert Agents: 专家角色，各司其职
- Protocol: Agent 间通信协议
- Workflow: Multi-Agent 工作流编排
"""

from src.multi_agent.base import (
    AgentRole,
    AgentMessage,
    TaskAssignment,
    ExpertAgent
)
from src.multi_agent.supervisor import SupervisorAgent
from src.multi_agent.protocol import AgentProtocol, MessageBus
from src.multi_agent.workflow import MultiAgentWorkflow

__all__ = [
    "AgentRole",
    "AgentMessage", 
    "TaskAssignment",
    "ExpertAgent",
    "SupervisorAgent",
    "AgentProtocol",
    "MessageBus",
    "MultiAgentWorkflow"
]
