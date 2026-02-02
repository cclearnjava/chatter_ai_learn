# src/multi_agent/experts/generator.py
"""
对话生成专家 Agent

职责：
- 基于上下文生成对话回复
- 融合推荐内容到回复中
- 保持人设一致性
"""

from typing import Dict, Any, List, Optional
import time
import logging

from src.multi_agent.base import (
    AgentRole,
    ExpertAgent,
    TaskAssignment,
    TaskResult
)

logger = logging.getLogger(__name__)


class GeneratorExpert(ExpertAgent):
    """
    对话生成专家
    
    能力：
    - response_generation: 回复生成
    - dialog_creation: 对话创建
    - persona_consistency: 人设一致性
    """
    
    role = AgentRole.GENERATOR
    name = "generator_expert"
    description = "生成自然、个性化的对话回复"
    
    capabilities = [
        "response_generation",
        "dialog_creation",
        "persona_consistency"
    ]
    
    def __init__(self, settings=None):
        super().__init__(settings)
        # 复用现有的生成 Agent
        self.chat_generator = None
        try:
            from src.agents.chat_generator import ChatGeneratorAgent
            self.chat_generator = ChatGeneratorAgent()
        except Exception as e:
            self.logger.warning(f"生成 Agent 初始化失败: {e}")
    
    async def execute(self, task: TaskAssignment) -> TaskResult:
        """执行生成任务"""
        start_time = time.time()
        
        try:
            context = task.context
            state_data = context.get("state", {})
            shared_context = context.get("shared_context", {})
            previous_results = context.get("previous_results", [])
            
            # 收集之前步骤的结果
            intention = shared_context.get("intention", "other")
            recommend_material = shared_context.get("recommend_material", [])
            faq_answer = shared_context.get("faq_answer")
            
            # 如果有 FAQ 答案，直接使用
            if faq_answer:
                return TaskResult(
                    task_id=task.task_id,
                    success=True,
                    output={
                        "final_response": faq_answer,
                        "generation_method": "faq"
                    },
                    execution_time=time.time() - start_time
                )
            
            # 使用现有生成器
            if self.chat_generator:
                from src.graph.state import AgentState
                state = AgentState(**state_data)
                
                # 更新状态中的意图和推荐
                if intention:
                    state.dialog_state.intention = intention
                if recommend_material:
                    state.dialog_state.recommend_material = recommend_material
                
                result = await self.chat_generator.run(state)
                final_response = result.get("final_response", "")
                
                return TaskResult(
                    task_id=task.task_id,
                    success=True,
                    output={
                        "final_response": final_response,
                        "generation_method": "llm"
                    },
                    execution_time=time.time() - start_time
                )
            
            # 兜底回复
            return TaskResult(
                task_id=task.task_id,
                success=True,
                output={
                    "final_response": "你好~有什么可以帮你的吗？",
                    "generation_method": "fallback"
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"生成失败: {e}", exc_info=True)
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
