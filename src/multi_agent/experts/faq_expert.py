# src/multi_agent/experts/faq_expert.py
"""
FAQ 专家 Agent

职责：
- FAQ 知识库匹配
- 常见问题快速解答
- 知识检索
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


class FAQExpert(ExpertAgent):
    """
    FAQ 专家
    
    能力：
    - faq_matching: FAQ 匹配
    - knowledge_retrieval: 知识检索
    """
    
    role = AgentRole.FAQ_EXPERT
    name = "faq_expert"
    description = "快速匹配 FAQ 知识库，解答常见问题"
    
    capabilities = [
        "faq_matching",
        "knowledge_retrieval"
    ]
    
    def __init__(self, settings=None):
        super().__init__(settings)
        # 复用现有的 FAQ 匹配 Agent
        self.faq_matcher = None
        try:
            from src.agents.faq_matcher_agent import FaqMatcherAgent
            self.faq_matcher = FaqMatcherAgent()
        except Exception as e:
            self.logger.warning(f"FAQ Agent 初始化失败: {e}")
        
        # 简化的 FAQ 库（实际应从配置或数据库加载）
        self.faq_database = {
            "价格": "我们的内容价格根据类型不同，从 $5 到 $50 不等，详细价格可以私聊我哦~",
            "付款": "支持多种付款方式，包括信用卡、PayPal 等，安全便捷~",
            "退款": "购买后如有问题可以联系客服处理，我们会尽力帮你解决~",
            "内容": "我有各种精彩内容等你来发现，可以告诉我你喜欢什么类型的~",
            "会员": "成为会员可以享受更多专属内容和优惠哦~",
        }
    
    async def execute(self, task: TaskAssignment) -> TaskResult:
        """执行 FAQ 匹配任务"""
        start_time = time.time()
        
        try:
            context = task.context
            state_data = context.get("state", {})
            message_content = state_data.get("message", {}).get("content", "")
            
            if not message_content:
                return TaskResult(
                    task_id=task.task_id,
                    success=True,
                    output={
                        "faq_matched": False,
                        "faq_answer": None
                    },
                    execution_time=time.time() - start_time
                )
            
            # 使用现有 FAQ 匹配器
            if self.faq_matcher:
                try:
                    from src.graph.state import AgentState
                    state = AgentState(**state_data)
                    result = await self.faq_matcher.run(state)
                    
                    faq_matched = result.get("faq_matched", False)
                    faq_answer = result.get("faq_answer", "")
                    
                    return TaskResult(
                        task_id=task.task_id,
                        success=True,
                        output={
                            "faq_matched": faq_matched,
                            "faq_answer": faq_answer,
                            "match_method": "agent"
                        },
                        execution_time=time.time() - start_time
                    )
                except Exception as e:
                    self.logger.warning(f"FAQ Agent 调用失败: {e}")
            
            # 简化的 FAQ 匹配
            faq_matched = False
            faq_answer = None
            
            for keyword, answer in self.faq_database.items():
                if keyword in message_content:
                    faq_matched = True
                    faq_answer = answer
                    break
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                output={
                    "faq_matched": faq_matched,
                    "faq_answer": faq_answer,
                    "match_method": "keyword"
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"FAQ 匹配失败: {e}", exc_info=True)
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
