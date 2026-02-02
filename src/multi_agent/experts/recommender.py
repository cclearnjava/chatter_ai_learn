# src/multi_agent/experts/recommender.py
"""
内容推荐专家 Agent

职责：
- 基于用户画像推荐内容
- PPV 素材匹配与排序
- 个性化推荐策略
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


class RecommenderExpert(ExpertAgent):
    """
    内容推荐专家
    
    能力：
    - content_recommendation: 内容推荐
    - ppv_selection: PPV 素材选择
    - personalization: 个性化策略
    """
    
    role = AgentRole.RECOMMENDER
    name = "recommender_expert"
    description = "基于用户画像和意图，推荐最合适的内容"
    
    capabilities = [
        "content_recommendation",
        "ppv_selection",
        "personalization"
    ]
    
    def __init__(self, settings=None):
        super().__init__(settings)
        # 复用现有的推荐 Agent
        self.recommender_agent = None
        try:
            from src.agents.recommender import RecommendAgent
            self.recommender_agent = RecommendAgent(settings=settings)
        except Exception as e:
            self.logger.warning(f"推荐 Agent 初始化失败: {e}")
    
    async def execute(self, task: TaskAssignment) -> TaskResult:
        """执行推荐任务"""
        start_time = time.time()
        
        try:
            context = task.context
            state_data = context.get("state", {})
            previous_results = context.get("previous_results", [])
            
            # 从之前的结果中获取意图信息
            intention = None
            for result in previous_results:
                if result.get("output", {}).get("intention"):
                    intention = result["output"]["intention"]
                    break
            
            # 如果有现成的推荐 Agent，复用它
            if self.recommender_agent:
                from src.graph.state import AgentState
                state = AgentState(**state_data)
                
                # 调用现有推荐逻辑
                result = await self.recommender_agent.run(state)
                
                recommend_material = result.get("dialog_state", {}).get("recommend_material", [])
                
                return TaskResult(
                    task_id=task.task_id,
                    success=True,
                    output={
                        "recommend_material": recommend_material,
                        "recommendation_count": len(recommend_material),
                        "strategy": "tag_based"
                    },
                    execution_time=time.time() - start_time
                )
            
            # 简化的推荐逻辑（兜底）
            return TaskResult(
                task_id=task.task_id,
                success=True,
                output={
                    "recommend_material": [],
                    "recommendation_count": 0,
                    "strategy": "fallback"
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"推荐失败: {e}", exc_info=True)
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
