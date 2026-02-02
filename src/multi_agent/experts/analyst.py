# src/multi_agent/experts/analyst.py
"""
意图分析专家 Agent

职责：
- 分析用户消息的意图
- 识别情感倾向
- 理解上下文语义
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
from src.graph.state import AgentState, ChatAction

logger = logging.getLogger(__name__)


class AnalystExpert(ExpertAgent):
    """
    意图分析专家
    
    能力：
    - intention_analysis: 意图识别
    - sentiment_analysis: 情感分析
    - context_understanding: 上下文理解
    """
    
    role = AgentRole.ANALYST
    name = "analyst_expert"
    description = "分析用户意图，识别需求类型和情感倾向"
    
    capabilities = [
        "intention_analysis",
        "sentiment_analysis", 
        "context_understanding"
    ]
    
    # 意图映射表
    INTENTION_ACTION_MAP = {
        "content_request": ChatAction.FAN_INTENTION_CONTENT_REQUEST.value,
        "ready_to_purchase": ChatAction.FAN_INTENTION_READY_TO_PURCHASE.value,
        "preview_request": ChatAction.FAN_INTENTION_PREVIEW_REQUEST.value,
        "negotiation": ChatAction.FAN_INTENTION_NEGOTIATION.value,
        "objection": ChatAction.FAN_INTENTION_OBJECTION.value,
        "ppv_inquiry": ChatAction.FAN_INTENTION_PPV_INQUIRY.value,
        "bond": ChatAction.FAN_INTENTION_BOND.value,
        "tease": ChatAction.FAN_INTENTION_TEASE.value,
        "sexting": ChatAction.FAN_INTENTION_SEXTING.value,
        "other": ChatAction.PPV_CHAT.value,
    }
    
    # 意图关键词（规则匹配）
    INTENTION_KEYWORDS = {
        "content_request": ["看", "发", "照片", "视频", "图片", "内容", "想要"],
        "ready_to_purchase": ["买", "购买", "付款", "下单", "要了"],
        "preview_request": ["预览", "先看", "试看", "免费"],
        "negotiation": ["便宜", "打折", "优惠", "少一点"],
        "ppv_inquiry": ["多少钱", "价格", "怎么买"],
        "bond": ["喜欢", "爱你", "想你", "亲爱的"],
        "tease": ["调皮", "逗你", "开玩笑"],
    }
    
    def __init__(self, settings=None):
        super().__init__(settings)
        # 可以在这里初始化 LLM 服务用于更精准的意图识别
        self.llm_service = None
        if settings:
            try:
                from src.services.llm_service import LLMService
                self.llm_service = LLMService()
            except Exception as e:
                self.logger.warning(f"LLM 服务初始化失败: {e}")
    
    async def execute(self, task: TaskAssignment) -> TaskResult:
        """执行意图分析任务"""
        start_time = time.time()
        
        try:
            context = task.context
            state_data = context.get("state", {})
            message_content = state_data.get("message", {}).get("content", "")
            
            if not message_content:
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    error="消息内容为空"
                )
            
            # 1. 规则匹配（快速）
            intention = self._rule_based_intention(message_content)
            
            # 2. 如果规则未匹配，尝试 LLM（可选）
            if intention == "other" and self.llm_service:
                intention = await self._llm_intention(message_content, context)
            
            # 3. 意图映射为动作
            action = self.INTENTION_ACTION_MAP.get(intention, ChatAction.PPV_CHAT.value)
            
            # 4. 情感分析（简单版）
            sentiment = self._analyze_sentiment(message_content)
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                output={
                    "intention": intention,
                    "action": action,
                    "sentiment": sentiment,
                    "confidence": 0.8 if intention != "other" else 0.5
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"意图分析失败: {e}", exc_info=True)
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _rule_based_intention(self, text: str) -> str:
        """基于规则的意图识别"""
        text_lower = text.lower()
        
        # 按优先级检查
        priority_order = [
            "ready_to_purchase",
            "content_request", 
            "preview_request",
            "ppv_inquiry",
            "negotiation",
            "bond",
            "tease"
        ]
        
        for intention in priority_order:
            keywords = self.INTENTION_KEYWORDS.get(intention, [])
            if any(kw in text_lower for kw in keywords):
                return intention
        
        return "other"
    
    async def _llm_intention(self, text: str, context: Dict) -> str:
        """基于 LLM 的意图识别"""
        if not self.llm_service:
            return "other"
        
        try:
            prompt = f"""分析以下用户消息的意图，只返回意图类型。

可选意图类型：
- content_request: 用户想要查看内容
- ready_to_purchase: 用户准备购买
- preview_request: 用户想要预览
- negotiation: 用户在讨价还价
- ppv_inquiry: 用户询问价格
- bond: 用户表达情感联系
- tease: 用户在调侃
- other: 其他

用户消息：{text}

只输出意图类型（如 content_request），不要解释："""

            response = await self.llm_service.run(
                prompts=prompt,
                max_new_tokens=20,
                temperature=0.1
            )
            
            intention = response[0].strip().lower() if response else "other"
            
            # 验证返回的意图是否有效
            if intention not in self.INTENTION_ACTION_MAP:
                intention = "other"
            
            return intention
            
        except Exception as e:
            self.logger.error(f"LLM 意图识别失败: {e}")
            return "other"
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """简单的情感分析"""
        positive_words = ["喜欢", "爱", "棒", "好", "开心", "感谢", "谢谢"]
        negative_words = ["不", "差", "讨厌", "失望", "生气", "烦"]
        
        positive_count = sum(1 for w in positive_words if w in text)
        negative_count = sum(1 for w in negative_words if w in text)
        
        if positive_count > negative_count:
            return {"label": "positive", "score": 0.7}
        elif negative_count > positive_count:
            return {"label": "negative", "score": 0.7}
        else:
            return {"label": "neutral", "score": 0.5}
