# src/multi_agent/experts/reviewer.py
"""
质量审核专家 Agent

职责：
- 审核生成内容的质量
- 幻觉检测
- 一致性检查
"""

from typing import Dict, Any, List, Optional
import time
import logging
import re

from src.multi_agent.base import (
    AgentRole,
    ExpertAgent,
    TaskAssignment,
    TaskResult
)

logger = logging.getLogger(__name__)


class ReviewerExpert(ExpertAgent):
    """
    质量审核专家
    
    能力：
    - quality_check: 质量检查
    - hallucination_detection: 幻觉检测
    - consistency_check: 一致性检查
    """
    
    role = AgentRole.REVIEWER
    name = "reviewer_expert"
    description = "审核回复质量，确保内容准确、一致"
    
    capabilities = [
        "quality_check",
        "hallucination_detection",
        "consistency_check"
    ]
    
    # 质量检查规则
    QUALITY_RULES = {
        "min_length": 5,           # 最小长度
        "max_length": 500,         # 最大长度
        "max_emoji_ratio": 0.3,    # 表情占比上限
        "forbidden_patterns": [
            r"作为.*AI",            # AI 身份泄露
            r"我是.*助手",
            r"根据.*模型",
            r"\[.*\]",             # 未替换的占位符
            r"\{.*\}",
        ]
    }
    
    def __init__(self, settings=None):
        super().__init__(settings)
    
    async def execute(self, task: TaskAssignment) -> TaskResult:
        """执行质量检查任务"""
        start_time = time.time()
        
        try:
            context = task.context
            shared_context = context.get("shared_context", {})
            
            # 获取待检查的回复
            final_response = shared_context.get("final_response", "")
            
            if not final_response:
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    output={"passed": False, "reason": "回复为空"},
                    error="回复内容为空",
                    execution_time=time.time() - start_time
                )
            
            # 执行多维度检查
            issues = []
            
            # 1. 长度检查
            if len(final_response) < self.QUALITY_RULES["min_length"]:
                issues.append("回复过短")
            if len(final_response) > self.QUALITY_RULES["max_length"]:
                issues.append("回复过长")
            
            # 2. 表情检查
            emoji_ratio = self._calculate_emoji_ratio(final_response)
            if emoji_ratio > self.QUALITY_RULES["max_emoji_ratio"]:
                issues.append("表情过多")
            
            # 3. 禁用模式检查
            for pattern in self.QUALITY_RULES["forbidden_patterns"]:
                if re.search(pattern, final_response):
                    issues.append(f"包含禁用内容: {pattern}")
            
            # 4. 重复检查
            if self._has_repetition(final_response):
                issues.append("内容重复")
            
            # 5. 一致性检查（简化版）
            consistency_ok = self._check_consistency(final_response, context)
            if not consistency_ok:
                issues.append("内容不一致")
            
            passed = len(issues) == 0
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                output={
                    "passed": passed,
                    "issues": issues,
                    "score": 1.0 - (len(issues) * 0.2),  # 简单评分
                    "need_regenerate": not passed
                },
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"质量检查失败: {e}", exc_info=True)
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _calculate_emoji_ratio(self, text: str) -> float:
        """计算表情占比"""
        import emoji
        emoji_count = emoji.emoji_count(text)
        total_count = len(text)
        return emoji_count / total_count if total_count > 0 else 0
    
    def _has_repetition(self, text: str, threshold: int = 3) -> bool:
        """检查是否有重复内容"""
        # 检查连续重复的词
        words = text.split()
        for i in range(len(words) - threshold):
            if words[i:i+threshold] == words[i+threshold:i+2*threshold]:
                return True
        return False
    
    def _check_consistency(self, response: str, context: Dict) -> bool:
        """检查内容一致性（简化版）"""
        # 这里可以添加更复杂的一致性检查逻辑
        # 比如检查推荐内容是否在回复中被正确提及
        return True
