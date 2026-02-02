# src/multi_agent/experts/safety_guard.py
"""
安全守卫专家 Agent

职责：
- 输入验证（Prompt 注入检测）
- 输出违规检测
- 敏感信息过滤
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


class SafetyGuardExpert(ExpertAgent):
    """
    安全守卫专家
    
    能力：
    - input_validation: 输入验证
    - violation_detection: 违规检测
    - output_filtering: 输出过滤
    """
    
    role = AgentRole.SAFETY_GUARD
    name = "safety_guard_expert"
    description = "守护系统安全，检测违规内容和恶意输入"
    
    capabilities = [
        "input_validation",
        "violation_detection",
        "output_filtering"
    ]
    
    # Prompt 注入检测模式
    INJECTION_PATTERNS = [
        r"忽略.*指令",
        r"ignore.*instruction",
        r"forget.*rule",
        r"你是.*不是",
        r"现在你是",
        r"假装你是",
        r"pretend.*you.*are",
        r"act.*as.*if",
        r"jailbreak",
        r"DAN",
        r"开发者模式",
        r"developer.*mode",
    ]
    
    # 敏感词模式（简化版，实际应从配置加载）
    SENSITIVE_PATTERNS = [
        r"密码",
        r"身份证",
        r"银行卡",
        r"password",
        r"credit.*card",
    ]
    
    def __init__(self, settings=None):
        super().__init__(settings)
        # 复用现有的违规检测 Agent
        self.violation_detector = None
        try:
            from src.agents.violation_detect import ViolationDetectionAgent
            self.violation_detector = ViolationDetectionAgent(settings=settings)
        except Exception as e:
            self.logger.warning(f"违规检测 Agent 初始化失败: {e}")
    
    async def execute(self, task: TaskAssignment) -> TaskResult:
        """执行安全检查任务"""
        start_time = time.time()
        
        try:
            task_type = task.task_type
            
            if task_type == "input_validation":
                return await self._validate_input(task, start_time)
            elif task_type == "violation_detection":
                return await self._detect_violation(task, start_time)
            else:
                return TaskResult(
                    task_id=task.task_id,
                    success=False,
                    error=f"未知任务类型: {task_type}",
                    execution_time=time.time() - start_time
                )
            
        except Exception as e:
            self.logger.error(f"安全检查失败: {e}", exc_info=True)
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    async def _validate_input(self, task: TaskAssignment, start_time: float) -> TaskResult:
        """验证输入安全性"""
        context = task.context
        state_data = context.get("state", {})
        message_content = state_data.get("message", {}).get("content", "")
        
        if not message_content:
            return TaskResult(
                task_id=task.task_id,
                success=True,
                output={"is_safe": True, "reason": "空输入"},
                execution_time=time.time() - start_time
            )
        
        issues = []
        
        # 1. Prompt 注入检测
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, message_content, re.IGNORECASE):
                issues.append(f"疑似 Prompt 注入: {pattern}")
        
        # 2. 敏感信息检测
        for pattern in self.SENSITIVE_PATTERNS:
            if re.search(pattern, message_content, re.IGNORECASE):
                issues.append(f"包含敏感信息: {pattern}")
        
        # 3. 长度检查（防止资源耗尽攻击）
        if len(message_content) > 2000:
            issues.append("输入过长")
        
        is_safe = len(issues) == 0
        
        return TaskResult(
            task_id=task.task_id,
            success=True,
            output={
                "is_safe": is_safe,
                "issues": issues,
                "sanitized_input": message_content if is_safe else None
            },
            execution_time=time.time() - start_time
        )
    
    async def _detect_violation(self, task: TaskAssignment, start_time: float) -> TaskResult:
        """检测输出违规"""
        context = task.context
        shared_context = context.get("shared_context", {})
        final_response = shared_context.get("final_response", "")
        
        if not final_response:
            return TaskResult(
                task_id=task.task_id,
                success=True,
                output={"is_safe": True, "reason": "空输出"},
                execution_time=time.time() - start_time
            )
        
        # 使用现有违规检测器
        if self.violation_detector:
            try:
                from src.graph.state import AgentState
                state_data = context.get("state", {})
                state = AgentState(**state_data)
                
                is_violation = await self.violation_detector.run_legacy(state, final_response)
                
                return TaskResult(
                    task_id=task.task_id,
                    success=True,
                    output={
                        "is_safe": not is_violation,
                        "violation_detected": is_violation
                    },
                    execution_time=time.time() - start_time
                )
            except Exception as e:
                self.logger.error(f"违规检测失败: {e}")
        
        # 简化的违规检测
        is_safe = True
        for pattern in self.SENSITIVE_PATTERNS:
            if re.search(pattern, final_response, re.IGNORECASE):
                is_safe = False
                break
        
        return TaskResult(
            task_id=task.task_id,
            success=True,
            output={"is_safe": is_safe},
            execution_time=time.time() - start_time
        )
