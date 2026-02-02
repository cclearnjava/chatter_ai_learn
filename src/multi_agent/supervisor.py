# src/multi_agent/supervisor.py
"""
Supervisor Agent - 决策中枢

核心职责：
1. 任务分析：分析用户输入，判断任务复杂度和类型
2. 任务规划：制定执行计划，分解为子任务
3. 任务分派：将子任务分配给合适的专家 Agent
4. 结果汇总：收集各专家结果，整合最终输出
5. 动态调整：根据执行情况调整计划
"""

from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
import asyncio
import logging
import json
from datetime import datetime

from src.multi_agent.base import (
    AgentRole,
    AgentMessage,
    TaskAssignment,
    TaskResult,
    TaskStatus,
    TaskPriority,
    MessageType,
    CollaborationContext
)
from src.multi_agent.protocol import MessageBus, AgentProtocol
from src.graph.state import AgentState
from src.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class TaskPlan(BaseModel):
    """任务执行计划"""
    
    plan_id: str = Field(default_factory=lambda: f"plan_{datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    parallel_groups: List[List[str]] = Field(default_factory=list)  # 可并行的任务组
    current_step: int = 0
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.now)
    
    def add_step(
        self, 
        agent: AgentRole, 
        task_type: str, 
        instruction: str,
        depends_on: List[str] = None,
        can_parallel: bool = False
    ):
        """添加执行步骤"""
        step_id = f"step_{len(self.steps) + 1}"
        self.steps.append({
            "step_id": step_id,
            "agent": agent,
            "task_type": task_type,
            "instruction": instruction,
            "depends_on": depends_on or [],
            "can_parallel": can_parallel,
            "status": "pending",
            "result": None
        })
        return step_id


class SupervisorAgent:
    """
    Supervisor Agent - 决策中枢
    
    采用 Plan-Execute-Reflect 模式：
    1. Plan: 分析任务，制定计划
    2. Execute: 分派任务给专家执行
    3. Reflect: 检查结果，必要时调整
    """
    
    role = AgentRole.SUPERVISOR
    name = "supervisor"
    description = "决策中枢，负责任务分析、规划、分派和结果汇总"
    
    def __init__(self, message_bus: MessageBus, settings=None):
        self.message_bus = message_bus
        self.settings = settings
        self.llm_service = LLMService() if settings else None
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # 专家能力映射
        self.expert_capabilities = {
            AgentRole.ANALYST: ["intention_analysis", "sentiment_analysis", "context_understanding"],
            AgentRole.FAQ_EXPERT: ["faq_matching", "knowledge_retrieval"],
            AgentRole.RECOMMENDER: ["content_recommendation", "ppv_selection", "personalization"],
            AgentRole.GENERATOR: ["response_generation", "dialog_creation"],
            AgentRole.REVIEWER: ["quality_check", "hallucination_detection", "consistency_check"],
            AgentRole.SAFETY_GUARD: ["violation_detection", "input_validation", "output_filtering"],
            AgentRole.PROFILE_ANALYST: ["profile_summary", "user_modeling"]
        }
        
        # 任务类型到专家的映射
        self.task_expert_mapping = {
            "intention_analysis": AgentRole.ANALYST,
            "faq_matching": AgentRole.FAQ_EXPERT,
            "content_recommendation": AgentRole.RECOMMENDER,
            "response_generation": AgentRole.GENERATOR,
            "quality_check": AgentRole.REVIEWER,
            "violation_detection": AgentRole.SAFETY_GUARD,
            "profile_summary": AgentRole.PROFILE_ANALYST
        }
    
    # =============== 核心流程 ===============
    
    async def process(self, state: AgentState) -> AgentState:
        """
        处理请求（主入口）
        
        Args:
            state: Agent 状态
            
        Returns:
            处理后的状态
        """
        self.logger.info(f"[Supervisor] 开始处理请求: {state.request_id}")
        
        # 创建协作上下文
        context = CollaborationContext(
            session_id=state.request_id,
            original_input=state.message.content if state.message else ""
        )
        
        try:
            # 1. 分析任务
            task_analysis = await self._analyze_task(state, context)
            self.logger.info(f"[Supervisor] 任务分析完成: {task_analysis}")
            
            # 2. 制定计划
            plan = await self._create_plan(state, task_analysis, context)
            self.logger.info(f"[Supervisor] 计划制定完成: {len(plan.steps)} 步骤")
            
            # 3. 执行计划
            results = await self._execute_plan(state, plan, context)
            
            # 4. 汇总结果
            final_state = await self._aggregate_results(state, results, context)
            
            # 5. 反思检查（可选重新规划）
            if await self._need_replan(final_state, context):
                self.logger.info("[Supervisor] 触发重新规划")
                final_state = await self._replan_and_execute(final_state, context)
            
            return final_state
            
        except Exception as e:
            self.logger.error(f"[Supervisor] 处理失败: {e}", exc_info=True)
            state.error_message = str(e)
            state.status = -1
            return state
    
    # =============== 任务分析 ===============
    
    async def _analyze_task(
        self, 
        state: AgentState, 
        context: CollaborationContext
    ) -> Dict[str, Any]:
        """
        分析任务，判断复杂度和所需专家
        """
        user_input = state.message.content if state.message else ""
        
        # 基础分析（规则 + LLM）
        analysis = {
            "input_length": len(user_input),
            "complexity": "simple",  # simple / medium / complex
            "required_experts": [],
            "task_types": [],
            "needs_faq": True,
            "needs_recommendation": False,
            "is_sensitive": False
        }
        
        # 规则判断
        if len(user_input) > 100:
            analysis["complexity"] = "medium"
        
        # 关键词检测
        content_keywords = ["看", "照片", "视频", "内容", "发", "推荐"]
        purchase_keywords = ["买", "购买", "价格", "多少钱"]
        
        if any(kw in user_input for kw in content_keywords):
            analysis["needs_recommendation"] = True
            analysis["task_types"].append("content_recommendation")
            analysis["required_experts"].append(AgentRole.RECOMMENDER)
        
        if any(kw in user_input for kw in purchase_keywords):
            analysis["task_types"].append("intention_analysis")
            analysis["required_experts"].append(AgentRole.ANALYST)
        
        # 默认需要的专家
        if AgentRole.ANALYST not in analysis["required_experts"]:
            analysis["required_experts"].insert(0, AgentRole.ANALYST)
        
        analysis["required_experts"].append(AgentRole.GENERATOR)
        analysis["required_experts"].append(AgentRole.REVIEWER)
        
        # 更新上下文
        context.update_shared_context("task_analysis", analysis)
        
        return analysis
    
    # =============== 计划制定 ===============
    
    async def _create_plan(
        self, 
        state: AgentState, 
        analysis: Dict[str, Any],
        context: CollaborationContext
    ) -> TaskPlan:
        """
        根据任务分析制定执行计划
        """
        plan = TaskPlan()
        
        # 阶段1：输入安全检查（始终执行）
        plan.add_step(
            agent=AgentRole.SAFETY_GUARD,
            task_type="input_validation",
            instruction="检查用户输入是否包含 Prompt 注入或恶意内容"
        )
        
        # 阶段2：FAQ 匹配（可选）
        if analysis.get("needs_faq", True):
            plan.add_step(
                agent=AgentRole.FAQ_EXPERT,
                task_type="faq_matching",
                instruction="在 FAQ 知识库中搜索匹配的答案"
            )
        
        # 阶段3：意图分析
        plan.add_step(
            agent=AgentRole.ANALYST,
            task_type="intention_analysis",
            instruction="分析用户意图，识别需求类型（内容请求/购买意向/闲聊等）"
        )
        
        # 阶段4：内容推荐（条件执行）
        if analysis.get("needs_recommendation", False):
            plan.add_step(
                agent=AgentRole.RECOMMENDER,
                task_type="content_recommendation",
                instruction="基于用户意图和画像，推荐合适的 PPV 内容",
                depends_on=["step_3"]  # 依赖意图分析
            )
        
        # 阶段5：回复生成
        plan.add_step(
            agent=AgentRole.GENERATOR,
            task_type="response_generation",
            instruction="基于上下文和推荐结果，生成自然的对话回复"
        )
        
        # 阶段6：质量审核
        plan.add_step(
            agent=AgentRole.REVIEWER,
            task_type="quality_check",
            instruction="检查生成的回复质量，包括一致性、幻觉检测等"
        )
        
        # 阶段7：输出安全检查
        plan.add_step(
            agent=AgentRole.SAFETY_GUARD,
            task_type="violation_detection",
            instruction="检查输出是否包含违规内容"
        )
        
        context.update_shared_context("execution_plan", plan.dict())
        
        return plan
    
    # =============== 计划执行 ===============
    
    async def _execute_plan(
        self, 
        state: AgentState, 
        plan: TaskPlan,
        context: CollaborationContext
    ) -> List[TaskResult]:
        """
        执行计划
        """
        results = []
        
        for i, step in enumerate(plan.steps):
            plan.current_step = i
            step["status"] = "in_progress"
            
            self.logger.info(
                f"[Supervisor] 执行步骤 {i+1}/{len(plan.steps)}: "
                f"{step['agent'].value} - {step['task_type']}"
            )
            
            # 检查依赖
            if step.get("depends_on"):
                for dep_id in step["depends_on"]:
                    dep_step = next((s for s in plan.steps if s["step_id"] == dep_id), None)
                    if dep_step and dep_step["status"] != "completed":
                        self.logger.warning(f"依赖步骤未完成: {dep_id}")
            
            # 创建任务消息
            message = AgentProtocol.create_task_message(
                sender=self.role,
                receiver=step["agent"],
                task_type=step["task_type"],
                instruction=step["instruction"],
                context={
                    "state": state.dict(),
                    "shared_context": context.shared_context,
                    "previous_results": [r.dict() for r in results]
                }
            )
            
            # 发送并等待结果
            response = await self.message_bus.send_and_wait(message)
            
            if response and response.content.get("result"):
                result_data = response.content["result"]
                result = TaskResult(**result_data) if isinstance(result_data, dict) else result_data
                results.append(result)
                step["status"] = "completed"
                step["result"] = result.dict() if hasattr(result, 'dict') else result
                
                # 更新共享上下文
                if hasattr(result, 'output') and result.output:
                    context.shared_context.update(result.output)
                    
                # 检查是否需要提前终止
                if await self._should_short_circuit(step, result, context):
                    self.logger.info(f"[Supervisor] 提前终止: {step['task_type']}")
                    break
            else:
                step["status"] = "failed"
                self.logger.error(f"[Supervisor] 步骤执行失败: {step['step_id']}")
        
        plan.status = "completed"
        return results
    
    async def _should_short_circuit(
        self, 
        step: Dict[str, Any], 
        result: TaskResult,
        context: CollaborationContext
    ) -> bool:
        """
        检查是否需要提前终止流程
        """
        # FAQ 匹配成功 → 可以跳过意图分析，直接生成
        if step["task_type"] == "faq_matching" and result.success:
            if result.output.get("faq_matched"):
                context.update_shared_context("skip_intention", True)
                return False  # 继续但跳过某些步骤
        
        # 输入安全检查失败 → 直接终止
        if step["task_type"] == "input_validation" and not result.success:
            return True
        
        # 质量检查失败 → 需要重试（不终止）
        if step["task_type"] == "quality_check" and not result.success:
            context.update_shared_context("need_regenerate", True)
            return False
        
        return False
    
    # =============== 结果汇总 ===============
    
    async def _aggregate_results(
        self, 
        state: AgentState, 
        results: List[TaskResult],
        context: CollaborationContext
    ) -> AgentState:
        """
        汇总各专家结果，更新最终状态
        """
        for result in results:
            if not result.success:
                continue
            
            output = result.output
            
            # 根据任务类型更新状态
            if "intention" in output:
                state.dialog_state.intention = output["intention"]
                state.dialog_state.action = output.get("action")
            
            if "recommend_material" in output:
                state.dialog_state.recommend_material = output["recommend_material"]
            
            if "final_response" in output:
                state.final_response = output["final_response"]
            
            if "is_safe" in output and not output["is_safe"]:
                # 内容不安全，使用兜底回复
                state.final_response = "抱歉，我无法回答这个问题。"
                state.status = -2
        
        return state
    
    # =============== 反思与重规划 ===============
    
    async def _need_replan(
        self, 
        state: AgentState, 
        context: CollaborationContext
    ) -> bool:
        """
        判断是否需要重新规划
        """
        # 质量检查标记需要重新生成
        if context.get_shared_context("need_regenerate", False):
            return True
        
        # 响应为空
        if not state.final_response:
            return True
        
        return False
    
    async def _replan_and_execute(
        self, 
        state: AgentState, 
        context: CollaborationContext
    ) -> AgentState:
        """
        重新规划并执行（用于错误恢复）
        """
        self.logger.info("[Supervisor] 执行重新规划")
        
        # 简化的重试计划：只重新生成
        retry_plan = TaskPlan()
        retry_plan.add_step(
            agent=AgentRole.GENERATOR,
            task_type="response_generation",
            instruction="重新生成回复，避免之前的问题",
        )
        retry_plan.add_step(
            agent=AgentRole.REVIEWER,
            task_type="quality_check",
            instruction="检查重新生成的回复质量"
        )
        
        # 清除重试标记
        context.update_shared_context("need_regenerate", False)
        
        results = await self._execute_plan(state, retry_plan, context)
        return await self._aggregate_results(state, results, context)
    
    # =============== 工具方法 ===============
    
    def get_expert_for_task(self, task_type: str) -> Optional[AgentRole]:
        """获取处理特定任务的专家"""
        return self.task_expert_mapping.get(task_type)
    
    def get_available_experts(self) -> List[AgentRole]:
        """获取所有可用的专家"""
        return list(self.message_bus.get_all_agents().keys())
