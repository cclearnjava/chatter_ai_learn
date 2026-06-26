# src/multi_agent/experts/profile.py
"""
画像专家 Agent

P0 职责:
- 汇总 creator/fan 已知画像事实
- 标记缺失或禁止编造的强事实字段
- 为生成与质检链路提供事实约束
"""

from typing import Any, Dict, List
import time

from src.multi_agent.base import AgentRole, ExpertAgent, TaskAssignment, TaskResult


class ProfileExpert(ExpertAgent):
    """画像专家，负责强事实字段的读取和缺失判断。"""

    role = AgentRole.PROFILE_ANALYST
    name = "profile_expert"
    description = "分析 creator/fan 画像，提供强事实约束"

    capabilities = [
        "profile_summary",
        "profile_fact_check",
        "persona_constraints",
    ]

    FACT_FIELDS = ("age", "gender", "location", "birthday", "name")
    PRIVATE_FIELDS = ("age", "gender", "birthday")

    async def execute(self, task: TaskAssignment) -> TaskResult:
        start_time = time.time()

        try:
            state_data = task.context.get("state", {})
            creator_profile = self._merge_profile(
                state_data.get("business_info", {}).get("creator_profile", {}),
                state_data.get("creator_profile", {}),
            )
            fan_profile = self._merge_profile(
                state_data.get("business_info", {}).get("fan_profile", {}),
                state_data.get("fan_profile", {}),
            )

            known_facts = {
                key: value
                for key, value in creator_profile.items()
                if key in self.FACT_FIELDS and value not in (None, "", [], {})
            }
            unknown_facts = [
                key for key in self.FACT_FIELDS if key not in known_facts
            ]
            forbidden_facts = [
                key for key in self.PRIVATE_FIELDS if key in unknown_facts
            ]

            return TaskResult(
                task_id=task.task_id,
                success=True,
                output={
                    "creator_profile": creator_profile,
                    "fan_profile": fan_profile,
                    "known_facts": known_facts,
                    "unknown_facts": unknown_facts,
                    "forbidden_facts": forbidden_facts,
                    "profile_constraints": {
                        "do_not_invent_fields": forbidden_facts,
                        "answer_unknown_with_fallback": True,
                    },
                },
                confidence=1.0,
                execution_time=time.time() - start_time,
            )
        except Exception as exc:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(exc),
                retryable=False,
                handoff_required=True,
                risk_level="medium",
                reason="profile_fact_extraction_failed",
                execution_time=time.time() - start_time,
            )

    def _merge_profile(self, *profiles: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for profile in profiles:
            if isinstance(profile, dict):
                merged.update({k: v for k, v in profile.items() if v not in (None, "")})
        return merged
