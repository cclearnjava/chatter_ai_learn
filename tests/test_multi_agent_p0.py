import pytest


def make_state(content="hello", creator_profile=None):
    from src.graph.state import AgentState, ChatHistory, ChatMessage

    return AgentState(
        request_id="req_p0",
        message=ChatMessage(role="fan", content=content),
        chat_history=ChatHistory(messages=[{"role": "fan", "content": content}]),
        creator_profile=creator_profile or {},
    )


def test_agent_state_exposes_p0_control_fields():
    state = make_state()

    assert state.retry_count == 0
    assert state.max_retry == 2
    assert state.handoff_required is False
    assert state.handoff_reason == ""
    assert state.trace_id == "req_p0"


def test_multi_agent_workflow_registers_profile_expert():
    from src.multi_agent.base import AgentRole
    from src.multi_agent.workflow import MultiAgentWorkflow

    workflow = MultiAgentWorkflow()

    assert AgentRole.PROFILE_ANALYST in workflow.message_bus.get_all_agents()


@pytest.mark.asyncio
async def test_faq_hit_short_circuits_intention_analysis():
    from src.multi_agent.base import AgentRole, TaskResult
    from src.multi_agent.protocol import MessageBus
    from src.multi_agent.supervisor import SupervisorAgent

    class StaticBus(MessageBus):
        async def send_and_wait(self, message, timeout=None):
            task = message.content["task"]
            task_type = task["task_type"]
            if task_type == "input_validation":
                result = TaskResult(task_id=task["task_id"], success=True, output={"is_safe": True})
            elif task_type == "faq_matching":
                result = TaskResult(
                    task_id=task["task_id"],
                    success=True,
                    output={
                        "faq_matched": True,
                        "faq_answer": "FAQ answer",
                        "confidence": 0.95,
                    },
                )
            elif task_type == "response_generation":
                result = TaskResult(
                    task_id=task["task_id"],
                    success=True,
                    output={"final_response": "FAQ answer"},
                )
            elif task_type == "profile_summary":
                result = TaskResult(
                    task_id=task["task_id"],
                    success=True,
                    output={"known_facts": {}, "unknown_facts": ["age"], "forbidden_facts": ["age"]},
                )
            elif task_type in {"quality_check", "violation_detection"}:
                result = TaskResult(task_id=task["task_id"], success=True, output={"passed": True, "is_safe": True})
            else:
                raise AssertionError(f"FAQ hit should skip {task_type}")

            from src.multi_agent.base import AgentMessage, MessageType

            return AgentMessage(
                message_type=MessageType.TASK_RESULT,
                sender=message.receiver,
                receiver=AgentRole.SUPERVISOR,
                content={"result": result.dict()},
                reply_to=message.message_id,
            )

    supervisor = SupervisorAgent(StaticBus())
    state = await supervisor.process(make_state("价格是多少"))

    assert state.final_response == "FAQ answer"
    assert state.dialog_state.intention is None


@pytest.mark.asyncio
async def test_replan_stops_at_max_retry_and_marks_handoff():
    from src.multi_agent.base import AgentRole, TaskResult
    from src.multi_agent.protocol import MessageBus
    from src.multi_agent.supervisor import SupervisorAgent

    class AlwaysBadReviewBus(MessageBus):
        async def send_and_wait(self, message, timeout=None):
            task = message.content["task"]
            task_type = task["task_type"]
            if task_type == "input_validation":
                result = TaskResult(task_id=task["task_id"], success=True, output={"is_safe": True})
            elif task_type == "response_generation":
                result = TaskResult(
                    task_id=task["task_id"],
                    success=True,
                    output={"final_response": "bad"},
                )
            elif task_type == "quality_check":
                result = TaskResult(
                    task_id=task["task_id"],
                    success=True,
                    output={"passed": False, "need_regenerate": True, "issues": ["too short"]},
                )
            elif task_type == "violation_detection":
                result = TaskResult(task_id=task["task_id"], success=True, output={"is_safe": True})
            else:
                result = TaskResult(task_id=task["task_id"], success=True, output={})

            from src.multi_agent.base import AgentMessage, MessageType

            return AgentMessage(
                message_type=MessageType.TASK_RESULT,
                sender=message.receiver,
                receiver=AgentRole.SUPERVISOR,
                content={"result": result.dict()},
                reply_to=message.message_id,
            )

    state = make_state("hello")
    state.max_retry = 1

    result = await SupervisorAgent(AlwaysBadReviewBus()).process(state)

    assert result.retry_count == 1
    assert result.handoff_required is True
    assert "quality_check_failed" in result.handoff_reason
