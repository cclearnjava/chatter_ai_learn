from typing import Dict, Any
from src.graph.state import AgentState
from settings.nacos_manager import get_nacos_manager
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class ContextBuilderNode:
    """上下文构建节点"""

    def __init__(self):
        self.nacos_manager = get_nacos_manager()

    async def run(self, state: AgentState) -> Dict[str, Any]:
        """构建和更新上下文"""
        logger.info("Building context")

        # 更新创建者档案（从Nacos配置）
        if hasattr(self.nacos_manager, 'creator_profile') and state.creator_profile.get("id"):
            creator_id = state.creator_profile["id"]
            if creator_id in self.nacos_manager.creator_profile:
                state.creator_profile = deepcopy(
                    self.nacos_manager.creator_profile[creator_id]["profile"]
                )
                state.creator_script = deepcopy(
                    self.nacos_manager.creator_config[creator_id]["script"]
                )
                logger.info(f"Updated creator profile: {state.creator_profile}, script: {state.creator_script}")

        # 更新对话状态的任务完成原因
        state.dialog_state.task_completed_reason = state.business_info.task_complete_reason

        logger.info("Context built successfully")
        return {
            "creator_profile": state.creator_profile,
            "creator_script": state.creator_script,
            "dialog_state": state.dialog_state
        }