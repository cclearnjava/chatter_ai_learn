from typing import Dict, Any
from src.graph.state import AgentState
from settings import get_nacos_manager
import logging

logger = logging.getLogger(__name__)


class InputValidatorNode:
    """输入验证节点"""

    def __init__(self):
        self.nacos_manager = get_nacos_manager()

    def run(self, state: AgentState) -> Dict[str, Any]:
        """验证输入参数"""
        logger.info("Validating input parameters")

        # 检查创建者和粉丝档案
        if not state.creator_profile or not state.fan_profile:
            logger.error("Input creator_profile or fan_profile have empty, check!")
            return {
                "status": -1,  # PARAMS_ERROR_STATUS
                "error_message": "Request parameters error",
                "messages": [{"type": "error", "content": "Missing creator or fan profile"}]
            }

        # 检查创建者ID
        creator_id = state.creator_profile.get("id")
        if not creator_id:
            logger.error("creator_id is empty, check!")
            return {
                "status": -1,
                "error_message": "Request parameters error",
                "messages": [{"type": "error", "content": "Missing creator ID"}]
            }

        # 检查粉丝ID
        fan_id = state.fan_profile.get("id")
        if not fan_id:
            logger.error("fan_id is empty, check!")
            return {
                "status": -1,
                "error_message": "Request parameters error",
                "messages": [{"type": "error", "content": "Missing fan ID"}]
            }

        # 检查创建者是否有效
        if creator_id not in self.nacos_manager.creator_config.keys():
            logger.error(f"creator_id {creator_id} not valid, check!")
            return {
                "status": -1,
                "error_message": "Request parameters error",
                "messages": [{"type": "error", "content": "Invalid creator ID"}]
            }

        logger.info("Input validation passed")
        return {
            "status": 0,  # SUCCESS_STATUS
            "messages": [{"type": "info", "content": "Input validation passed"}]
        }