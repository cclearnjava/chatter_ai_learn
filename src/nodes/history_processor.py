from typing import Dict, Any, List, Optional
from src.graph.state import AgentState
from src.services.history_service import HistoryService
from src.utils.schema import MessageItem, TextContentItem
from src.utils.constant import DEFAULT_MAX_HISTORY_LENGTH
import logging
import copy

logger = logging.getLogger(__name__)


class HistoryProcessorNode:
    """历史处理器节点（优化版）
    核心能力：
    1. 动态适配多业务场景/chat/cola）
    2. 完善历史拉取容错与降级
    3. 补全ChatHistory所有核心字段到AgentState
    4. 历史长度控制、无效消息过滤
    5. 兼容新增消息的临时存储（为后续持久化做准备）
    """

    def __init__(self):
        self.history_service = HistoryService()
        # 可配置项（建议后续迁移到Nacos/配置文件）
        self.max_history_length = DEFAULT_MAX_HISTORY_LENGTH or 10  # 保留最近N条历史
        self.supported_business_types = ["links", "chat", "cola"]  # 支持的业务类型

    def _filter_invalid_messages(self, messages: Optional[List[Any]]) -> List[Any]:
        """过滤无效消息（空内容、非文本类型、空白字符）"""
        if not messages:
            return []

        valid_messages = []
        for msg in messages:
            # 跳过空消息
            if not msg:
                continue
            # 过滤无内容的MessageItem
            if isinstance(msg, MessageItem):
                if not msg.content:
                    continue
                # 检查content是否有有效文本
                has_valid_content = any(
                    hasattr(item, "text") and item.text.strip()
                    for item in msg.content
                )
                if has_valid_content:
                    valid_messages.append(msg)
            else:
                valid_messages.append(msg)
        return valid_messages

    def _get_business_type(self, state: AgentState) -> str:
        """动态获取业务类型（优先级：business_info.scene > 默认chat）"""
        default_business = "chat"
        if not hasattr(state, "business_info"):
            return default_business

        scene = state.business_info.scene
        return scene if scene in self.supported_business_types else default_business

    def _safe_retrieve_history(self, state: AgentState, business: str) -> tuple[List[Any], Dict[str, Any]]:
        """安全拉取历史记录（带异常处理和降级）"""
        try:
            if business == "links":
                # links业务：从business_info取摘要，历史为空
                summary = copy.deepcopy(state.business_info.summary) if hasattr(state.business_info, "summary") else {}
                return [], summary

            # 非links业务：从历史服务拉取
            creator_id = state.creator_profile.get("id")
            fan_id = state.fan_profile.get("id")

            # ID为空时返回空历史
            if not creator_id or not fan_id:
                logger.warning("Creator ID/Fan ID is empty, return empty history")
                return [], {}

            stored_history, summary = self.history_service.retrieve(creator_id, fan_id)
            logger.info(f"Retrieved {len(stored_history)} history items for creator {creator_id}, fan {fan_id}")
            return stored_history, copy.deepcopy(summary)

        except Exception as e:
            logger.error(f"Retrieve history failed: {str(e)}", exc_info=True)
            # 拉取失败时降级返回空历史
            return [], {}

    def _truncate_history(self, history_list: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """截断历史长度（保留最近N条）"""
        if len(history_list) > self.max_history_length:
            truncated = history_list[-self.max_history_length:]
            logger.info(f"History truncated from {len(history_list)} to {len(truncated)} items")
            return truncated
        return history_list

    def run(self, state: AgentState) -> Dict[str, Any]:
        """处理聊天历史（主逻辑）"""
        logger.info("Starting chat history processing")

        # 1. 基础参数准备
        business = self._get_business_type(state)
        logger.info(f"Current business type: {business}")

        # 2. 安全拉取历史记录
        stored_history, summary = self._safe_retrieve_history(state, business)

        # 3. 过滤无效消息（避免空内容/非文本干扰）
        raw_messages = state.message_item.content if (hasattr(state, "message_item") and state.message_item) else []
        filtered_messages = self._filter_invalid_messages(raw_messages)
        logger.info(f"Filtered messages: {len(filtered_messages)} valid items (raw: {len(raw_messages)})")

        # 4. 构建ChatHistory（兼容所有业务类型）
        chat_history = self.history_service.create_chat_history(
            messages=filtered_messages,
            stored_history=stored_history,
            summary=summary,
            business=business
        )

        # 5. 补全AgentState上下文（覆盖ChatHistory所有核心字段）
        # 核心历史字段
        state.chat_history = chat_history
        state.message = chat_history.get_last_fan_question()  # 最后一个粉丝问题
        state.last_message = chat_history.get_last_message()  # 最后一条完整消息
        state.history_summary = chat_history.get_summary()  # 历史摘要
        state.all_history = chat_history.get_all_history()  # 全量历史（stored+new）

        # 扁平化历史（截断过长内容）
        flatten_history = chat_history.get_flatten_history()
        state.flatten_history = self._truncate_history(flatten_history)

        # 新增历史（为后续持久化准备）
        state.new_history = chat_history.get_new_history()

        # 6. 日志输出关键信息
        logger.info(f"Last fan question: {state.message.content[:50]}..." if state.message.content else "Empty")
        logger.info(f"Flatten history count: {len(state.flatten_history)} (max: {self.max_history_length})")
        logger.info(f"History summary: {list(state.history_summary.keys()) if state.history_summary else 'None'}")

        # 7. 返回需要更新到AgentState的字段
        return {
            "chat_history": chat_history,
            "message": state.message,
            "last_message": state.last_message,
            "history_summary": state.history_summary,
            "all_history": state.all_history,
            "flatten_history": state.flatten_history,
            "new_history": state.new_history
        }