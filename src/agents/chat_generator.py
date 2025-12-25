from typing import Dict, Any
from src.graph.state import AgentState
from src.services.llm_service import LLMService
from src.services.prompt_service import PromptService
from src.utils.helpers import remove_placeholder, trim_excess_emojis
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LLMLogitProcessor(Enum):
    BAD_WORD = "bad_word"
    TOKEN_LENGTH = "token_length"
    NO_REPEAT_NGRAM = "no_repeat_ngram"
    NO_REPEAT_TOKEN = "no_repeat_token"
    EOS_BOOST = "eos_boost"


class ChatGeneratorAgent:
    """聊天生成智能体"""

    def __init__(self):
        self.llm_service = LLMService()
        self.prompt_service = PromptService()
        self.template = self.prompt_service.get_template("infloww")  # 使用infloww模板

    async def run(self, state: AgentState) -> Dict[str, Any]:
        """生成聊天响应"""
        logger.info("Running chat generation")

        # 获取历史记录和当前消息
        chat_history = state.chat_history.get_flatten_history()
        message = state.message

        # 根据动作生成不同的提示
        chat_prompt = self.prompt_service.get_chat_prompt(
            self.template,
            message,
            chat_history,
            state.creator_profile,
            state.chat_history.summary,
            state.dialog_state.action,
            ppv=state.dialog_state.recommend_material[0] if state.dialog_state.recommend_material else None,
        )

        async def _chat_run():
            return await self.llm_service.run(
                prompts=chat_prompt,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                num_return_sequences=1,
                repetition_penalty=1.1,
                max_new_tokens=200,
                logit_processors=[LLMLogitProcessor.BAD_WORD.value, LLMLogitProcessor.EOS_BOOST.value,
                                  LLMLogitProcessor.NO_REPEAT_NGRAM.value],
                history=self.prompt_service.format_conversation_in_prompt(chat_history),
            )

        chat_responses = await _chat_run()

        # 移除重复响应
        chat_responses = self.remove_repeated_response(chat_responses, chat_history)

        if not chat_responses:
            chat_responses = await _chat_run()
            logger.info(f"re-run chat prompt:\n{chat_prompt}")
            logger.info(f"re-run chat responses:\n{chat_responses}")

        response = chat_responses[0] if chat_responses else ""

        # 处理响应
        response = trim_excess_emojis(response)
        response = remove_placeholder(response)

        # 更新对话状态
        state.dialog_state.response = response

        logger.info(f"Chat generation completed: {response[:50]}...")
        return {
            "dialog_state": state.dialog_state,
            "final_response": response
        }

    def remove_repeated_response(self, responses: list[str], chat_history: list[dict[str, str]]) -> list[str]:
        """移除重复响应"""
        from src.utils.helpers import remove_repeated_response as helper_remove_repeated
        return helper_remove_repeated(responses, chat_history)