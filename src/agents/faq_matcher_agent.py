from typing import Dict, Any

from settings.setttings import SettingsManager
from src.agents.base_agent import Agent
from src.graph.state import AgentState
import logging

logger = logging.getLogger(__name__)


class FaqMatcherAgent(Agent):


    async def run(self, state: AgentState) -> Dict[str, Any]:
        """执行FAQ匹配"""
        logger.info("Starting FAQ matching")

        # 1. 获取用户最后一条问题
        user_question = state.message.content.strip() if state.message else ""
        if not user_question:
            state.faq_matched = False
            return {"faq_matched": False}

        # 2. 核心逻辑：匹配FAQ（示例：实际需替换为你的FAQ库查询逻辑）
        faq_lib = {
            "如何购买会员": "会员购买链接：xxx，价格xxx",
            "提现多久到账": "提现后24小时内到账，节假日顺延"
        }
        matched = False
        for faq_question in faq_lib.keys():
            if faq_question in user_question:
                # 匹配到FAQ，将结果存入state
                state.faq_answer = faq_lib[faq_question]
                matched = True
                break

        # 3. 设置匹配结果到state
        state.faq_matched = matched
        logger.info(f"FAQ matching result: {'matched' if matched else 'not matched'}")

        # 4. 返回需要更新到state的字段
        return {
            "faq_matched": matched,
            "faq_answer": state.faq_answer if matched else ""
        }