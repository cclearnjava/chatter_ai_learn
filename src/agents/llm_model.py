import logging
import string
import urllib
from typing import List

import httpx
import requests
from openai import AsyncOpenAI

from settings.setttings import SettingsManager
from src.agents.chat_generator import LLMLogitProcessor

logger = logging.getLogger(__name__)


class LLMModel:

    def __init__(self, settings: SettingsManager):

        self.settings = settings
        self.llm_model_client = AsyncOpenAI(
            # None refers to the default OpenAI API base URL
            base_url=None if settings.llm.infer_backend == "openai" else settings.llm.vllm_serving_url,
            api_key=settings.llm.api_key,
            http_client=httpx.AsyncClient(
                proxies=settings.server.proxy,
                timeout=10.0,
            ) if settings.llm.use_proxy else None,
            default_headers={
                "Content-Type": "application/json",
                "Authorization": self.settings.llm.vllm_serving_authorization
            } if settings.llm.vllm_serving_authorization else None
        )

        self.llm_async_client = httpx.AsyncClient()

        self.bad_word_ids = []
        self.end_punctuations = [".", "!", "?", "~"]

        self._init_bad_word_ids()

    def tokenize(self, inputs: str) -> List[int]:
        return requests.post(
            url=urllib.parse.urljoin(self.settings.llm.vllm_serving_url, "tokenize"),
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": self.settings.llm.vllm_serving_authorization,
            },
            json={
                "prompt": inputs,
            }).json()['tokens']

    async def async_tokenize(self, inputs: str) -> List[int]:
        response = await self.llm_async_client.post(
            url=urllib.parse.urljoin(self.settings.llm.vllm_serving_url, "tokenize"),
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "Authorization": self.settings.llm.vllm_serving_authorization,
            },
            json={
                "prompt": inputs,
            }
        )
        return response.json()["tokens"]

    def _init_bad_word_ids(self):
        """
        bad word ids, [aa, aaa...ZZ, ZZZ, ZZZZ, ZZZZZ, ZZZZZZ, ZZZZZZZ]
        """
        letters = string.ascii_letters

        combinations = []
        for letter in letters:
            for length in range(2, 7):
                combinations.append(letter * length)

        logger.info(f"LLM bad word combinations: {combinations[:30]}...")

        unique_token_ids = set()

        for combination in combinations:
            token_ids = self.tokenize(combination)
            unique_token_ids.update(token_ids)

        unique_token_ids_list = list(unique_token_ids)

        self.bad_word_ids = unique_token_ids_list
        logger.info(f"LLM bad word ids: {self.bad_word_ids}")

    async def run(self, prompts: str, temperature: float, top_p: float, top_k: int, num_return_sequences: int,
                  repetition_penalty: float = None, max_new_tokens: int = 50, logit_processors: List[str] = None,
                  history: str = None,
                  seed: int = None) -> List[str]:

        logit_bias = {}
        if logit_processors is not None and LLMLogitProcessor.BAD_WORD.value in logit_processors:
            # -100 means 禁止token出现
            logit_bias = {str(token_id): -100 for token_id in self.bad_word_ids}

        if self.settings.llm.infer_backend == "openai":
            response = await self.llm_model_client.chat.completions.create(
                model=self.settings.llm.model_id,
                messages=parse_prompt_to_messages(prompts),
            )
            return [x.message.content for x in response.choices]
        else:
            extra_body = {
                "top_k": top_k,
                "repetition_penalty": repetition_penalty
            }

            if logit_processors is not None and LLMLogitProcessor.NO_REPEAT_NGRAM.value in logit_processors:
                history_token = await self.async_tokenize(history) if history else []
                extra_body.update(
                    {
                        "logits_processors": [
                            {
                                "qualname": "llm.logits_process.NoRepeatNgramLogitsProcessor",
                                "args": [],
                                "kwargs": {"ngram_size": 3, "check_top_k": 10, "history_token": history_token}
                            }
                        ]
                    }
                )

            response = await self.llm_model_client.completions.create(
                model=self.settings.llm.model_id,
                prompt=prompts,
                n=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
                logit_bias=logit_bias,
                seed=seed,
                extra_body=extra_body
            )

            if logit_processors is not None and LLMLogitProcessor.EOS_BOOST.value in logit_processors:
                return [self.eos_post_processor(x.text) for x in response.choices]

            return [x.text for x in response.choices]

    def eos_post_processor(self, input_text: str, threshold: int = 140) -> str:
        """
        threshold: length threshold of chars
        1 token ~ 3.5 chars in Qwen3
        """
        if len(input_text) <= threshold:
            return input_text

        for i in range(threshold, len(input_text)):
            if input_text[i] in self.end_punctuations:
                logger.info("Post-processing -- EOS Boosted")
                return input_text[:i + 1]

        return input_text
