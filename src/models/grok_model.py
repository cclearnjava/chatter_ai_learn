from src.models.base_model import BaseLLMModel, ChatMessage, ModelResponse
from typing import List, Dict, Any, Optional
import aiohttp
import logging
import time

logger = logging.getLogger("auto_chat")

class GrokModel(BaseLLMModel):
    """Grok 模型适配类（支持 grok-1/grok-1-beta）"""
    def __init__(self, model_config: Dict[str, Any]):
        # 从配置读取核心信息
        self.api_key = model_config["api_key"]
        self.api_base = model_config.get("api_base", "https://api.x.ai/v1")
        self.model_name = model_config["model_name"]  # grok-1/grok-1-beta
        # 默认生成参数（与 OpenAI 格式兼容，Grok 支持大部分 OpenAI 聊天参数）
        self.default_params = {
            "temperature": model_config.get("temperature", 0.7),
            "top_p": model_config.get("top_p", 0.8),
            "max_tokens": model_config.get("max_tokens", 4096),
            "stop": model_config.get("stop", None),
            "stream": False  # Grok 支持流式，此处默认关闭（如需流式可扩展）
        }
        # Grok 鉴权头（与 OpenAI 一致，使用 Bearer Token）
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def generate(self, messages: List[ChatMessage], generation_params: Optional[Dict[str, Any]] = None) -> ModelResponse:
        """
        异步调用 Grok 模型生成回复（兼容统一接口）
        :param messages: 统一格式的对话历史
        :param generation_params: 动态生成参数（优先级高于默认配置）
        :return: 统一格式的 ModelResponse
        """
        # 1. 合并默认参数与动态参数
        final_params = {**self.default_params, **(generation_params or {})}
        # 2. 构造 Grok API 请求体（与 OpenAI Chat API 格式完全兼容）
        request_body = {
            "model": self.model_name,
            "messages": [msg.dict() for msg in messages],
            **final_params
        }
        # 3. 构造 API 地址
        api_url = f"{self.api_base}/chat/completions"
        logger.debug(f"Grok 模型请求体：{request_body}，API 地址：{api_url}")

        try:
            # 4. 发送异步请求
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.post(
                    url=api_url,
                    headers=self.headers,
                    json=request_body,
                    timeout=aiohttp.ClientTimeout(total=30)  # 设置超时时间
                ) as resp:
                    response_time = time.time() - start_time
                    raw_response = await resp.json()

                    # 5. 处理响应状态码
                    if resp.status != 200:
                        error_msg = f"Grok 模型调用失败（状态码：{resp.status}）：{raw_response.get('error', {}).get('message', '未知错误')}"
                        logger.error(error_msg)
                        return ModelResponse(
                            success=False,
                            content=None,
                            usage=None,
                            error_msg=error_msg,
                            raw_response=raw_response
                        )

                    # 6. 解析响应结果（与 OpenAI 格式一致）
                    content = raw_response["choices"][0]["message"]["content"]
                    usage = raw_response.get("usage", {})  # 包含 prompt_tokens/completion_tokens/total_tokens
                    logger.info(f"Grok 模型调用成功，耗时：{response_time:.2f}s，Token 用量：{usage}")

                    # 7. 返回统一格式结果
                    return ModelResponse(
                        success=True,
                        content=content,
                        usage=usage,
                        error_msg=None,
                        raw_response=raw_response
                    )

        except aiohttp.ClientTimeoutError:
            error_msg = "Grok 模型调用超时（超过 30 秒）"
            logger.error(error_msg)
            return ModelResponse(
                success=False,
                content=None,
                usage=None,
                error_msg=error_msg,
                raw_response=None
            )
        except Exception as e:
            error_msg = f"Grok 模型调用异常：{str(e)}"
            logger.error(error_msg)
            return ModelResponse(
                success=False,
                content=None,
                usage=None,
                error_msg=error_msg,
                raw_response=None
            )

    def get_model_name(self) -> str:
        """返回模型标识名称（用于日志和埋点）"""
        return f"grok-{self.model_name}"