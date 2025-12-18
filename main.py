# main.py
import asyncio

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, status, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from src.graph.state import AgentState, ChatMessage, MessageItem, BusinessInfo
from src.graph.workflow import ChatWorkflow
from src.core.flow import AutoChatFlow
from settings.settings import get_settings
from src.utils.logging_config import setup_logging
from src.utils.trace import TraceMiddleware
from src.utils.timing import apply_timing_decorators
import logging
import time

setup_logging()
logger = logging.getLogger("auto_chat")
settings = get_settings()

# 应用LLM调用时间记录装饰器
apply_timing_decorators()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await app.state.auto_chat_flow.nacos_manager.load_initial_config()
        logger.info("初始配置加载成功")

        config_watcher_task = asyncio.create_task(
            app.state.auto_chat_flow.nacos_manager.setup_config_watchers()
        )
        app.state.config_watcher_task = config_watcher_task
        logger.info("配置热更新监听器设置成功")
    except Exception as e:
        logger.error(f"启动配置初始化失败: {e}")
        raise e

    yield

    try:
        if hasattr(app.state, 'config_watcher_task'):
            app.state.config_watcher_task.cancel()
            try:
                await app.state.config_watcher_task
            except asyncio.CancelledError:
                pass

        await app.state.auto_chat_flow.nacos_manager.cleanup()
        logger.info("应用关闭，资源清理完成")
    except Exception as e:
        logger.error(f"资源清理失败: {e}")


def create_app():
    app = FastAPI(lifespan=lifespan)

    # 初始化核心组件
    # AutoChatFlow：负责基础服务
    # ChatWorkflow：负责工作流编排
    # AgentState：负责状态管理
    auto_chat_flow = AutoChatFlow(settings=settings)
    chat_workflow = ChatWorkflow()

    app.state.auto_chat_flow = auto_chat_flow
    app.state.chat_workflow = chat_workflow

    # 配置CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 在生产环境中限制为特定域名
        allow_credentials=True,
        allow_methods=["POST"],  # 只允许POST方法
        allow_headers=["*"],  # 允许所有请求头
    )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(
            f"{request.client.host}:{request.client.port} - \"{request.method} {request.url.path} HTTP/{request.scope.get('http_version', '1.1')}\" {response.status_code} - {process_time:.4f}s")
        return response

    app.add_middleware(TraceMiddleware)  # trace 中间件

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.error(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors()},
        )

    @app.post("/chat")
    async def handle_messages(request_data: dict):
        """处理聊天请求"""
        try:
            # 解析请求数据
            messages = [MessageItem(**msg) for msg in request_data.get("messages", [])]
            business_info = BusinessInfo(**request_data.get("business_info", {}))
            mode = request_data.get("mode", settings.server.mode)

            # 创建初始状态
            initial_state = AgentState(
                request_id=request_data.get("request_id", "default"),
                message=ChatMessage(role="fan", content=""),  # 从消息中获取
                message_item=messages[-1] if messages else None,
                chat_history=auto_chat_flow.create_chat_history(messages),
                creator_profile=business_info.creator_profile,
                fan_profile=business_info.fan_profile,
                mode=mode,
                scene=business_info.scene,
                business_info=business_info
            )

            # 运行工作流
            result_state = await chat_workflow.run(initial_state)

            # 构建响应
            response = {
                "id": result_state.request_id,
                "status": result_state.status,
                "error_message": result_state.error_message,
                "response": [{"role": "creator", "content": [{"type": "text", "text": result_state.final_response}]}],
                "record_data": result_state.get_formatted_record_data()
            }

            return response

        except Exception as e:
            logger.error(f"Error handling V3 chat request: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": str(e)}
            )

    @app.get("/health")
    async def check_health():
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"health": "ok", "status": "application is running"},
        )

    logger.info("Startup successful!")
    return app


if __name__ == '__main__':
    app = create_app()
    uvicorn.run(app, host=settings.server.host, port=settings.server.port, log_config=None, log_level="debug",
                access_log=False)