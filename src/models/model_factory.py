from src.models.base_model import BaseLLMModel
from src.models.qwen_model import QwenModel
from src.models.gpt_model import GPTModel
from src.models.ernie_model import ErnieModel
from src.models.grok_model import GrokModel  # 导入 Grok 适配类
from src.ab_test.traffic_allocator import ABTestTrafficAllocator
from typing import Dict, Optional
import yaml
import os

class ModelFactory:
    """模型工厂：支持普通模型创建 + A/B 测试模型创建"""
    def __init__(self, config_path: str = "config/model_config.yaml"):
        # 加载配置
        self.config = self._load_config(config_path)
        self.model_configs = self.config.get("models", {})
        self.ab_test_config = self.config.get("ab_test", {})
        # 初始化 A/B 测试流量分配器
        self.traffic_allocator = ABTestTrafficAllocator(self.ab_test_config) if self.ab_test_config.get("enable") else None

    def _load_config(self, config_path: str) -> Dict:
        """加载模型配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"模型配置文件不存在：{config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _create_model_instance(self, model_key: str) -> BaseLLMModel:
        """创建单个模型实例（新增 Grok 模型支持）"""
        model_config = self.model_configs.get(model_key)
        if not model_config:
            raise ValueError(f"模型配置不存在：{model_key}")
        model_type = model_config.get("type")
        # 根据模型类型创建实例（新增 grok 分支）
        if model_type == "qwen":
            return QwenModel(model_config)
        elif model_type == "gpt":
            return GPTModel(model_config)
        elif model_type == "ernie":
            return ErnieModel(model_config)
        elif model_type == "grok":  # 新增 Grok 模型判断
            return GrokModel(model_config)
        else:
            raise NotImplementedError(f"不支持的模型类型：{model_type}")

    # 其他方法（get_default_model/get_ab_test_model/get_model）不变...