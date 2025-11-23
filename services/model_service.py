"""
模型服务类 - 提供统一的模型创建和管理功能
"""

from google.adk.models.lite_llm import LiteLlm


class ModelService:
    """模型服务类，提供统一的模型创建和管理"""

    # 默认配置
    DEFAULT_API_BASE = "http://localhost:11434/v1"
    DEFAULT_PROVIDER = "openai"
    DEFAULT_API_KEY = "ollama"

    # 可用的模型列表
    AVAILABLE_MODELS = [
        "qwen3:30b-a3b-instruct-2507-fp16",
        "qwen3-coder:30b-a3b-fp16",
        "deepseek-v3.1:671b-cloud",
        "qwen3-coder:480b-cloud",
        "kimi-k2:1t-cloud",
        "qwen3:30b",
        "qwen3-coder:30b",
        "gpt-oss:20b"
    ]

    def __init__(self, api_base: str = None, provider: str = None, api_key: str = None):
        """
        初始化模型服务

        Args:
            api_base: API基础URL，默认为Ollama的OpenAI兼容接口
            provider: LLM提供商，默认为openai
            api_key: API密钥，Ollama不需要真实密钥但需要提供值
        """
        self.api_base = api_base or self.DEFAULT_API_BASE
        self.provider = provider or self.DEFAULT_PROVIDER
        self.api_key = api_key or self.DEFAULT_API_KEY

    def create_model(self, model_name: str) -> LiteLlm:
        """
        创建LiteLlm模型实例

        Args:
            model_name: 模型名称，必须是AVAILABLE_MODELS中的模型

        Returns:
            LiteLlm: 配置好的模型实例

        Raises:
            ValueError: 如果模型名称不在可用模型列表中
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"模型 '{model_name}' 不在可用模型列表中。可用模型: {', '.join(self.AVAILABLE_MODELS)}")

        return LiteLlm(
            model=model_name,
            api_base=self.api_base,
            custom_llm_provider=self.provider,
            api_key=self.api_key
        )

    def get_available_models(self) -> list:
        """获取可用模型列表"""
        return self.AVAILABLE_MODELS.copy()

    def add_model(self, model_name: str):
        """添加新的模型到可用列表"""
        if model_name not in self.AVAILABLE_MODELS:
            self.AVAILABLE_MODELS.append(model_name)

    def remove_model(self, model_name: str):
        """从可用列表中移除模型"""
        if model_name in self.AVAILABLE_MODELS:
            self.AVAILABLE_MODELS.remove(model_name)


# 创建默认的模型服务实例
model_service = ModelService()