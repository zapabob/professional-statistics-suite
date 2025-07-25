"""
Professional Statistics Suite - AI Package
AI統合と機械学習機能のパッケージ
"""

# AI関連モジュールのインポート
from src.ai.ai_integration import (
    AIOrchestrator,
    QueryProcessor, 
    ContextManager,
    AnalysisContext,
    GoogleProvider,
    OllamaProvider,
    LMStudioProvider,
    KoboldCppProvider
)

from src.ai.contextual_retriever import (
    ContextualRetriever,
    RetrievalContext,
    RetrievalResult,
    CacheEntry,
    AnalysisContext,
    DataCharacteristics,
    StatisticalMethod,
    DataType
)

from src.ai.gguf_model_selector import (
    GGUFModelSelector,
    create_gguf_selector_dialog
)

from src.ai.local_llm_statistical_assistant import (
    GGUFModelConfig,
    StatisticalQuery,
    StatisticalResponse,
    LocalLLMStatisticalAssistant,
    StatisticalAssistantDemo
)

from src.ai.demo_statistical_gguf import (
    demonstrate_gguf_integration,
    interactive_gguf_session
)

# パッケージ情報
__version__ = "2.0.0"
__author__ = "Professional Statistics Suite Team"
__description__ = "AI統合と機械学習機能"

# 利用可能なクラスと関数のリスト
__all__ = [
    # AI Integration
    "AIOrchestrator",
    "QueryProcessor", 
    "ContextManager",
    "AnalysisContext",
    "GoogleProvider",
    "OllamaProvider",
    "LMStudioProvider",
    "KoboldCppProvider",
    
    # Contextual Retriever
    "ContextualRetriever",
    "RetrievalContext",
    "RetrievalResult",
    "CacheEntry",
    "AnalysisContext",
    "DataCharacteristics",
    "StatisticalMethod",
    "DataType",
    
    # GGUF Model Selector
    "GGUFModelSelector",
    "create_gguf_selector_dialog",
    
    # Local LLM Statistical Assistant
    "GGUFModelConfig",
    "StatisticalQuery",
    "StatisticalResponse",
    "LocalLLMStatisticalAssistant",
    "StatisticalAssistantDemo",
    
    # Demo modules
    "demonstrate_gguf_integration",
    "interactive_gguf_session",
    
    # Package info
    "__version__",
    "__author__",
    "__description__"
]

