"""
Professional Statistics Suite - AI Package (Production)
AI統合と機械学習機能のパッケージ（本番環境用）
"""

# AI関連モジュールのインポート
from .ai_integration import (
    AIOrchestrator,
    QueryProcessor, 
    ContextManager,
    AnalysisContext,
    GoogleProvider,
    OllamaProvider,
    LMStudioProvider,
    KoboldCppProvider
)

from .contextual_retriever import (
    ContextualRetriever,
    RetrievalContext,
    RetrievalResult,
    CacheEntry,
    AnalysisContext,
    DataCharacteristics,
    StatisticalMethod,
    DataType
)

from .gguf_model_selector import (
    GGUFModelSelector,
    create_gguf_selector_dialog
)

from .local_llm_statistical_assistant import (
    LocalLLMStatisticalAssistant,
    StatisticalAnalysisEngine,
    ModelLoader
)

from .demo_statistical_gguf import (
    StatisticalGGUFDemo,
    GGUFModelDemo,
    StatisticalAnalysisDemo
)

# パッケージ情報
__version__ = "2.0.0"
__author__ = "Professional Statistics Suite Team"
__description__ = "AI統合と機械学習機能（本番環境用）"

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
    "LocalLLMStatisticalAssistant",
    "StatisticalAnalysisEngine",
    "ModelLoader",
    
    # Demo modules
    "StatisticalGGUFDemo",
    "GGUFModelDemo",
    "StatisticalAnalysisDemo",
    
    # Package info
    "__version__",
    "__author__",
    "__description__"
]

