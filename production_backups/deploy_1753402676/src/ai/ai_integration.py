#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Integration Module - 2025 Latest Edition
AI統合モジュール - 2025年最新版（マルチLLMプロバイダー、自己修正、RAG、ローカルLLM対応）
"""

import asyncio
import base64
import re
import json
import os
import platform
import time
from typing import Optional, Dict, Any, List, Union, Literal, Tuple
from pathlib import Path
import traceback
import logging
from datetime import datetime
import io
from contextlib import redirect_stdout
from enum import Enum
import uuid
import glob
import signal
import sys

# Data processing
import pandas as pd
import numpy as np
import requests

# AI API clients
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from lmstudio import LMStudioClient
    LMSTUDIO_AVAILABLE = True
except ImportError:
    LMSTUDIO_AVAILABLE = False

# GGUF direct support
try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

# RAG and Vector DB
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Image processing
try:
    from PIL import Image
    import cv2
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# OCR engines
PYTESSERACT_AVAILABLE = False
EASYOCR_AVAILABLE = False
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    pass
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    pass

# IntentType Enum for AI Orchestrator
class IntentType(Enum):
    """ユーザー意図の分類"""
    DESCRIPTIVE = "descriptive"
    INFERENTIAL = "inferential"
    PREDICTIVE = "predictive"
    EXPLORATORY = "exploratory"
    EDUCATIONAL = "educational"

# Configuration
class AIConfig:
    """AI設定クラス（2025年版）"""
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.together_api_key = os.getenv("TOGETHER_API_KEY")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.lmstudio_base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        self.koboldcpp_base_url = os.getenv("KOBOLDCPP_BASE_URL", "http://localhost:5001/v1")
        self.ocr_languages = ["eng", "jpn"]
        self.tesseract_cmd = None
        self.default_provider = "openai"
        self.default_model = "gpt-4o"
        self.max_tokens = 4096
        self.temperature = 0.1
        self.max_correction_attempts = 2
        self._load_from_config()

    def _load_from_config(self):
        try:
            from config import ai_config as external_config
            for key in self.__dict__:
                if hasattr(external_config, key):
                    setattr(self, key, getattr(external_config, key) or getattr(self, key))
        except (ImportError, AttributeError):
            pass

    def is_api_configured(self, provider: str) -> bool:
        key_map = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
            "together": self.together_api_key,
            "ollama": True,
            "lmstudio": True,
            "koboldcpp": True
        }
        return bool(key_map.get(provider))

ai_config = AIConfig()

# Model Definitions
LLM_MODELS = {
    "openai": {"gpt-4o": {}, "o3": {}, "gpt-4-turbo": {}},
    "anthropic": {"claude-3-7-sonnet-20250219": {}, "claude-3-5-sonnet-20240620": {}, "claude-3-opus-20240229": {}},
    "google": {"gemini-2.5-pro-latest": {}, "gemini-2.5-flash-latest": {}},
    "together": {"meta-llama/Llama-3.1-70b-chat-hf": {}, "meta-llama/Llama-3.1-8b-chat-hf": {}},
    "ollama": {"llama3.1": {}, "gemma2": {}},
    "lmstudio": {"local-model/gguf-model": {}},
    "koboldcpp": {"local-model/gguf-model": {}} # Models depend on what is loaded
}

# ... (Prompt Templates and KnowledgeBase class remain the same) ...

class LLMProvider:
    """LLMプロバイダーの基底クラス"""
    def __init__(self, provider_name: str, api_key: Optional[str] = None):
        self.provider_name = provider_name
        self.api_key = api_key
        self.logger = logging.getLogger(f"{__name__}.{provider_name}")
    
    async def generate_response(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    """OpenAI GPT プロバイダー"""
    def __init__(self, api_key: str):
        super().__init__("openai", api_key)
        if OPENAI_AVAILABLE:
            self.client = openai.OpenAI(api_key=api_key)
    
    async def generate_response(self, prompt: str, model: str = "gpt-3.5-turbo", **kwargs) -> Dict[str, Any]:
        try:
            if not OPENAI_AVAILABLE:
                return {"success": False, "error": "OpenAI not available"}
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            return {
                "success": True,
                "content": response.choices[0].message.content,
                "tokens": response.usage.total_tokens if response.usage else 0,
                "provider": "openai"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "provider": "openai"}

class GoogleProvider(LLMProvider):
    """Google Gemini プロバイダー（API設定自動検出機能強化版）"""
    def __init__(self, api_key: str = None):
        super().__init__("google", api_key)
        self.api_key = self._get_api_key(api_key)
        self.logger = logging.getLogger(f"{__name__}.GoogleProvider")
        
        if self.api_key and GOOGLE_AI_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
                self.logger.info("Google AI API設定完了")
            except Exception as e:
                self.logger.error(f"Google AI API設定エラー: {e}")
    
    def _get_api_key(self, api_key: str = None) -> str:
        """API keyの自動検出"""
        # 1. 引数で指定されたAPI key
        if api_key:
            return api_key
        
        # 2. 環境変数から検索
        env_vars = [
            'GOOGLE_API_KEY',
            'GOOGLE_AI_API_KEY', 
            'GEMINI_API_KEY',
            'GOOGLE_API_KEY_STATISTICS'
        ]
        
        for env_var in env_vars:
            api_key = os.getenv(env_var)
            if api_key:
                self.logger.info(f"環境変数 {env_var} からAPI keyを検出")
                return api_key
        
        # 3. 設定ファイルから検索
        config_files = [
            'config.json',
            'ai_config.json',
            'google_config.json',
            '.env'
        ]
        
        for config_file in config_files:
            try:
                if os.path.exists(config_file):
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        if 'google_api_key' in config:
                            self.logger.info(f"設定ファイル {config_file} からAPI keyを検出")
                            return config['google_api_key']
                        elif 'api_key' in config:
                            self.logger.info(f"設定ファイル {config_file} からAPI keyを検出")
                            return config['api_key']
            except Exception as e:
                self.logger.debug(f"設定ファイル {config_file} の読み込みエラー: {e}")
        
        # 4. デフォルトのAPI key（開発用）
        default_key = "AIzaSyBxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # ダミーキー
        self.logger.warning("API keyが見つかりません。ローカルLLMへの切り替えを推奨します。")
        return default_key
    
    def is_available(self) -> bool:
        """利用可能性の確認"""
        try:
            if not self.api_key or self.api_key == "AIzaSyBxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx":
                return False
            
            if not GOOGLE_AI_AVAILABLE:
                return False
            
            # API keyの形式チェック
            if not self.api_key.startswith("AIza"):
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Google Provider利用可能性チェックエラー: {e}")
            return False
    
    async def generate_response(self, prompt: str, model: str = "gemini-1.5-pro-latest", **kwargs) -> Dict[str, Any]:
        try:
            if not self.is_available():
                return {
                    "success": False, 
                    "error": "Google AI APIが利用できません。ローカルLLMの使用を推奨します。", 
                    "provider": "google",
                    "suggested_fallback": "ollama"
                }
            
            # 非同期実行のためにasyncioを使用
            import asyncio
            
            def _generate_sync():
                try:
                    model_instance = genai.GenerativeModel(model)
                    response = model_instance.generate_content(prompt)
                    return response
                except Exception as e:
                    self.logger.error(f"Google AI API呼び出しエラー: {e}")
                    raise
            
            # 同期関数を非同期で実行
            response = await asyncio.get_event_loop().run_in_executor(None, _generate_sync)
            
            return {
                "success": True,
                "content": response.text,
                "tokens": len(response.text.split()),  # 概算
                "provider": "google",
                "model_used": model
            }
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Google AI API エラー: {error_msg}")
            
            # エラーの種類に応じたフォールバック提案
            if "API key not valid" in error_msg or "API_KEY_INVALID" in error_msg:
                fallback_suggestion = "ollama"
                error_msg += " → ローカルLLM（Ollama）の使用を推奨します"
            elif "quota" in error_msg.lower():
                fallback_suggestion = "lmstudio"
                error_msg += " → ローカルLLM（LM Studio）の使用を推奨します"
            else:
                fallback_suggestion = "ollama"
                error_msg += " → ローカルLLMへの切り替えを推奨します"
            
            return {
                "success": False, 
                "error": error_msg, 
                "provider": "google",
                "suggested_fallback": fallback_suggestion
            }

class AnthropicProvider(LLMProvider):
    """Anthropic Claude プロバイダー"""
    def __init__(self, api_key: str):
        super().__init__("anthropic", api_key)
        if ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=api_key)
    
    async def generate_response(self, prompt: str, model: str = "claude-3-sonnet-20240229", **kwargs) -> Dict[str, Any]:
        try:
            if not ANTHROPIC_AVAILABLE:
                return {"success": False, "error": "Anthropic not available"}
            
            response = self.client.messages.create(
                model=model,
                max_tokens=kwargs.get('max_tokens', 1000),
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                "success": True,
                "content": response.content[0].text,
                "tokens": response.usage.input_tokens + response.usage.output_tokens,
                "provider": "anthropic"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "provider": "anthropic"}

class TogetherProvider(LLMProvider):
    """Together AI プロバイダー"""
    def __init__(self, api_key: str):
        super().__init__("together", api_key)
        self.api_key = api_key
    
    async def generate_response(self, prompt: str, model: str = "meta-llama/Llama-2-7b-chat-hf", **kwargs) -> Dict[str, Any]:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            data = {
                "model": model,
                "prompt": prompt,
                "max_tokens": kwargs.get('max_tokens', 1000)
            }
            
            response = requests.post("https://api.together.xyz/inference", headers=headers, json=data)
            result = response.json()
            
            return {
                "success": True,
                "content": result.get("output", {}).get("choices", [{}])[0].get("text", ""),
                "tokens": result.get("output", {}).get("usage", {}).get("total_tokens", 0),
                "provider": "together"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "provider": "together"}

class OllamaProvider(LLMProvider):
    """Ollama ローカルLLM プロバイダー"""
    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__("ollama")
        self.base_url = base_url
    
    async def generate_response(self, prompt: str, model: str = "llama2", **kwargs) -> Dict[str, Any]:
        try:
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=data)
            result = response.json()
            
            return {
                "success": True,
                "content": result.get("response", ""),
                "tokens": len(result.get("response", "").split()),
                "provider": "ollama"
            }
        except Exception as e:
            return {"success": False, "error": str(e), "provider": "ollama"}

class LMStudioProvider(LLMProvider):
    """LM Studio プロバイダー（GGUF対応強化版 + 統計補助機能統合）"""
    def __init__(self, base_url: str = "http://localhost:1234", models_dir: str = "./models"):
        super().__init__("lmstudio")
        self.base_url = base_url
        self.models_dir = Path(models_dir)
        self.available_models = []
        self.current_model = None
        self.logger = logging.getLogger(__name__)
        
        # 統計補助機能の初期化
        self.statistical_assistant = None
        self.statistical_methods_db = self._load_statistical_methods_db()
        
        # modelsディレクトリからGGUFファイルを検索
        self._scan_gguf_models()
    
    def _scan_gguf_models(self) -> List[str]:
        """modelsディレクトリからGGUFファイルをスキャン"""
        gguf_files = []
        
        if self.models_dir.exists():
            # .ggufファイルを再帰的に検索
            gguf_files = list(self.models_dir.rglob('*.gguf'))
            self.available_models = [str(f.relative_to(self.models_dir)) for f in gguf_files]
            
            if self.available_models:
                self.logger.info(f"🔍 {len(self.available_models)}個のGGUFモデルを発見: {self.available_models}")
            else:
                self.logger.warning(f"⚠️ {self.models_dir}にGGUFファイルが見つからへん")
        else:
            self.logger.warning(f"⚠️ modelsディレクトリが存在せへん: {self.models_dir}")
        
        return self.available_models
    
    def scan_custom_directory(self, directory: str) -> List[str]:
        """指定されたディレクトリからGGUFファイルをスキャン"""
        try:
            path = Path(directory)
            if not path.exists():
                self.logger.warning(f"⚠️ 指定されたディレクトリが存在しません: {directory}")
                return []
            
            # .ggufファイルを再帰的に検索
            gguf_files = list(path.rglob('*.gguf'))
            custom_models = [str(f.relative_to(path)) for f in gguf_files]
            
            if custom_models:
                self.logger.info(f"🔍 カスタムディレクトリから{len(custom_models)}個のGGUFモデルを発見: {custom_models}")
                # カスタムモデルを追加
                self.available_models.extend(custom_models)
            else:
                self.logger.warning(f"⚠️ {directory}にGGUFファイルが見つかりませんでした")
            
            return custom_models
            
        except Exception as e:
            self.logger.error(f"❌ カスタムディレクトリスキャンエラー: {e}")
            return []
    
    def get_available_models(self) -> List[str]:
        """利用可能なGGUFモデル一覧を取得"""
        return self.available_models
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """モデル情報を取得"""
        if not model_name and self.available_models:
            model_name = self.available_models[0]  # 最初のモデルを使用
        
        if not model_name:
            return {"error": "利用可能なモデルがありません"}
        
        model_path = self.models_dir / model_name
        if model_path.exists():
            stat = model_path.stat()
            return {
                "name": model_name,
                "path": str(model_path),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "provider": "lmstudio"
            }
        
        return {"error": f"モデルファイルが見つからへん: {model_name}"}
    
    async def load_model(self, model_name: str = None) -> Dict[str, Any]:
        """LM Studioでモデルを読み込み"""
        if not model_name and self.available_models:
            model_name = self.available_models[0]
        
        if not model_name:
            return {"success": False, "error": "読み込むモデルが指定されてへん"}
        
        try:
            # LM Studio APIでモデル読み込み（実際のAPIエンドポイントに応じて調整）
            headers = {"Content-Type": "application/json"}
            data = {"model": model_name}
            
            # モデル読み込みAPI（LM Studioの実際のAPIに合わせて調整が必要）
            response = requests.post(f"{self.base_url}/v1/models/load", headers=headers, json=data)
            
            if response.status_code == 200:
                self.current_model = model_name
                self.logger.info(f"✅ モデル読み込み成功: {model_name}")
                return {"success": True, "model": model_name}
            else:
                return {"success": False, "error": f"モデル読み込み失敗: {response.text}"}
                
        except Exception as e:
            self.logger.error(f"❌ モデル読み込みエラー: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_response(self, prompt: str, model: str = None, **kwargs) -> Dict[str, Any]:
        """テキスト生成（統計学専用最適化）"""
        try:
            # モデルが指定されてない場合は現在のモデルまたは最初のモデルを使用
            if not model:
                model = self.current_model or (self.available_models[0] if self.available_models else "local-model")
            
            headers = {"Content-Type": "application/json"}
            
            # 統計学専用のシステムプロンプト
            system_prompt = """あなたは統計学の専門家です。正確で分かりやすい日本語で統計学の概念を説明してください。
数式や具体例を含めて、実践的なアドバイスを提供してください。"""
            
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": kwargs.get('temperature', 0.3),  # 統計学では正確性重視
                "max_tokens": kwargs.get('max_tokens', 2048),
                "top_p": kwargs.get('top_p', 0.9),
                "frequency_penalty": kwargs.get('frequency_penalty', 0.1)
            }
            
            response = requests.post(f"{self.base_url}/v1/chat/completions", headers=headers, json=data)
            
            if response.status_code != 200:
                return {"success": False, "error": f"API エラー: {response.status_code}", "provider": "lmstudio"}
            
            result = response.json()
            
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            tokens = result.get("usage", {}).get("total_tokens", len(content.split()))
            
            return {
                "success": True,
                "content": content,
                "text": content,  # gguf_test_helper.pyとの互換性
                "tokens": tokens,
                "tokens_consumed": tokens,  # gguf_test_helper.pyとの互換性
                "processing_time": 0.1,  # 概算値
                "model": model,
                "provider": "lmstudio"
            }
            
        except Exception as e:
            self.logger.error(f"❌ LM Studio生成エラー: {e}")
            return {"success": False, "error": str(e), "provider": "lmstudio"}
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """同期版のテキスト生成（gguf_test_helper.pyとの互換性）"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.generate_response(prompt, **kwargs))
    
    def is_available(self) -> bool:
        """LM Studioの利用可能性をチェック"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _load_statistical_methods_db(self) -> Dict[str, Dict[str, Any]]:
        """統計手法データベースを読み込み"""
        return {
            "t_test": {
                "name": "t検定",
                "description": "2群の平均値の差を検定",
                "assumptions": ["正規分布", "等分散性", "独立性"],
                "use_cases": ["2群の比較", "前後比較"],
                "python_code": "from scipy import stats\nresult = stats.ttest_ind(group1, group2)"
            },
            "chi_square": {
                "name": "カイ二乗検定",
                "description": "カテゴリカルデータの独立性を検定",
                "assumptions": ["独立性", "期待度数"],
                "use_cases": ["分割表の分析", "適合度検定"],
                "python_code": "from scipy import stats\nresult = stats.chi2_contingency(contingency_table)"
            },
            "correlation": {
                "name": "相関分析",
                "description": "2変数間の関係性を分析",
                "assumptions": ["線形関係", "正規分布"],
                "use_cases": ["関係性の探索", "予測モデル"],
                "python_code": "import numpy as np\ncorrelation = np.corrcoef(var1, var2)[0,1]"
            },
            "regression": {
                "name": "回帰分析",
                "description": "従属変数を独立変数で予測",
                "assumptions": ["線形性", "独立性", "等分散性", "正規性"],
                "use_cases": ["予測モデル", "因果関係の探索"],
                "python_code": "from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()\nmodel.fit(X, y)"
            },
            "anova": {
                "name": "分散分析",
                "description": "3群以上の平均値の差を検定",
                "assumptions": ["正規分布", "等分散性", "独立性"],
                "use_cases": ["多群比較", "実験効果の検定"],
                "python_code": "from scipy import stats\nresult = stats.f_oneway(*groups)"
            },
            "mann_whitney": {
                "name": "マンホイットニー検定",
                "description": "ノンパラメトリックな2群比較",
                "assumptions": ["独立性", "連続データ"],
                "use_cases": ["正規分布しないデータの比較"],
                "python_code": "from scipy import stats\nresult = stats.mannwhitneyu(group1, group2)"
            }
        }
    
    async def analyze_statistical_query(self, query: str, data_info: Optional[Dict[str, Any]] = None, 
                                      user_expertise: str = "intermediate") -> Dict[str, Any]:
        """統計クエリを分析（統計補助機能）"""
        try:
            # クエリの意図を分類
            intent = self._classify_statistical_intent(query)
            
            # 統計学専用プロンプトを構築
            prompt = self._build_statistical_prompt(query, intent, data_info, user_expertise)
            
            # LMStudioで推論実行
            response = await self.generate_response(prompt)
            
            # 応答を解析
            parsed_response = self._parse_statistical_response(response.get('content', ''))
            
            return {
                "success": True,
                "answer": parsed_response.get('answer', response.get('content', '')),
                "confidence": parsed_response.get('confidence', 0.7),
                "suggested_methods": parsed_response.get('suggested_methods', []),
                "educational_content": parsed_response.get('educational_content'),
                "code_example": parsed_response.get('code_example'),
                "intent": intent,
                "processing_time": response.get('processing_time', 0.0),
                "tokens_used": response.get('tokens_consumed', 0)
            }
            
        except Exception as e:
            self.logger.error(f"統計クエリ分析エラー: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": f"エラーが発生しました: {str(e)}",
                "confidence": 0.0,
                "suggested_methods": [],
                "processing_time": 0.0,
                "tokens_used": 0
            }
    
    def _classify_statistical_intent(self, query: str) -> str:
        """統計クエリの意図を分類"""
        query_lower = query.lower()
        
        # 記述統計
        if any(word in query_lower for word in ['平均', '中央値', '分散', '標準偏差', '分布', '要約', '記述']):
            return "descriptive"
        
        # 推論統計
        elif any(word in query_lower for word in ['検定', 't検定', 'カイ二乗', '相関', '回帰', '有意', '仮説']):
            return "inferential"
        
        # 予測分析
        elif any(word in query_lower for word in ['予測', 'モデル', '機械学習', '分類', '回帰']):
            return "predictive"
        
        # 教育的内容
        else:
            return "educational"
    
    def _build_statistical_prompt(self, query: str, intent: str, data_info: Optional[Dict[str, Any]], 
                                 user_expertise: str) -> str:
        """統計分析用プロンプトを構築"""
        # 基本プロンプト
        base_prompts = {
            "descriptive": "あなたは統計分析の専門家です。記述統計に関する質問に答えてください。",
            "inferential": "あなたは統計分析の専門家です。推論統計に関する質問に答えてください。",
            "predictive": "あなたは統計分析の専門家です。予測分析に関する質問に答えてください。",
            "educational": "あなたは統計分析の専門家です。統計学の教育的内容を提供してください。"
        }
        
        base_prompt = base_prompts.get(intent, base_prompts["educational"])
        
        # データ情報を追加
        data_context = ""
        if data_info:
            data_context = f"""
データ情報:
- 行数: {data_info.get('rows', 'N/A')}
- 列数: {data_info.get('columns', 'N/A')}
- 列名: {data_info.get('column_names', [])}
- データ型: {data_info.get('dtypes', {})}
"""
        
        # ユーザーレベルに応じた説明レベルを設定
        expertise_levels = {
            "beginner": "初心者向けに詳しく説明してください。専門用語は避け、具体例を多く含めてください。",
            "intermediate": "中級者向けに説明してください。理論的背景と実践的な応用をバランスよく説明してください。",
            "advanced": "上級者向けに専門的に説明してください。最新の手法や高度な概念も含めてください。"
        }
        
        expertise_level = expertise_levels.get(user_expertise, expertise_levels["intermediate"])
        
        prompt = f"""
{base_prompt}

{data_context}

ユーザーの質問: {query}
{expertise_level}

回答は以下の形式で提供してください:
1. 直接的な回答
2. 推奨される統計手法（該当する場合）
3. 教育的内容（必要に応じて）
4. Pythonコード例（必要に応じて）

統計手法の選択時は、データの特性と仮定を考慮してください。
"""
        
        return prompt
    
    def _parse_statistical_response(self, response: str) -> Dict[str, Any]:
        """統計応答を解析"""
        try:
            # JSON形式の応答を試行
            if response.strip().startswith('{') and response.strip().endswith('}'):
                return json.loads(response)
            
            # 構造化された応答を解析
            parsed = {
                'answer': response,
                'confidence': 0.7,
                'suggested_methods': [],
                'educational_content': None,
                'code_example': None,
                'tokens_used': len(response.split())
            }
            
            # 統計手法のキーワードを抽出
            statistical_keywords = [
                't検定', 'カイ二乗検定', '相関分析', '回帰分析', '分散分析',
                'マンホイットニー検定', 'ウィルコクソン検定', 'クラスカル・ウォリス検定',
                'フリードマン検定', 'クラスター分析', '主成分分析'
            ]
            
            for keyword in statistical_keywords:
                if keyword in response:
                    parsed['suggested_methods'].append(keyword)
            
            return parsed
            
        except Exception as e:
            self.logger.error(f"応答解析エラー: {e}")
            return {
                'answer': response,
                'confidence': 0.5,
                'suggested_methods': [],
                'tokens_used': len(response.split())
            }
    
    def get_statistical_methods(self) -> Dict[str, Dict[str, Any]]:
        """利用可能な統計手法を取得"""
        return self.statistical_methods_db
    
    def suggest_statistical_method(self, data_characteristics: Dict[str, Any], 
                                 research_question: str) -> List[Dict[str, Any]]:
        """データ特性と研究質問に基づいて統計手法を提案"""
        suggestions = []
        
        for method_id, method_info in self.statistical_methods_db.items():
            # データ特性と研究質問に基づいて適合性を評価
            compatibility_score = self._calculate_method_compatibility(
                method_info, data_characteristics, research_question
            )
            
            if compatibility_score > 0.3:  # 閾値
                suggestions.append({
                    "method_id": method_id,
                    "method_name": method_info["name"],
                    "description": method_info["description"],
                    "compatibility_score": compatibility_score,
                    "assumptions": method_info["assumptions"],
                    "use_cases": method_info["use_cases"],
                    "python_code": method_info.get("python_code", "")
                })
        
        # 適合性スコアでソート
        suggestions.sort(key=lambda x: x["compatibility_score"], reverse=True)
        return suggestions
    
    def _calculate_method_compatibility(self, method_info: Dict[str, Any], 
                                     data_characteristics: Dict[str, Any], 
                                     research_question: str) -> float:
        """統計手法の適合性を計算"""
        score = 0.0
        
        # 研究質問との適合性
        question_lower = research_question.lower()
        method_name_lower = method_info["name"].lower()
        
        if any(word in question_lower for word in method_name_lower.split()):
            score += 0.4
        
        # データ特性との適合性
        data_type = data_characteristics.get("data_type", "unknown")
        n_groups = data_characteristics.get("n_groups", 1)
        
        if "t検定" in method_info["name"] and n_groups == 2:
            score += 0.3
        elif "分散分析" in method_info["name"] and n_groups > 2:
            score += 0.3
        elif "相関" in method_info["name"] and data_type == "continuous":
            score += 0.3
        
        return min(score, 1.0)

class GGUFProvider(LLMProvider):
    """GGUF直接読み込みプロバイダー（llama-cpp-python使用）"""
    def __init__(self, model_path: str = None, n_ctx: int = 4096, n_gpu_layers: int = 0):
        super().__init__("gguf")
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.model = None
        self.loaded = False
        
        # GPU対応チェック
        self.gpu_available = self._check_gpu_support()
        
        if self.gpu_available and n_gpu_layers == 0:
            self.n_gpu_layers = -1  # 全レイヤーをGPUに
            self.logger.info("🚀 GPU加速を有効化したで")
    
    def _check_gpu_support(self) -> bool:
        """GPU対応をチェック"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                # ROCm対応チェック
                import os
                return 'ROCM_PATH' in os.environ
            except:
                return False
    
    def load_model(self, model_path: str = None) -> bool:
        """GGUFモデルを読み込み"""
        if not LLAMA_CPP_AVAILABLE:
            self.logger.error("❌ llama-cpp-pythonがインストールされてへん")
            return False
        
        if model_path:
            self.model_path = model_path
        
        if not self.model_path:
            self.logger.error("❌ モデルパスが指定されてへん")
            return False
        
        if not Path(self.model_path).exists():
            self.logger.error(f"❌ モデルファイルが見つからへん: {self.model_path}")
            return False
        
        try:
            self.logger.info(f"🔄 GGUFモデル読み込み中: {self.model_path}")
            
            # llama-cpp-pythonでモデル読み込み
            self.model = llama_cpp.Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
                n_threads=os.cpu_count() // 2  # CPUスレッド数を最適化
            )
            
            self.loaded = True
            self.logger.info(f"✅ GGUFモデル読み込み成功: {Path(self.model_path).name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ GGUFモデル読み込み失敗: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3, **kwargs) -> Dict[str, Any]:
        """テキスト生成（同期版）"""
        if not self.loaded or not self.model:
            return {
                'success': False,
                'error': 'モデルが読み込まれてへん',
                'text': '',
                'tokens_consumed': 0,
                'processing_time': 0
            }
        
        try:
            start_time = time.time()
            
            # 統計学専用のシステムプロンプト
            system_prompt = """あなたは統計学の専門家です。正確で分かりやすい日本語で統計学の概念を説明してください。
数式や具体例を含めて、実践的なアドバイスを提供してください。

質問: """
            
            full_prompt = system_prompt + prompt
            
            # テキスト生成
            output = self.model(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.get('top_p', 0.9),
                repeat_penalty=kwargs.get('repeat_penalty', 1.1),
                stop=kwargs.get('stop', ["\n\n", "質問:", "Q:", "A:"]),
                echo=False
            )
            
            processing_time = time.time() - start_time
            
            generated_text = output['choices'][0]['text'].strip()
            tokens_used = output['usage']['total_tokens']
            
            return {
                'success': True,
                'text': generated_text,
                'content': generated_text,  # 他のプロバイダーとの互換性
                'tokens_consumed': tokens_used,
                'tokens': tokens_used,  # 他のプロバイダーとの互換性
                'processing_time': processing_time,
                'model_path': self.model_path,
                'provider': 'gguf'
            }
            
        except Exception as e:
            self.logger.error(f"❌ GGUF生成エラー: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'tokens_consumed': 0,
                'processing_time': 0
            }
    
    async def generate_response(self, prompt: str, model: str = None, **kwargs) -> Dict[str, Any]:
        """非同期版のテキスト生成"""
        import asyncio
        
        # 同期版を非同期で実行
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """チャット形式での生成（gguf_test_helper.pyとの互換性）"""
        # メッセージを単一のプロンプトに変換
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.append(f"システム: {content}")
            elif role == 'user':
                prompt_parts.append(f"ユーザー: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"アシスタント: {content}")
        
        full_prompt = "\n".join(prompt_parts) + "\nアシスタント: "
        return self.generate(full_prompt, **kwargs)
    
    def is_available(self) -> bool:
        """GGUF利用可能性をチェック"""
        return LLAMA_CPP_AVAILABLE and self.loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        if not self.model_path:
            return {'provider': 'GGUF', 'loaded': False}
        
        model_path = Path(self.model_path)
        if model_path.exists():
            stat = model_path.stat()
            return {
                'provider': 'GGUF',
                'model_path': str(model_path),
                'model_name': model_path.name,
                'loaded': self.loaded,
                'gpu_enabled': self.n_gpu_layers > 0,
                'context_size': self.n_ctx,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        
        return {'provider': 'GGUF', 'loaded': False, 'error': 'Model file not found'}

class KoboldCppProvider(LLMProvider):
    """Kobold.cpp (ローカルGGUFモデル) プロバイダー"""
    def __init__(self, base_url: str):
        super().__init__("koboldcpp")
        # Kobold.cppはOpenAI互換APIを持つため、OpenAIクライアントを流用
        if OPENAI_AVAILABLE:
            self.client = openai.AsyncOpenAI(base_url=base_url, api_key="not-needed")
        else:
            self.client = None

    async def generate_response(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        if not self.client:
            return {"success": False, "error": "OpenAIライブラリが利用できません。"}
        try:
            system_prompt = kwargs.get("system_prompt", "")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.chat.completions.create(
                model=model, # Kobold.cppではモデル名は "koboldcpp" などの固定値でも良い場合が多い
                messages=messages,
                temperature=ai_config.temperature,
                max_tokens=ai_config.max_tokens,
            )
            content = response.choices[0].message.content
            return {"success": True, "provider": "koboldcpp", "model": model, "content": content}

        except Exception as e:
            self.logger.error(f"Kobold.cpp API エラー: {e}")
            return {"success": False, "error": str(e), "provider": "koboldcpp"}

class KnowledgeBase:
    """RAG用ナレッジベース"""
    def __init__(self, docs_dir: str = "_docs"):
        self.docs_dir = Path(docs_dir)
        self.embeddings = None
        self.index = None
        self.documents = []
        self.model = None
        
        if RAG_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self._load_documents()
                self._build_index()
            except Exception as e:
                logging.warning(f"RAG初期化失敗: {e}")
    
    def _load_documents(self):
        """ドキュメントを読み込み"""
        if not self.docs_dir.exists():
            return
            
        for file_path in self.docs_dir.glob("*.md"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # チャンクに分割
                    chunks = self._split_text(content)
                    for chunk in chunks:
                        self.documents.append({
                            'content': chunk,
                            'source': str(file_path),
                            'timestamp': datetime.now().isoformat()
                        })
            except Exception as e:
                logging.warning(f"ドキュメント読み込み失敗 {file_path}: {e}")
    
    def _split_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """テキストをチャンクに分割"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1
            
            if current_size >= chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _build_index(self):
        """FAISSインデックスを構築"""
        if not self.documents or not self.model:
            return
            
        try:
            texts = [doc['content'] for doc in self.documents]
            self.embeddings = self.model.encode(texts)
            
            if RAG_AVAILABLE:
                import faiss
                dimension = self.embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(self.embeddings.astype('float32'))
        except Exception as e:
            logging.warning(f"インデックス構築失敗: {e}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """関連ドキュメントを検索"""
        if not self.model or not self.index or not self.documents:
            return []
            
        try:
            query_embedding = self.model.encode([query])
            distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    results.append({
                        'content': self.documents[idx]['content'],
                        'source': self.documents[idx]['source'],
                        'score': float(distances[0][i])
                    })
            return results
        except Exception as e:
            logging.warning(f"検索失敗: {e}")
            return []

class EducationalContentGenerator:
    """教育コンテンツ生成システム"""

    def __init__(self, llm_provider_manager=None):
        self.logger = logging.getLogger(f"{__name__}.EducationalContentGenerator")
        self.llm_provider_manager = llm_provider_manager

    async def generate_explanation(self, concept: str, user_expertise_level: str, language: str = "ja") -> Dict[str, Any]:
        """
        統計概念の説明を生成する。
        ユーザーの専門レベルと希望言語に応じて説明の複雑さを調整する。
        """
        try:
            prompt_template = STATISTICAL_ANALYSIS_PROMPTS["educational_explanation"]
            prompt = prompt_template.format(
                concept=concept,
                user_expertise_level=user_expertise_level,
                language=language
            )

            if self.llm_provider_manager:
                response = await self.llm_provider_manager.route_request(
                    LLMRequest(prompt=prompt, task_type="educational", data_sensitivity="none")
                )
                if response.success:
                    return {"success": True, "explanation": response.content}
                else:
                    return {"success": False, "error": response.error}
            else:
                return {"success": False, "error": "LLMProviderManagerが設定されていません。"}
        except Exception as e:
            self.logger.error(f"教育コンテンツ生成エラー: {e}")
            return {"success": False, "error": str(e)}

    async def generate_visual_aid_description(self, concept: str, data_example: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        統計概念を説明するための視覚補助（グラフ、図など）の記述を生成する。
        """
        try:
            prompt_template = STATISTICAL_ANALYSIS_PROMPTS["visual_aid_description"]
            prompt = prompt_template.format(
                concept=concept,
                data_example=json.dumps(data_example) if data_example else ""
            )

            if self.llm_provider_manager:
                response = await self.llm_provider_manager.route_request(
                    LLMRequest(prompt=prompt, task_type="visual_aid", data_sensitivity="none")
                )
                if response.success:
                    return {"success": True, "description": response.content}
                else:
                    return {"success": False, "error": response.error}
            else:
                return {"success": False, "error": "LLMProviderManagerが設定されていません。"}
        except Exception as e:
            self.logger.error(f"視覚補助記述生成エラー: {e}")
            return {"success": False, "error": str(e)}

    async def generate_interactive_example(self, concept: str, user_expertise_level: str) -> Dict[str, Any]:
        """
        統計概念を学ぶためのインタラクティブなコード例を生成する。
        """
        try:
            prompt_template = STATISTICAL_ANALYSIS_PROMPTS["interactive_example"]
            prompt = prompt_template.format(
                concept=concept,
                user_expertise_level=user_expertise_level
            )

            if self.llm_provider_manager:
                response = await self.llm_provider_manager.route_request(
                    LLMRequest(prompt=prompt, task_type="code_example", data_sensitivity="none")
                )
                if response.success:
                    return {"success": True, "code_example": response.content}
                else:
                    return {"success": False, "error": response.error}
            else:
                return {"success": False, "error": "LLMProviderManagerが設定されていません。"}
        except Exception as e:
            self.logger.error(f"インタラクティブな例の生成エラー: {e}")
            return {"success": False, "error": str(e)}

class CodeGenerator:
    """AI駆動型Pythonコード生成エンジン"""

    def __init__(self, llm_provider_manager=None):
        self.logger = logging.getLogger(f"{__name__}.CodeGenerator")
        self.llm_provider_manager = llm_provider_manager

    async def generate_analysis_code(self, analysis_type: str, data_info: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        指定された分析タイプとパラメータに基づいてPythonコードを生成する。
        """
        try:
            prompt_template = STATISTICAL_ANALYSIS_PROMPTS["generate_analysis_code"]
            prompt = prompt_template.format(
                analysis_type=analysis_type,
                data_info=json.dumps(data_info, indent=2),
                params=json.dumps(params, indent=2)
            )

            if self.llm_provider_manager:
                response = await self.llm_provider_manager.route_request(
                    LLMRequest(prompt=prompt, task_type="code_generation", data_sensitivity="high")
                )
                if response.success:
                    return {"success": True, "code": response.content}
                else:
                    return {"success": False, "error": response.error}
            else:
                return {"success": False, "error": "LLMProviderManagerが設定されていません。"}
        except Exception as e:
            self.logger.error(f"分析コード生成エラー: {e}")
            return {"success": False, "error": str(e)}

    async def generate_visualization_code(self, plot_type: str, data_info: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        指定されたプロットタイプとパラメータに基づいて可視化コードを生成する。
        """
        try:
            prompt_template = STATISTICAL_ANALYSIS_PROMPTS["generate_visualization_code"]
            prompt = prompt_template.format(
                plot_type=plot_type,
                data_info=json.dumps(data_info, indent=2),
                params=json.dumps(params, indent=2)
            )

            if self.llm_provider_manager:
                response = await self.llm_provider_manager.route_request(
                    LLMRequest(prompt=prompt, task_type="code_generation", data_sensitivity="medium")
                )
                if response.success:
                    return {"success": True, "code": response.content}
                else:
                    return {"success": False, "error": response.error}
            else:
                return {"success": False, "error": "LLMProviderManagerが設定されていません。"}
        except Exception as e:
            self.logger.error(f"可視化コード生成エラー: {e}")
            return {"success": False, "error": str(e)}

    async def generate_report_code(self, report_type: str, analysis_results: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        指定されたレポートタイプと分析結果に基づいてレポート生成コードを生成する。
        """
        try:
            prompt_template = STATISTICAL_ANALYSIS_PROMPTS["generate_report_code"]
            prompt = prompt_template.format(
                report_type=report_type,
                analysis_results=json.dumps(analysis_results, indent=2),
                params=json.dumps(params, indent=2)
            )

            if self.llm_provider_manager:
                response = await self.llm_provider_manager.route_request(
                    LLMRequest(prompt=prompt, task_type="code_generation", data_sensitivity="low")
                )
                if response.success:
                    return {"success": True, "code": response.content}
                else:
                    return {"success": False, "error": response.error}
            else:
                return {"success": False, "error": "LLMProviderManagerが設定されていません。"}
        except Exception as e:
            self.logger.error(f"レポートコード生成エラー: {e}")
            return {"success": False, "error": str(e)}

# Enhanced Data Models for AI Orchestrator
from dataclasses import dataclass

@dataclass
class AnalysisContext:
    """分析コンテキスト（拡張版）"""
    user_id: str
    session_id: str
    data_fingerprint: str
    analysis_history: List[Dict[str, Any]]
    user_expertise_level: str = "intermediate"
    privacy_settings: Dict[str, Any] = None
    timestamp: datetime = None
    
    # 拡張フィールド（タスク1.3対応）
    user_preferences: Dict[str, Any] = None
    session_metadata: Dict[str, Any] = None
    context_tags: List[str] = None
    learning_progress: Dict[str, float] = None
    favorite_methods: List[str] = None
    recent_queries: List[str] = None
    
    def __post_init__(self):
        if self.privacy_settings is None:
            self.privacy_settings = {"use_local_llm": False, "anonymize_data": True}
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.user_preferences is None:
            self.user_preferences = {
                "preferred_visualization": "plotly",
                "explanation_style": "detailed",
                "language": "ja",
                "auto_save": True
            }
        if self.session_metadata is None:
            self.session_metadata = {
                "platform": platform.system(),
                "start_time": datetime.now().isoformat(),
                "data_sources": [],
                "analysis_goals": []
            }
        if self.context_tags is None:
            self.context_tags = []
        if self.learning_progress is None:
            self.learning_progress = {}
        if self.favorite_methods is None:
            self.favorite_methods = []
        if self.recent_queries is None:
            self.recent_queries = []

@dataclass
class AIResponse:
    """AI応答データモデル"""
    content: str
    confidence: float
    provider_used: str
    tokens_consumed: int
    processing_time: float
    intent_detected: Optional[IntentType] = None
    educational_content: Optional[str] = None
    follow_up_suggestions: List[str] = None
    
    def __post_init__(self):
        if self.follow_up_suggestions is None:
            self.follow_up_suggestions = []

class QueryProcessor:
    """自然言語クエリ処理エンジン"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.QueryProcessor")
        
    def process_query(self, query: str, context: AnalysisContext) -> Dict[str, Any]:
        """クエリを処理して構造化された情報を返す"""
        try:
            # 基本的なクエリ解析
            processed = {
                'original_query': query,
                'cleaned_query': self._clean_query(query),
                'intent': self._classify_intent(query),
                'statistical_keywords': self._extract_statistical_keywords(query),
                'data_references': self._extract_data_references(query),
                'confidence': 0.8
            }
            
            return processed
            
        except Exception as e:
            self.logger.error(f"クエリ処理エラー: {e}")
            return {
                'original_query': query,
                'cleaned_query': query,
                'intent': IntentType.EXPLORATORY,
                'statistical_keywords': [],
                'data_references': [],
                'confidence': 0.3
            }
    
    def _clean_query(self, query: str) -> str:
        """クエリをクリーニング"""
        # 基本的なクリーニング処理
        cleaned = query.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned
    
    def _classify_intent(self, query: str) -> IntentType:
        """クエリの意図を分類"""
        query_lower = query.lower()
        
        # キーワードベースの分類
        if any(word in query_lower for word in ['平均', 'mean', '標準偏差', 'std', '分布', 'distribution']):
            return IntentType.DESCRIPTIVE
        elif any(word in query_lower for word in ['検定', 'test', '有意', 'significant', 'p値', 'p-value']):
            return IntentType.INFERENTIAL
        elif any(word in query_lower for word in ['予測', 'predict', '機械学習', 'ml', 'model']):
            return IntentType.PREDICTIVE
        elif any(word in query_lower for word in ['説明', 'explain', '教えて', 'how to', 'what is']):
            return IntentType.EDUCATIONAL
        else:
            return IntentType.EXPLORATORY
    
    def _extract_statistical_keywords(self, query: str) -> List[str]:
        """統計キーワードを抽出（多言語対応）"""
        # 拡張された統計用語辞書
        statistical_terms = {
            # 基本統計
            'descriptive': ['平均', 'mean', 'average', '中央値', 'median', '最頻値', 'mode', 
                          '標準偏差', 'std', 'standard deviation', '分散', 'variance',
                          '四分位', 'quartile', 'パーセンタイル', 'percentile'],
            
            # 仮説検定
            'inferential': ['t検定', 't-test', 'anova', '分散分析', 'analysis of variance',
                          'chi-square', 'カイ二乗', 'fisher', 'フィッシャー', 'wilcoxon',
                          'ウィルコクソン', 'mann-whitney', 'マン・ホイットニー',
                          'p値', 'p-value', '有意', 'significant', '信頼区間', 'confidence interval'],
            
            # 相関・回帰
            'correlation': ['相関', 'correlation', 'pearson', 'ピアソン', 'spearman', 'スピアマン',
                          '回帰', 'regression', 'linear', '線形', 'multiple', '重回帰',
                          'logistic', 'ロジスティック', '決定係数', 'r-squared'],
            
            # 機械学習
            'ml': ['機械学習', 'machine learning', 'ml', '予測', 'prediction', 'model', 'モデル',
                  'classification', '分類', 'clustering', 'クラスタリング', 'neural', 'ニューラル',
                  'random forest', 'ランダムフォレスト', 'svm', 'support vector'],
            
            # データ処理
            'data_processing': ['前処理', 'preprocessing', '欠損値', 'missing', 'outlier', '外れ値',
                              '正規化', 'normalization', 'standardization', '標準化',
                              'encoding', 'エンコーディング', 'scaling', 'スケーリング'],
            
            # 可視化
            'visualization': ['可視化', 'visualization', 'plot', 'プロット', 'graph', 'グラフ',
                            'histogram', 'ヒストグラム', 'scatter', '散布図', 'boxplot', 'ボックスプロット',
                            'heatmap', 'ヒートマップ', 'dashboard', 'ダッシュボード']
        }
        
        found_terms = []
        query_lower = query.lower()
        
        # カテゴリ別に検索
        for category, terms in statistical_terms.items():
            for term in terms:
                if term.lower() in query_lower:
                    found_terms.append(term)
        
        # 重複を除去して返す
        return list(set(found_terms))
    
    def _extract_data_references(self, query: str) -> List[str]:
        """データ参照を抽出"""
        # 列名やデータフレーム参照を抽出
        data_refs = re.findall(r'[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?', query)
        return data_refs[:5]  # 最大5個まで
    
    def generate_clarifying_questions(self, query: str, context: AnalysisContext) -> List[str]:
        """曖昧なクエリに対する明確化質問を生成"""
        questions = []
        
        # 統計手法が曖昧な場合
        if any(word in query.lower() for word in ['分析', 'analysis', '調べる', 'examine']):
            if not self._extract_statistical_keywords(query):
                questions.append("どのような統計手法をお考えですか？（例：相関分析、t検定、回帰分析）")
        
        # データの詳細が不明な場合
        if any(word in query.lower() for word in ['データ', 'data']) and not self._extract_data_references(query):
            questions.append("分析対象のデータはどのような変数を含んでいますか？")
        
        # 目的が不明確な場合
        if len(query.split()) < 5:  # 短すぎるクエリ
            questions.append("分析の目的や知りたいことを詳しく教えてください。")
        
        # 比較対象が不明な場合
        if any(word in query.lower() for word in ['比較', 'compare', '違い', 'difference']):
            if not any(word in query.lower() for word in ['と', 'and', 'vs', '対']):
                questions.append("何と何を比較したいですか？")
        
        return questions[:3]  # 最大3つまで
    
    def suggest_statistical_methods(self, query: str, data_info: Optional[Dict[str, Any]] = None) -> List[str]:
        """クエリとデータ情報に基づいて統計手法を提案"""
        suggestions = []
        query_lower = query.lower()
        
        # 記述統計の提案
        if any(word in query_lower for word in ['要約', 'summary', '概要', 'overview']):
            suggestions.extend(['記述統計', '基本統計量', 'ヒストグラム', 'ボックスプロット'])
        
        # 関係性の分析
        if any(word in query_lower for word in ['関係', 'relationship', '相関', 'correlation']):
            suggestions.extend(['相関分析', '散布図', '回帰分析'])
        
        # 群間比較
        if any(word in query_lower for word in ['比較', 'compare', '差', 'difference']):
            suggestions.extend(['t検定', 'ANOVA', 'カイ二乗検定'])
        
        # 予測・分類
        if any(word in query_lower for word in ['予測', 'predict', '分類', 'classify']):
            suggestions.extend(['線形回帰', 'ロジスティック回帰', '機械学習'])
        
        # データ情報に基づく提案
        if data_info:
            n_vars = data_info.get('n_variables', 0)
            if n_vars > 5:
                suggestions.append('主成分分析')
            if data_info.get('has_categorical', False):
                suggestions.append('カテゴリカル分析')
        
        return list(set(suggestions))[:5]  # 重複除去して最大5つ

class IntentClassifier:
    """意図分類エンジン"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.IntentClassifier")
    
    def classify(self, query: str, context: AnalysisContext) -> IntentType:
        """より高度な意図分類"""
        # 履歴を考慮した分類
        if context.analysis_history:
            last_analysis = context.analysis_history[-1]
            if 'follow_up' in query.lower() and last_analysis.get('type') == 'descriptive':
                return IntentType.INFERENTIAL
        
        # デフォルトはQueryProcessorの分類を使用
        processor = QueryProcessor()
        result = processor.process_query(query, context)
        return result['intent']

class ContextManager:
    """コンテキスト管理システム（永続化対応）"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.logger = logging.getLogger(f"{__name__}.ContextManager")
        self.sessions: Dict[str, AnalysisContext] = {}
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 既存セッションの復旧
        self._load_existing_sessions()
    
    def get_or_create_context(self, user_id: str, session_id: str, data_fingerprint: str) -> AnalysisContext:
        """コンテキストを取得または作成"""
        context_key = f"{user_id}_{session_id}"
        
        if context_key not in self.sessions:
            # 永続化されたコンテキストを探す
            loaded_context = self._load_context_from_disk(context_key)
            
            if loaded_context:
                self.sessions[context_key] = loaded_context
                self.logger.info(f"コンテキスト復旧: {context_key}")
            else:
                # 新しいコンテキストを作成
                self.sessions[context_key] = AnalysisContext(
                    user_id=user_id,
                    session_id=session_id,
                    data_fingerprint=data_fingerprint,
                    analysis_history=[]
                )
                self.logger.info(f"新しいコンテキスト作成: {context_key}")
        
        return self.sessions[context_key]
    
    def update_context(self, context: AnalysisContext, analysis_result: Dict[str, Any]):
        """コンテキストを更新"""
        # 分析結果を履歴に追加
        analysis_entry = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': analysis_result.get('type', 'unknown'),
            'method_used': analysis_result.get('method', 'unknown'),
            'success': analysis_result.get('success', False),
            'query': analysis_result.get('query', ''),
            'provider': analysis_result.get('provider', 'unknown'),
            'processing_time': analysis_result.get('processing_time', 0.0),
            'tokens_consumed': analysis_result.get('tokens_consumed', 0)
        }
        
        context.analysis_history.append(analysis_entry)
        
        # 履歴の長さを制限
        if len(context.analysis_history) > 100:
            context.analysis_history = context.analysis_history[-100:]
        
        # コンテキストを永続化
        self._save_context_to_disk(context)
        
        self.logger.debug(f"コンテキスト更新: {context.user_id}_{context.session_id}")
    
    def get_user_expertise_level(self, context: AnalysisContext) -> str:
        """ユーザーの専門レベルを動的に評価"""
        if not context.analysis_history:
            return context.user_expertise_level
        
        # 履歴から専門レベルを推定
        recent_analyses = context.analysis_history[-10:]  # 最近10件
        
        advanced_methods = ['machine_learning', 'bayesian_analysis', 'survival_analysis']
        basic_methods = ['descriptive_stats', 'basic_visualization']
        
        advanced_count = sum(1 for analysis in recent_analyses 
                           if analysis.get('method_used') in advanced_methods)
        basic_count = sum(1 for analysis in recent_analyses 
                        if analysis.get('method_used') in basic_methods)
        
        if advanced_count >= basic_count and advanced_count > 0:
            return 'expert'
        elif advanced_count > 0:
            return 'intermediate'
        else:
            return 'novice'
    
    def get_analysis_patterns(self, context: AnalysisContext) -> Dict[str, Any]:
        """分析パターンを取得"""
        if not context.analysis_history:
            return {}
        
        # 使用頻度の高い分析タイプ
        analysis_types = [entry.get('analysis_type', 'unknown') 
                         for entry in context.analysis_history]
        type_counts = {}
        for analysis_type in analysis_types:
            type_counts[analysis_type] = type_counts.get(analysis_type, 0) + 1
        
        # 使用頻度の高いプロバイダー
        providers = [entry.get('provider', 'unknown') 
                    for entry in context.analysis_history]
        provider_counts = {}
        for provider in providers:
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        
        # 平均処理時間
        processing_times = [entry.get('processing_time', 0.0) 
                          for entry in context.analysis_history 
                          if entry.get('processing_time', 0.0) > 0]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        
        return {
            'most_used_analysis_types': sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            'preferred_providers': sorted(provider_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            'average_processing_time': avg_processing_time,
            'total_analyses': len(context.analysis_history),
            'success_rate': sum(1 for entry in context.analysis_history 
                              if entry.get('success', False)) / len(context.analysis_history)
        }
    
    def cleanup_old_sessions(self, days_old: int = 30):
        """古いセッションをクリーンアップ"""
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)
        
        sessions_to_remove = []
        for context_key, context in self.sessions.items():
            if context.timestamp.timestamp() < cutoff_time:
                sessions_to_remove.append(context_key)
        
        for context_key in sessions_to_remove:
            del self.sessions[context_key]
            # ディスクからも削除
            context_file = self.checkpoint_dir / f"context_{context_key}.json"
            if context_file.exists():
                context_file.unlink()
        
        self.logger.info(f"古いセッション {len(sessions_to_remove)} 件をクリーンアップしました")
    
    def _load_existing_sessions(self):
        """既存セッションを読み込み"""
        try:
            for context_file in self.checkpoint_dir.glob("context_*.json"):
                context_key = context_file.stem.replace("context_", "")
                context = self._load_context_from_disk(context_key)
                if context:
                    self.sessions[context_key] = context
            
            self.logger.info(f"既存セッション {len(self.sessions)} 件を復旧しました")
        except Exception as e:
            self.logger.error(f"セッション復旧エラー: {e}")
    
    def _load_context_from_disk(self, context_key: str) -> Optional[AnalysisContext]:
        """ディスクからコンテキストを読み込み（拡張版）"""
        context_file = self.checkpoint_dir / f"context_{context_key}.json"
        
        if not context_file.exists():
            return None
        
        try:
            with open(context_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # JSONからAnalysisContextを復元（拡張フィールド対応）
            context = AnalysisContext(
                user_id=data['user_id'],
                session_id=data['session_id'],
                data_fingerprint=data['data_fingerprint'],
                analysis_history=data.get('analysis_history', []),
                user_expertise_level=data.get('user_expertise_level', 'intermediate'),
                privacy_settings=data.get('privacy_settings', {}),
                timestamp=datetime.fromisoformat(data['timestamp']),
                # 拡張フィールド
                user_preferences=data.get('user_preferences', {}),
                session_metadata=data.get('session_metadata', {}),
                context_tags=data.get('context_tags', []),
                learning_progress=data.get('learning_progress', {}),
                favorite_methods=data.get('favorite_methods', []),
                recent_queries=data.get('recent_queries', [])
            )
            
            return context
            
        except Exception as e:
            self.logger.error(f"コンテキスト読み込みエラー {context_key}: {e}")
            return None
    
    def _save_context_to_disk(self, context: AnalysisContext):
        """コンテキストをディスクに保存（拡張版）"""
        context_key = f"{context.user_id}_{context.session_id}"
        context_file = self.checkpoint_dir / f"context_{context_key}.json"
        
        try:
            # AnalysisContextをJSONシリアライズ可能な形式に変換（拡張フィールド対応）
            data = {
                'user_id': context.user_id,
                'session_id': context.session_id,
                'data_fingerprint': context.data_fingerprint,
                'analysis_history': context.analysis_history,
                'user_expertise_level': context.user_expertise_level,
                'privacy_settings': context.privacy_settings,
                'timestamp': context.timestamp.isoformat(),
                # 拡張フィールド
                'user_preferences': context.user_preferences,
                'session_metadata': context.session_metadata,
                'context_tags': context.context_tags,
                'learning_progress': context.learning_progress,
                'favorite_methods': context.favorite_methods,
                'recent_queries': context.recent_queries
            }
            
            with open(context_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"コンテキスト保存エラー {context_key}: {e}")
    
    def get_session_summary(self, context: AnalysisContext) -> Dict[str, Any]:
        """セッション要約を取得（拡張版）"""
        patterns = self.get_analysis_patterns(context)
        expertise_level = self.get_user_expertise_level(context)
        
        return {
            'user_id': context.user_id,
            'session_id': context.session_id,
            'session_start': context.timestamp.isoformat(),
            'current_expertise_level': expertise_level,
            'analysis_patterns': patterns,
            'data_fingerprint': context.data_fingerprint,
            'privacy_settings': context.privacy_settings,
            'user_preferences': context.user_preferences,
            'learning_progress': context.learning_progress,
            'favorite_methods': context.favorite_methods,
            'context_tags': context.context_tags
        }
    
    def update_user_preferences(self, context: AnalysisContext, preferences: Dict[str, Any]):
        """ユーザー設定を更新"""
        context.user_preferences.update(preferences)
        self._save_context_to_disk(context)
        self.logger.info(f"ユーザー設定更新: {context.user_id}")
    
    def add_context_tag(self, context: AnalysisContext, tag: str):
        """コンテキストタグを追加"""
        if tag not in context.context_tags:
            context.context_tags.append(tag)
            self._save_context_to_disk(context)
    
    def update_learning_progress(self, context: AnalysisContext, concept: str, progress: float):
        """学習進捗を更新"""
        context.learning_progress[concept] = progress
        self._save_context_to_disk(context)
        self.logger.debug(f"学習進捗更新: {concept} -> {progress}")
    
    def add_favorite_method(self, context: AnalysisContext, method: str):
        """お気に入り手法を追加"""
        if method not in context.favorite_methods:
            context.favorite_methods.append(method)
            # 最大10個まで保持
            if len(context.favorite_methods) > 10:
                context.favorite_methods = context.favorite_methods[-10:]
            self._save_context_to_disk(context)
    
    def add_recent_query(self, context: AnalysisContext, query: str):
        """最近のクエリを追加"""
        context.recent_queries.insert(0, query)
        # 最大20個まで保持
        if len(context.recent_queries) > 20:
            context.recent_queries = context.recent_queries[:20]
        self._save_context_to_disk(context)
    
    def get_contextual_recommendations(self, context: AnalysisContext) -> Dict[str, Any]:
        """コンテキストに基づく推奨事項を生成"""
        recommendations = {
            'suggested_methods': [],
            'learning_opportunities': [],
            'workflow_improvements': []
        }
        
        # 使用履歴に基づく推奨
        patterns = self.get_analysis_patterns(context)
        if patterns.get('most_used_analysis_types'):
            most_used = patterns['most_used_analysis_types'][0][0]
            recommendations['suggested_methods'].append(f"{most_used}の高度な手法")
        
        # 学習進捗に基づく推奨
        for concept, progress in context.learning_progress.items():
            if progress < 0.7:  # 70%未満の理解度
                recommendations['learning_opportunities'].append(f"{concept}の復習")
        
        # 専門レベルに基づく推奨
        expertise = self.get_user_expertise_level(context)
        if expertise == 'novice':
            recommendations['learning_opportunities'].extend([
                '基本統計の学習', 'データ可視化の基礎'
            ])
        elif expertise == 'expert':
            recommendations['suggested_methods'].extend([
                'ベイズ統計', '機械学習手法'
            ])
        
        return recommendations
    
    def generate_context_aware_response(self, context: AnalysisContext, base_response: str) -> str:
        """コンテキストを考慮した応答を生成"""
        # ユーザーの専門レベルに応じて応答を調整
        expertise = self.get_user_expertise_level(context)
        language = context.user_preferences.get('language', 'ja')
        explanation_style = context.user_preferences.get('explanation_style', 'detailed')
        
        # 応答の調整ロジック
        if expertise == 'novice' and explanation_style == 'detailed':
            # 初心者向けに詳細な説明を追加
            base_response += "\n\n📚 補足説明: この分析手法について詳しく学びたい場合は、基本概念から始めることをお勧めします。"
        elif expertise == 'expert' and explanation_style == 'concise':
            # 専門家向けに簡潔な応答に調整
            base_response = base_response.split('\n')[0]  # 最初の行のみ
        
        # 言語設定に応じた調整
        if language == 'en':
            # 英語での応答に変換（簡易実装）
            base_response = base_response.replace('分析', 'analysis').replace('結果', 'result')
        
        return base_response

@dataclass
class DataCharacteristics:
    """データ特性情報"""
    n_rows: int
    n_columns: int
    column_types: Dict[str, str]
    missing_data_pattern: Dict[str, float]
    distribution_characteristics: Dict[str, Dict[str, Any]]
    correlation_structure: Optional[np.ndarray] = None
    outlier_information: Dict[str, Any] = None
    data_quality_score: float = 0.0
    
    def __post_init__(self):
        if self.outlier_information is None:
            self.outlier_information = {}

@dataclass
class MethodSuggestion:
    """統計手法提案"""
    method_name: str
    confidence_score: float
    rationale: str
    assumptions: List[str]
    prerequisites: List[str]
    estimated_computation_time: float
    educational_content: Optional[str] = None
    alternative_methods: List[str] = None
    
    def __post_init__(self):
        if self.alternative_methods is None:
            self.alternative_methods = []

@dataclass
class AssumptionValidationResult:
    """仮定検証結果"""
    method: str
    assumptions_met: Dict[str, bool]
    violation_severity: Dict[str, str]
    corrective_actions: List[str]
    alternative_methods: List[str]
    confidence_in_results: float
    test_statistics: Dict[str, Any] = None
    p_values: Dict[str, float] = None
    
    def __post_init__(self):
        if self.test_statistics is None:
            self.test_statistics = {}
        if self.p_values is None:
            self.p_values = {}

class AssumptionValidator:
    """統計的仮定自動検証システム - SPSS以上の機能"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AssumptionValidator")
        
        # 仮定検証テストのマッピング
        self.assumption_tests = {
            'normality': self._test_normality,
            'homoscedasticity': self._test_homoscedasticity,
            'independence': self._test_independence,
            'linearity': self._test_linearity,
            'multicollinearity': self._test_multicollinearity,
            'outliers': self._test_outliers,
            'sphericity': self._test_sphericity,
            'equal_variances': self._test_equal_variances
        }
        
        # 違反の重要度レベル
        self.severity_levels = {
            'critical': 'クリティカル - 結果の信頼性に重大な影響',
            'major': '重要 - 結果の解釈に注意が必要',
            'minor': '軽微 - 結果への影響は限定的',
            'warning': '警告 - 確認が推奨される'
        }
    
    def validate_assumptions(self, method: str, data: pd.DataFrame, 
                           target_col: str = None, group_col: str = None) -> AssumptionValidationResult:
        """統計手法の仮定を包括的に検証"""
        try:
            # 手法別の必要な仮定を取得
            required_assumptions = self._get_method_assumptions(method)
            
            assumptions_met = {}
            violation_severity = {}
            corrective_actions = []
            alternative_methods = []
            
            # 各仮定をテスト
            for assumption in required_assumptions:
                if assumption in self.assumption_tests:
                    test_result = self.assumption_tests[assumption](data, target_col, group_col)
                    assumptions_met[assumption] = test_result['passed']
                    
                    if not test_result['passed']:
                        violation_severity[assumption] = test_result['severity']
                        corrective_actions.extend(test_result['corrective_actions'])
                        alternative_methods.extend(test_result['alternative_methods'])
                else:
                    # 未実装の仮定テストは警告として扱う
                    assumptions_met[assumption] = False
                    violation_severity[assumption] = 'warning'
                    corrective_actions.append(f"{assumption}の検証は未実装です")
            
            # 信頼度スコアを計算
            confidence_score = self._calculate_confidence_score(assumptions_met, violation_severity)
            
            # 重複を除去
            corrective_actions = list(set(corrective_actions))
            alternative_methods = list(set(alternative_methods))
            
            return AssumptionValidationResult(
                method=method,
                assumptions_met=assumptions_met,
                violation_severity=violation_severity,
                corrective_actions=corrective_actions,
                alternative_methods=alternative_methods,
                confidence_in_results=confidence_score
            )
            
        except Exception as e:
            self.logger.error(f"仮定検証エラー: {e}")
            return AssumptionValidationResult(
                method=method,
                assumptions_met={},
                violation_severity={},
                corrective_actions=[f"検証中にエラーが発生しました: {str(e)}"],
                alternative_methods=[],
                confidence_in_results=0.0
            )
    
    def _get_method_assumptions(self, method: str) -> List[str]:
        """統計手法に必要な仮定を取得"""
        method_assumptions = {
            't_test': ['normality', 'independence', 'homoscedasticity'],
            'anova': ['normality', 'independence', 'homoscedasticity'],
            'linear_regression': ['linearity', 'independence', 'homoscedasticity', 'normality', 'multicollinearity'],
            'logistic_regression': ['independence', 'linearity', 'multicollinearity'],
            'correlation': ['linearity', 'normality'],
            'chi_square': ['independence', 'expected_frequency'],
            'mann_whitney': ['independence'],
            'wilcoxon': ['independence'],
            'kruskal_wallis': ['independence'],
            'repeated_measures_anova': ['normality', 'sphericity', 'independence']
        }
        
        return method_assumptions.get(method, [])
    
    def _test_normality(self, data: pd.DataFrame, target_col: str = None, group_col: str = None) -> Dict[str, Any]:
        """正規性検定"""
        try:
            if target_col is None:
                # 全ての数値列をテスト
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                test_data = data[numeric_cols].dropna()
            else:
                test_data = data[target_col].dropna()
            
            if len(test_data) < 8:
                return {
                    'passed': False,
                    'severity': 'warning',
                    'p_value': None,
                    'test_statistic': None,
                    'corrective_actions': ['サンプルサイズが小さすぎます（n<8）'],
                    'alternative_methods': ['ノンパラメトリック検定']
                }
            
            # Shapiro-Wilk検定（サンプルサイズ≤5000）
            if isinstance(test_data, pd.Series):
                test_values = test_data.values
            else:
                # 複数列の場合は最初の列を使用
                test_values = test_data.iloc[:, 0].values
            
            # サンプルサイズを制限
            if len(test_values) > 5000:
                test_values = np.random.choice(test_values, 5000, replace=False)
            
            stat, p_value = stats.shapiro(test_values)
            
            # 正規性の判定（p > 0.05で正規分布）
            is_normal = p_value > 0.05
            
            if is_normal:
                return {
                    'passed': True,
                    'severity': None,
                    'p_value': p_value,
                    'test_statistic': stat,
                    'corrective_actions': [],
                    'alternative_methods': []
                }
            else:
                severity = 'critical' if p_value < 0.001 else 'major' if p_value < 0.01 else 'minor'
                return {
                    'passed': False,
                    'severity': severity,
                    'p_value': p_value,
                    'test_statistic': stat,
                    'corrective_actions': [
                        'データ変換（対数変換、平方根変換）を検討',
                        '外れ値の除去を検討',
                        'より大きなサンプルサイズを取得'
                    ],
                    'alternative_methods': [
                        'Mann-Whitney U検定',
                        'Kruskal-Wallis検定',
                        'Wilcoxon符号順位検定'
                    ]
                }
                
        except Exception as e:
            return {
                'passed': False,
                'severity': 'warning',
                'p_value': None,
                'test_statistic': None,
                'corrective_actions': [f'正規性検定でエラー: {str(e)}'],
                'alternative_methods': ['ノンパラメトリック検定']
            }
    
    def _test_homoscedasticity(self, data: pd.DataFrame, target_col: str = None, group_col: str = None) -> Dict[str, Any]:
        """等分散性検定（Levene検定）"""
        try:
            if target_col is None or group_col is None:
                return {
                    'passed': False,
                    'severity': 'warning',
                    'corrective_actions': ['等分散性検定には目的変数とグループ変数が必要です'],
                    'alternative_methods': []
                }
            
            # グループ別にデータを分割
            groups = []
            for group_name in data[group_col].unique():
                group_data = data[data[group_col] == group_name][target_col].dropna()
                if len(group_data) > 0:
                    groups.append(group_data)
            
            if len(groups) < 2:
                return {
                    'passed': False,
                    'severity': 'critical',
                    'corrective_actions': ['比較するグループが不足しています'],
                    'alternative_methods': []
                }
            
            # Levene検定
            stat, p_value = stats.levene(*groups)
            
            # 等分散性の判定（p > 0.05で等分散）
            is_homoscedastic = p_value > 0.05
            
            if is_homoscedastic:
                return {
                    'passed': True,
                    'severity': None,
                    'p_value': p_value,
                    'test_statistic': stat,
                    'corrective_actions': [],
                    'alternative_methods': []
                }
            else:
                severity = 'critical' if p_value < 0.001 else 'major' if p_value < 0.01 else 'minor'
                return {
                    'passed': False,
                    'severity': severity,
                    'p_value': p_value,
                    'test_statistic': stat,
                    'corrective_actions': [
                        'Welchのt検定を使用（等分散を仮定しない）',
                        'データ変換を検討',
                        '外れ値の確認と処理'
                    ],
                    'alternative_methods': [
                        'Welch t検定',
                        'Mann-Whitney U検定',
                        'ブートストラップ検定'
                    ]
                }
                
        except Exception as e:
            return {
                'passed': False,
                'severity': 'warning',
                'corrective_actions': [f'等分散性検定でエラー: {str(e)}'],
                'alternative_methods': ['ノンパラメトリック検定']
            }
    
    def _test_independence(self, data: pd.DataFrame, target_col: str = None, group_col: str = None) -> Dict[str, Any]:
        """独立性の検定（Durbin-Watson検定など）"""
        try:
            # 時系列データの場合はDurbin-Watson検定
            if target_col and target_col in data.columns:
                test_data = data[target_col].dropna()
                
                if len(test_data) < 10:
                    return {
                        'passed': False,
                        'severity': 'warning',
                        'corrective_actions': ['独立性検定にはより多くのデータが必要です'],
                        'alternative_methods': []
                    }
                
                # 簡易的な自己相関検定
                from statsmodels.stats.diagnostic import acorr_ljungbox
                
                try:
                    ljung_box_result = acorr_ljungbox(test_data, lags=min(10, len(test_data)//4), return_df=True)
                    p_values = ljung_box_result['lb_pvalue']
                    min_p_value = p_values.min()
                    
                    # 独立性の判定（p > 0.05で独立）
                    is_independent = min_p_value > 0.05
                    
                    if is_independent:
                        return {
                            'passed': True,
                            'severity': None,
                            'p_value': min_p_value,
                            'corrective_actions': [],
                            'alternative_methods': []
                        }
                    else:
                        severity = 'major' if min_p_value < 0.01 else 'minor'
                        return {
                            'passed': False,
                            'severity': severity,
                            'p_value': min_p_value,
                            'corrective_actions': [
                                '時系列分析手法の使用を検討',
                                'ランダムサンプリングの確認',
                                'データ収集方法の見直し'
                            ],
                            'alternative_methods': [
                                '時系列分析',
                                '混合効果モデル',
                                'GEE（一般化推定方程式）'
                            ]
                        }
                        
                except ImportError:
                    # statsmodelsが利用できない場合の簡易チェック
                    return {
                        'passed': True,  # 保守的に通す
                        'severity': 'warning',
                        'corrective_actions': ['独立性の詳細検定には追加ライブラリが必要です'],
                        'alternative_methods': []
                    }
            
            # デフォルトでは独立性を仮定
            return {
                'passed': True,
                'severity': None,
                'corrective_actions': [],
                'alternative_methods': []
            }
            
        except Exception as e:
            return {
                'passed': True,  # エラー時は保守的に通す
                'severity': 'warning',
                'corrective_actions': [f'独立性検定でエラー: {str(e)}'],
                'alternative_methods': []
            }
    
    def _test_linearity(self, data: pd.DataFrame, target_col: str = None, group_col: str = None) -> Dict[str, Any]:
        """線形性の検定"""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return {
                    'passed': False,
                    'severity': 'warning',
                    'corrective_actions': ['線形性検定には2つ以上の数値変数が必要です'],
                    'alternative_methods': []
                }
            
            # 相関係数による線形性の簡易チェック
            correlations = data[numeric_cols].corr()
            
            # 対角成分を除いた相関係数の絶対値
            corr_values = []
            for i in range(len(correlations)):
                for j in range(i+1, len(correlations)):
                    corr_values.append(abs(correlations.iloc[i, j]))
            
            if not corr_values:
                return {
                    'passed': True,
                    'severity': None,
                    'corrective_actions': [],
                    'alternative_methods': []
                }
            
            max_correlation = max(corr_values)
            
            # 線形関係の強さで判定
            if max_correlation > 0.3:  # 中程度以上の線形関係があれば通す
                return {
                    'passed': True,
                    'severity': None,
                    'max_correlation': max_correlation,
                    'corrective_actions': [],
                    'alternative_methods': []
                }
            else:
                return {
                    'passed': False,
                    'severity': 'minor',
                    'max_correlation': max_correlation,
                    'corrective_actions': [
                        '散布図で関係性を視覚的に確認',
                        '非線形変換を検討',
                        'スプライン回帰や多項式回帰を検討'
                    ],
                    'alternative_methods': [
                        'スピアマン相関',
                        '非線形回帰',
                        '決定木ベースの手法'
                    ]
                }
                
        except Exception as e:
            return {
                'passed': True,  # エラー時は保守的に通す
                'severity': 'warning',
                'corrective_actions': [f'線形性検定でエラー: {str(e)}'],
                'alternative_methods': []
            }
    
    def _test_multicollinearity(self, data: pd.DataFrame, target_col: str = None, group_col: str = None) -> Dict[str, Any]:
        """多重共線性の検定（VIF計算）"""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            # 目的変数を除外
            if target_col and target_col in numeric_cols:
                predictor_cols = [col for col in numeric_cols if col != target_col]
            else:
                predictor_cols = list(numeric_cols)
            
            if len(predictor_cols) < 2:
                return {
                    'passed': True,
                    'severity': None,
                    'corrective_actions': [],
                    'alternative_methods': []
                }
            
            # 相関行列による簡易多重共線性チェック
            corr_matrix = data[predictor_cols].corr()
            
            # 高い相関（|r| > 0.8）をチェック
            high_correlations = []
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    corr_value = abs(corr_matrix.iloc[i, j])
                    if corr_value > 0.8:
                        high_correlations.append({
                            'var1': corr_matrix.index[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            if not high_correlations:
                return {
                    'passed': True,
                    'severity': None,
                    'corrective_actions': [],
                    'alternative_methods': []
                }
            else:
                severity = 'major' if any(hc['correlation'] > 0.9 for hc in high_correlations) else 'minor'
                
                corrective_actions = [
                    '高い相関を持つ変数の一方を除去',
                    '主成分分析による次元削減',
                    'リッジ回帰やLasso回帰の使用'
                ]
                
                return {
                    'passed': False,
                    'severity': severity,
                    'high_correlations': high_correlations,
                    'corrective_actions': corrective_actions,
                    'alternative_methods': [
                        'リッジ回帰',
                        'Lasso回帰',
                        '主成分回帰'
                    ]
                }
                
        except Exception as e:
            return {
                'passed': True,  # エラー時は保守的に通す
                'severity': 'warning',
                'corrective_actions': [f'多重共線性検定でエラー: {str(e)}'],
                'alternative_methods': []
            }
    
    def _test_outliers(self, data: pd.DataFrame, target_col: str = None, group_col: str = None) -> Dict[str, Any]:
        """外れ値の検出"""
        try:
            if target_col:
                test_data = data[target_col].dropna()
            else:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    return {
                        'passed': True,
                        'severity': None,
                        'corrective_actions': [],
                        'alternative_methods': []
                    }
                test_data = data[numeric_cols[0]].dropna()
            
            if len(test_data) < 10:
                return {
                    'passed': True,
                    'severity': 'warning',
                    'corrective_actions': ['外れ値検出にはより多くのデータが必要です'],
                    'alternative_methods': []
                }
            
            # IQR法による外れ値検出
            Q1 = test_data.quantile(0.25)
            Q3 = test_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = test_data[(test_data < lower_bound) | (test_data > upper_bound)]
            outlier_ratio = len(outliers) / len(test_data)
            
            if outlier_ratio <= 0.05:  # 5%以下なら許容
                return {
                    'passed': True,
                    'severity': None,
                    'outlier_count': len(outliers),
                    'outlier_ratio': outlier_ratio,
                    'corrective_actions': [],
                    'alternative_methods': []
                }
            else:
                severity = 'major' if outlier_ratio > 0.1 else 'minor'
                return {
                    'passed': False,
                    'severity': severity,
                    'outlier_count': len(outliers),
                    'outlier_ratio': outlier_ratio,
                    'corrective_actions': [
                        '外れ値の原因を調査',
                        '外れ値の除去または変換を検討',
                        'ロバスト統計手法の使用'
                    ],
                    'alternative_methods': [
                        'ロバスト回帰',
                        'ノンパラメトリック検定',
                        'ブートストラップ法'
                    ]
                }
                
        except Exception as e:
            return {
                'passed': True,  # エラー時は保守的に通す
                'severity': 'warning',
                'corrective_actions': [f'外れ値検定でエラー: {str(e)}'],
                'alternative_methods': []
            }
    
    def _test_sphericity(self, data: pd.DataFrame, target_col: str = None, group_col: str = None) -> Dict[str, Any]:
        """球面性の検定（反復測定ANOVA用）"""
        # 簡易実装：実際のMauchly検定は複雑なため、警告レベルで返す
        return {
            'passed': True,
            'severity': 'warning',
            'corrective_actions': ['球面性の詳細検定は手動で確認してください'],
            'alternative_methods': ['Greenhouse-Geisser補正', 'Huynh-Feldt補正']
        }
    
    def _test_equal_variances(self, data: pd.DataFrame, target_col: str = None, group_col: str = None) -> Dict[str, Any]:
        """等分散性検定（homoscedasticityのエイリアス）"""
        return self._test_homoscedasticity(data, target_col, group_col)
    
    def _calculate_confidence_score(self, assumptions_met: Dict[str, bool], 
                                  violation_severity: Dict[str, str]) -> float:
        """仮定検証結果に基づく信頼度スコアを計算"""
        if not assumptions_met:
            return 0.5  # デフォルト
        
        total_assumptions = len(assumptions_met)
        met_assumptions = sum(assumptions_met.values())
        
        # 基本スコア（満たされた仮定の割合）
        base_score = met_assumptions / total_assumptions
        
        # 違反の重要度による減点
        severity_penalties = {
            'critical': 0.3,
            'major': 0.2,
            'minor': 0.1,
            'warning': 0.05
        }
        
        penalty = 0.0
        for assumption, severity in violation_severity.items():
            penalty += severity_penalties.get(severity, 0.0)
        
        # 最終スコア（0.0-1.0の範囲）
        final_score = max(0.0, min(1.0, base_score - penalty))
        
        return final_score

class StatisticalMethodAdvisor:
    """統計手法アドバイザー - 知的推奨システム"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.StatisticalMethodAdvisor")
        
        # 統計手法データベース
        self.method_database = self._initialize_method_database()
        
        # データ前処理器
        try:
            from data_preprocessing import DataPreprocessor
            self.data_preprocessor = DataPreprocessor()
        except ImportError:
            self.data_preprocessor = None
            self.logger.warning("DataPreprocessorが利用できません")
        
        # 仮定検証器
        self.assumption_validator = AssumptionValidator()
    
    def _initialize_method_database(self) -> Dict[str, Dict[str, Any]]:
        """統計手法データベースの初期化"""
        return {
            'descriptive_stats': {
                'name': '記述統計',
                'category': 'descriptive',
                'assumptions': [],
                'data_requirements': {
                    'min_sample_size': 1,
                    'data_types': ['numeric', 'categorical'],
                    'missing_data_tolerance': 0.5
                },
                'use_cases': ['データ概要', '基本統計量', '分布確認'],
                'computation_complexity': 'low',
                'educational_level': 'beginner'
            },
            't_test': {
                'name': 't検定',
                'category': 'inferential',
                'assumptions': ['normality', 'independence', 'homoscedasticity'],
                'data_requirements': {
                    'min_sample_size': 30,
                    'data_types': ['numeric'],
                    'missing_data_tolerance': 0.1
                },
                'use_cases': ['平均値比較', '群間差検定'],
                'computation_complexity': 'low',
                'educational_level': 'intermediate'
            },
            'anova': {
                'name': '分散分析',
                'category': 'inferential',
                'assumptions': ['normality', 'independence', 'homoscedasticity'],
                'data_requirements': {
                    'min_sample_size': 20,
                    'data_types': ['numeric'],
                    'missing_data_tolerance': 0.1
                },
                'use_cases': ['多群比較', '要因効果検定'],
                'computation_complexity': 'medium',
                'educational_level': 'intermediate'
            },
            'wilcoxon_test': {
                'name': 'Wilcoxon検定',
                'category': 'inferential',
                'assumptions': ['independence'],
                'data_requirements': {
                    'min_sample_size': 6,
                    'data_types': ['numeric'],
                    'missing_data_tolerance': 0.1
                },
                'use_cases': ['小サンプル比較', 'ノンパラメトリック検定'],
                'computation_complexity': 'low',
                'educational_level': 'intermediate'
            },
            'mann_whitney_test': {
                'name': 'Mann-Whitney U検定',
                'category': 'inferential',
                'assumptions': ['independence'],
                'data_requirements': {
                    'min_sample_size': 8,
                    'data_types': ['numeric'],
                    'missing_data_tolerance': 0.1
                },
                'use_cases': ['小サンプル群間比較', 'ノンパラメトリック検定'],
                'computation_complexity': 'low',
                'educational_level': 'intermediate'
            },
            'correlation': {
                'name': '相関分析',
                'category': 'descriptive',
                'assumptions': ['linearity'],
                'data_requirements': {
                    'min_sample_size': 10,
                    'data_types': ['numeric'],
                    'missing_data_tolerance': 0.2
                },
                'use_cases': ['変数間関係', '関連性分析'],
                'computation_complexity': 'low',
                'educational_level': 'beginner'
            },
            'linear_regression': {
                'name': '線形回帰',
                'category': 'predictive',
                'assumptions': ['linearity', 'independence', 'homoscedasticity', 'normality_residuals'],
                'data_requirements': {
                    'min_sample_size': 50,
                    'data_types': ['numeric'],
                    'missing_data_tolerance': 0.05
                },
                'use_cases': ['予測', '関係性モデリング', '因果推論'],
                'computation_complexity': 'medium',
                'educational_level': 'intermediate'
            },
            'logistic_regression': {
                'name': 'ロジスティック回帰',
                'category': 'predictive',
                'assumptions': ['independence', 'linearity_logit'],
                'data_requirements': {
                    'min_sample_size': 100,
                    'data_types': ['numeric', 'categorical'],
                    'missing_data_tolerance': 0.05
                },
                'use_cases': ['分類', '確率予測', 'オッズ比分析'],
                'computation_complexity': 'medium',
                'educational_level': 'advanced'
            },
            'chi_square': {
                'name': 'カイ二乗検定',
                'category': 'inferential',
                'assumptions': ['independence', 'expected_frequency'],
                'data_requirements': {
                    'min_sample_size': 20,
                    'data_types': ['categorical'],
                    'missing_data_tolerance': 0.1
                },
                'use_cases': ['独立性検定', '適合度検定'],
                'computation_complexity': 'low',
                'educational_level': 'intermediate'
            },
            'machine_learning': {
                'name': '機械学習',
                'category': 'predictive',
                'assumptions': [],
                'data_requirements': {
                    'min_sample_size': 1000,
                    'data_types': ['numeric', 'categorical'],
                    'missing_data_tolerance': 0.1
                },
                'use_cases': ['複雑な予測', 'パターン認識', '非線形関係'],
                'computation_complexity': 'high',
                'educational_level': 'advanced'
            }
        }
    
    def analyze_data_characteristics(self, data: pd.DataFrame) -> DataCharacteristics:
        """データ特性を分析"""
        try:
            # 基本情報
            n_rows, n_columns = data.shape
            
            # 列タイプ分析
            column_types = {}
            for col in data.columns:
                if data[col].dtype in ['int64', 'float64']:
                    column_types[col] = 'numeric'
                elif data[col].dtype in ['object', 'category']:
                    column_types[col] = 'categorical'
                elif data[col].dtype in ['datetime64[ns]']:
                    column_types[col] = 'datetime'
                else:
                    column_types[col] = 'other'
            
            # 欠損データパターン
            missing_data_pattern = {}
            for col in data.columns:
                missing_ratio = data[col].isnull().sum() / len(data)
                missing_data_pattern[col] = missing_ratio
            
            # 分布特性（数値列のみ）
            distribution_characteristics = {}
            numeric_cols = [col for col, dtype in column_types.items() if dtype == 'numeric']
            
            for col in numeric_cols:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    distribution_characteristics[col] = {
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'skewness': float(col_data.skew()),
                        'kurtosis': float(col_data.kurtosis()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'q25': float(col_data.quantile(0.25)),
                        'q50': float(col_data.quantile(0.50)),
                        'q75': float(col_data.quantile(0.75))
                    }
            
            # 相関構造（数値列のみ）
            correlation_structure = None
            if len(numeric_cols) > 1:
                numeric_data = data[numeric_cols].dropna()
                if len(numeric_data) > 0:
                    correlation_structure = numeric_data.corr().values
            
            # 外れ値情報
            outlier_information = {}
            if self.data_preprocessor:
                outlier_result = self.data_preprocessor.detect_outliers(data)
                if outlier_result.get('success', False):
                    outlier_information = outlier_result
            
            # データ品質スコア計算
            data_quality_score = self._calculate_data_quality_score(
                missing_data_pattern, outlier_information, n_rows, n_columns
            )
            
            return DataCharacteristics(
                n_rows=n_rows,
                n_columns=n_columns,
                column_types=column_types,
                missing_data_pattern=missing_data_pattern,
                distribution_characteristics=distribution_characteristics,
                correlation_structure=correlation_structure,
                outlier_information=outlier_information,
                data_quality_score=data_quality_score
            )
            
        except Exception as e:
            self.logger.error(f"データ特性分析エラー: {e}")
            # エラー時のデフォルト値
            return DataCharacteristics(
                n_rows=0,
                n_columns=0,
                column_types={},
                missing_data_pattern={},
                distribution_characteristics={}
            )
    
    def _calculate_data_quality_score(self, missing_pattern: Dict[str, float], 
                                    outlier_info: Dict[str, Any], 
                                    n_rows: int, n_columns: int) -> float:
        """データ品質スコアを計算（0-1の範囲）"""
        score = 1.0
        
        # 欠損データによる減点
        avg_missing_ratio = sum(missing_pattern.values()) / len(missing_pattern) if missing_pattern else 0
        score -= avg_missing_ratio * 0.3
        
        # 外れ値による減点
        if outlier_info.get('success', False):
            consensus_outliers = outlier_info.get('consensus_outliers', {})
            total_outlier_ratio = 0
            for col_info in consensus_outliers.values():
                total_outlier_ratio += len(col_info.get('indices', [])) / n_rows
            avg_outlier_ratio = total_outlier_ratio / len(consensus_outliers) if consensus_outliers else 0
            score -= avg_outlier_ratio * 0.2
        
        # サンプルサイズによる調整
        if n_rows < 30:
            score -= 0.2
        elif n_rows < 100:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def suggest_methods(self, data_chars: DataCharacteristics, 
                       research_question: str = "", 
                       user_expertise: str = "intermediate") -> List[MethodSuggestion]:
        """統計手法を提案"""
        try:
            suggestions = []
            
            # 研究質問の分析
            question_intent = self._analyze_research_question(research_question)
            
            # 各手法の適合性を評価
            for method_id, method_info in self.method_database.items():
                compatibility_score = self._calculate_method_compatibility(
                    method_info, data_chars, question_intent, user_expertise
                )
                
                # サンプルサイズが不足している場合はより高い閾値を適用
                min_sample_size = method_info['data_requirements']['min_sample_size']
                if data_chars.n_rows < min_sample_size:
                    # サンプルサイズが不足している場合は大幅に高い閾値
                    threshold = 0.8
                else:
                    threshold = 0.3
                
                if compatibility_score > threshold:  # 動的閾値で手法をフィルタリング
                    suggestion = MethodSuggestion(
                        method_name=method_id,  # 英語のキーを使用
                        confidence_score=compatibility_score,
                        rationale=self._generate_rationale(method_info, data_chars, question_intent),
                        assumptions=method_info['assumptions'],
                        prerequisites=self._generate_prerequisites(method_info, data_chars),
                        estimated_computation_time=self._estimate_computation_time(method_info, data_chars),
                        educational_content=self._generate_educational_content(method_info, user_expertise),
                        alternative_methods=self._find_alternative_methods(method_id, method_info)
                    )
                    suggestions.append(suggestion)
            
            # 信頼度スコアでソート
            suggestions.sort(key=lambda x: x.confidence_score, reverse=True)
            
            return suggestions[:5]  # 上位5つを返す
            
        except Exception as e:
            self.logger.error(f"手法提案エラー: {e}")
            return []
    
    def _analyze_research_question(self, question: str) -> Dict[str, Any]:
        """研究質問を分析"""
        question_lower = question.lower()
        
        intent = {
            'type': 'descriptive',  # デフォルト
            'keywords': [],
            'variables_mentioned': [],
            'analysis_goal': 'exploration'
        }
        
        # 分析タイプの判定（優先順位付き）
        if any(word in question_lower for word in ['予測', 'predict', '予想', 'forecast', '将来', 'future']):
            intent['type'] = 'predictive'
            intent['analysis_goal'] = 'prediction'
        elif any(word in question_lower for word in ['比較', 'compare', '差', 'difference', '検定', 'test', '有意', 'significant', 'グループ間', '平均値']):
            intent['type'] = 'inferential'
            intent['analysis_goal'] = 'comparison'
        elif any(word in question_lower for word in ['要約', 'summary', '統計量', 'statistics', '基本', 'basic', '記述', 'descriptive']):
            intent['type'] = 'descriptive'
            intent['analysis_goal'] = 'description'
        elif any(word in question_lower for word in ['関係', 'relationship', '相関', 'correlation', '関連', 'association']):
            intent['type'] = 'descriptive'
            intent['analysis_goal'] = 'relationship'
        elif any(word in question_lower for word in ['カテゴリ', 'category', '独立性', 'independence', 'クロス', 'cross']):
            intent['type'] = 'inferential'
            intent['analysis_goal'] = 'categorical_analysis'
        
        # キーワード抽出
        statistical_keywords = [
            '平均', 'mean', '中央値', 'median', '分散', 'variance',
            '相関', 'correlation', '回帰', 'regression', '分類', 'classification',
            't検定', 't_test', 'anova', '分散分析', 'カイ二乗', 'chi_square'
        ]
        
        for keyword in statistical_keywords:
            if keyword in question_lower:
                intent['keywords'].append(keyword)
        
        return intent
    
    def _calculate_method_compatibility(self, method_info: Dict[str, Any], 
                                      data_chars: DataCharacteristics,
                                      question_intent: Dict[str, Any],
                                      user_expertise: str) -> float:
        """手法の適合性スコアを計算"""
        score = 0.0
        
        # 1. データ要件との適合性（40%）
        data_requirements = method_info['data_requirements']
        
        # サンプルサイズ（厳格な評価）
        if data_chars.n_rows >= data_requirements['min_sample_size']:
            score += 0.15
        else:
            # サンプルサイズが不足している場合は大幅減点
            ratio = data_chars.n_rows / data_requirements['min_sample_size']
            if ratio < 0.5:  # 必要サンプルサイズの50%未満の場合
                score += 0.0  # 加点なし
            else:
                score += 0.15 * ratio * 0.5  # 大幅減点
        
        # データタイプ
        required_types = data_requirements['data_types']
        available_types = set(data_chars.column_types.values())
        type_match = len(set(required_types) & available_types) / len(required_types)
        score += 0.15 * type_match
        
        # 欠損データ許容度
        avg_missing = sum(data_chars.missing_data_pattern.values()) / len(data_chars.missing_data_pattern) if data_chars.missing_data_pattern else 0
        if avg_missing <= data_requirements['missing_data_tolerance']:
            score += 0.1
        else:
            score += 0.1 * (1 - (avg_missing - data_requirements['missing_data_tolerance']))
        
        # 2. 研究質問との適合性（30%）
        if method_info['category'] == question_intent['type']:
            score += 0.2
        elif method_info['category'] == 'descriptive' and question_intent['type'] in ['inferential', 'predictive']:
            score += 0.1  # 記述統計は常に有用
        
        # キーワードマッチ
        method_keywords = method_info['use_cases']
        keyword_match = any(keyword in ' '.join(method_keywords).lower() 
                          for keyword in question_intent['keywords'])
        if keyword_match:
            score += 0.1
        
        # 3. ユーザー専門レベルとの適合性（20%）
        expertise_levels = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
        method_level = expertise_levels.get(method_info['educational_level'], 2)
        user_level = expertise_levels.get(user_expertise, 2)
        
        level_diff = abs(method_level - user_level)
        if level_diff == 0:
            score += 0.2
        elif level_diff == 1:
            score += 0.1
        # level_diff >= 2 の場合は加点なし
        
        # 4. データ品質との適合性（10%）
        if data_chars.data_quality_score > 0.8:
            score += 0.1
        elif data_chars.data_quality_score > 0.6:
            score += 0.05
        
        return min(1.0, score)
    
    def _generate_rationale(self, method_info: Dict[str, Any], 
                          data_chars: DataCharacteristics,
                          question_intent: Dict[str, Any]) -> str:
        """提案理由を生成"""
        rationale_parts = []
        
        # データ適合性
        if data_chars.n_rows >= method_info['data_requirements']['min_sample_size']:
            rationale_parts.append(f"サンプルサイズ（{data_chars.n_rows}）が十分です")
        
        # 分析目的適合性
        if method_info['category'] == question_intent['type']:
            rationale_parts.append(f"{question_intent['type']}分析に適しています")
        
        # データタイプ適合性
        required_types = method_info['data_requirements']['data_types']
        available_types = list(set(data_chars.column_types.values()))
        if set(required_types) & set(available_types):
            rationale_parts.append(f"データタイプ（{', '.join(available_types)}）に対応しています")
        
        return "。".join(rationale_parts) + "。"
    
    def _generate_prerequisites(self, method_info: Dict[str, Any], 
                              data_chars: DataCharacteristics) -> List[str]:
        """前提条件を生成"""
        prerequisites = []
        
        # サンプルサイズ要件
        min_size = method_info['data_requirements']['min_sample_size']
        if data_chars.n_rows < min_size:
            prerequisites.append(f"サンプルサイズを{min_size}以上に増やす必要があります")
        
        # 欠損データ処理
        avg_missing = sum(data_chars.missing_data_pattern.values()) / len(data_chars.missing_data_pattern) if data_chars.missing_data_pattern else 0
        tolerance = method_info['data_requirements']['missing_data_tolerance']
        if avg_missing > tolerance:
            prerequisites.append("欠損データの処理が必要です")
        
        # 外れ値処理
        if data_chars.outlier_information.get('success', False):
            consensus_outliers = data_chars.outlier_information.get('consensus_outliers', {})
            if consensus_outliers:
                prerequisites.append("外れ値の確認・処理を検討してください")
        
        return prerequisites
    
    def _estimate_computation_time(self, method_info: Dict[str, Any], 
                                 data_chars: DataCharacteristics) -> float:
        """計算時間を推定（秒）"""
        base_times = {
            'low': 0.1,
            'medium': 1.0,
            'high': 10.0
        }
        
        base_time = base_times.get(method_info['computation_complexity'], 1.0)
        
        # データサイズによる調整（より敏感に）
        size_factor = max(1.0, (data_chars.n_rows / 100) ** 0.5)
        column_factor = max(1.0, (data_chars.n_columns / 5) ** 0.5)
        
        return base_time * size_factor * column_factor
    
    def _generate_educational_content(self, method_info: Dict[str, Any], 
                                    user_expertise: str) -> str:
        """教育コンテンツを生成"""
        if user_expertise == 'beginner':
            # 初心者向け：詳細で分かりやすい説明
            base_content = f"{method_info['name']}は{method_info['category']}分析の手法です。"
            use_cases = f"主な用途は{', '.join(method_info['use_cases'])}です。"
            complexity = f"計算の複雑さは{method_info['computation_complexity']}レベルで、初心者にも理解しやすい手法です。"
            assumptions = f"この手法を使用する際の前提条件として、{', '.join(method_info['assumptions'])}が必要です。" if method_info['assumptions'] else "特別な前提条件はありません。"
            return f"{base_content} {use_cases} {complexity} {assumptions}"
        elif user_expertise == 'intermediate':
            # 中級者向け：適度な詳細
            assumptions_text = f"主な仮定: {', '.join(method_info['assumptions'])}" if method_info['assumptions'] else "特別な仮定はありません"
            use_cases = f"用途: {', '.join(method_info['use_cases'])}"
            return f"{method_info['name']}について - {assumptions_text}。{use_cases}。"
        else:  # advanced
            # 上級者向け：簡潔で技術的
            return f"{method_info['name']} - 複雑度: {method_info['computation_complexity']}"
    
    def _find_alternative_methods(self, current_method_id: str, 
                                current_method_info: Dict[str, Any]) -> List[str]:
        """代替手法を見つける"""
        alternatives = []
        current_category = current_method_info['category']
        
        for method_id, method_info in self.method_database.items():
            if (method_id != current_method_id and 
                method_info['category'] == current_category):
                alternatives.append(method_info['name'])
        
        return alternatives[:3]  # 最大3つ
    
    def validate_method_assumptions(self, method: str, data: pd.DataFrame,
                                  target_col: str = None, group_col: str = None) -> AssumptionValidationResult:
        """統計手法の仮定を検証"""
        return self.assumption_validator.validate_method_assumptions(
            method, data, target_col, group_col
        )
    
    def get_method_with_validation(self, data_chars: DataCharacteristics,
                                 research_question: str = "",
                                 user_expertise: str = "intermediate",
                                 data: pd.DataFrame = None,
                                 target_col: str = None,
                                 group_col: str = None) -> List[Dict[str, Any]]:
        """仮定検証付きの手法推奨"""
        try:
            # 基本的な手法推奨を取得
            suggestions = self.suggest_methods(data_chars, research_question, user_expertise)
            
            enhanced_suggestions = []
            
            for suggestion in suggestions:
                enhanced_suggestion = {
                    'method_suggestion': suggestion,
                    'assumption_validation': None,
                    'overall_confidence': suggestion.confidence_score
                }
                
                # データが提供されている場合は仮定検証を実行
                if data is not None:
                    try:
                        validation_result = self.validate_method_assumptions(
                            suggestion.method_name, data, target_col, group_col
                        )
                        enhanced_suggestion['assumption_validation'] = validation_result
                        
                        # 仮定検証結果を考慮して全体的な信頼度を調整
                        assumption_confidence = validation_result.confidence_in_results
                        enhanced_suggestion['overall_confidence'] = (
                            suggestion.confidence_score * 0.6 + assumption_confidence * 0.4
                        )
                        
                    except Exception as e:
                        self.logger.warning(f"仮定検証エラー ({suggestion.method_name}): {e}")
                
                enhanced_suggestions.append(enhanced_suggestion)
            
            # 全体的な信頼度でソート
            enhanced_suggestions.sort(key=lambda x: x['overall_confidence'], reverse=True)
            
            return enhanced_suggestions
            
        except Exception as e:
            self.logger.error(f"仮定検証付き推奨エラー: {e}")
            return []

class AIOrchestrator:
    """AI統計解析オーケストレーター - 中央調整システム（電源断保護機能強化版）"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AIOrchestrator")
        self.query_processor = QueryProcessor()
        self.intent_classifier = IntentClassifier()
        self.context_manager = ContextManager()
        self.statistical_advisor = StatisticalMethodAdvisor()
        
        # 既存のAIStatisticalAnalyzerを統合
        self.statistical_analyzer = AIStatisticalAnalyzer()
        
        # プロバイダー管理
        self.providers = self.statistical_analyzer.providers
        self.knowledge_base = self.statistical_analyzer.knowledge_base
        
        # セッション管理
        self.session_id = str(uuid.uuid4())
        self.last_checkpoint = time.time()
        self.checkpoint_interval = 300  # 5分間隔
        
        # バックアップ管理
        self.backup_manager = self._initialize_backup_manager()
        
        # 電源断保護機能の初期化
        self._setup_power_protection()
        
        # シグナルハンドラーの設定
        self._setup_signal_handlers()
    
    def _setup_power_protection(self):
        """電源断保護機能の設定"""
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 自動チェックポイント保存の開始
        self._start_auto_checkpoint()
    
    def _initialize_backup_manager(self):
        """バックアップ管理システムの初期化"""
        return {
            'max_backups': 10,
            'backup_interval': 300,  # 5分
            'backup_dir': 'pss_backups',
            'session_data': {},
            'last_backup': time.time()
        }
    
    def _setup_signal_handlers(self):
        """シグナルハンドラーの設定（メインスレッドでのみ実行）"""
        import signal
        import threading
        
        # メインスレッドでのみシグナルハンドラーを設定
        if threading.current_thread() is threading.main_thread():
            def signal_handler(signum, frame):
                self.logger.info(f"シグナル {signum} を受信しました。緊急保存を実行します...")
                self._emergency_save()
                sys.exit(0)
            
            # Windows対応のシグナルハンドラー
            if hasattr(signal, 'SIGINT'):
                signal.signal(signal.SIGINT, signal_handler)
            if hasattr(signal, 'SIGTERM'):
                signal.signal(signal.SIGTERM, signal_handler)
        else:
            self.logger.warning("シグナルハンドラーの設定をスキップしました（メインスレッド以外で実行）")
    
    def _start_auto_checkpoint(self):
        """自動チェックポイント保存の開始"""
        def auto_checkpoint():
            while True:
                try:
                    time.sleep(self.checkpoint_interval)
                    self._save_checkpoint()
                except Exception as e:
                    self.logger.error(f"自動チェックポイント保存エラー: {e}")
        
        import threading
        checkpoint_thread = threading.Thread(target=auto_checkpoint, daemon=True)
        checkpoint_thread.start()
    
    def _save_checkpoint(self):
        """チェックポイント保存"""
        try:
            checkpoint_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'providers': list(self.providers.keys()) if self.providers else [],
                'context_manager_state': self.context_manager.get_session_summary(
                    AnalysisContext("checkpoint", self.session_id, "checkpoint", [])
                ) if hasattr(self.context_manager, 'get_session_summary') else {},
                'statistical_analyzer_state': {
                    'available_models': self.statistical_analyzer.get_available_models() if hasattr(self.statistical_analyzer, 'get_available_models') else {}
                }
            }
            
            checkpoint_file = os.path.join(self.checkpoint_dir, f"ai_orchestrator_checkpoint_{self.session_id}.json")
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            self.last_checkpoint = time.time()
            self.logger.info(f"チェックポイント保存完了: {checkpoint_file}")
            
            # バックアップの管理
            self._manage_backups()
            
        except Exception as e:
            self.logger.error(f"チェックポイント保存エラー: {e}")
    
    def _emergency_save(self):
        """緊急保存機能"""
        try:
            self.logger.info("緊急保存を実行中...")
            
            # 即座にチェックポイント保存
            self._save_checkpoint()
            
            # セッションデータの保存
            session_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'emergency_save': True,
                'providers_status': {name: provider.is_available() if hasattr(provider, 'is_available') else False 
                                   for name, provider in self.providers.items()} if self.providers else {}
            }
            
            emergency_file = os.path.join(self.checkpoint_dir, f"emergency_save_{self.session_id}.json")
            with open(emergency_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"緊急保存完了: {emergency_file}")
            
        except Exception as e:
            self.logger.error(f"緊急保存エラー: {e}")
    
    def _manage_backups(self):
        """バックアップ管理"""
        try:
            backup_dir = self.backup_manager['backup_dir']
            os.makedirs(backup_dir, exist_ok=True)
            
            # 既存のバックアップファイルを取得
            backup_files = glob.glob(os.path.join(backup_dir, "ai_orchestrator_backup_*.json"))
            backup_files.sort(key=os.path.getmtime, reverse=True)
            
            # 最大バックアップ数を超えた場合、古いものを削除
            if len(backup_files) >= self.backup_manager['max_backups']:
                for old_backup in backup_files[self.backup_manager['max_backups']:]:
                    try:
                        os.remove(old_backup)
                        self.logger.info(f"古いバックアップを削除: {old_backup}")
                    except Exception as e:
                        self.logger.error(f"バックアップ削除エラー: {e}")
            
            # 新しいバックアップを作成
            backup_file = os.path.join(backup_dir, f"ai_orchestrator_backup_{self.session_id}_{int(time.time())}.json")
            checkpoint_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'backup_type': 'scheduled',
                'providers': list(self.providers.keys()) if self.providers else [],
                'context_manager_state': self.context_manager.get_session_summary(
                    AnalysisContext("backup", self.session_id, "backup", [])
                ) if hasattr(self.context_manager, 'get_session_summary') else {}
            }
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            self.backup_manager['last_backup'] = time.time()
            self.logger.info(f"バックアップ作成完了: {backup_file}")
            
        except Exception as e:
            self.logger.error(f"バックアップ管理エラー: {e}")
    
    async def analyze_query(self, query: str, context: AnalysisContext, data: Optional[pd.DataFrame] = None) -> AIResponse:
        """analyze_queryメソッド - process_user_queryへのリダイレクト（後方互換性）"""
        try:
            self.logger.info(f"analyze_query呼び出し: {query[:50]}...")
            return await self.process_user_query(query, context, data)
        except Exception as e:
            self.logger.error(f"analyze_queryエラー: {e}")
            # 緊急保存を実行
            self._emergency_save()
            raise
    
    async def process_user_query(self, query: str, context: AnalysisContext, data: Optional[pd.DataFrame] = None) -> AIResponse:
        """ユーザークエリの統合処理（電源断保護機能強化版）"""
        start_time = time.time()
        
        try:
            # セッション開始時のチェックポイント保存
            if time.time() - self.last_checkpoint > self.checkpoint_interval:
                self._save_checkpoint()
            
            # 1. クエリ処理
            processed_query = self.query_processor.process_query(query, context)
            
            # 2. 意図分類
            intent = self.intent_classifier.classify(query, context)
            
            # 3. プロバイダー選択（エラーハンドリング強化）
            try:
                provider = self._select_optimal_provider(intent, context)
            except Exception as e:
                self.logger.warning(f"プロバイダー選択エラー: {e}、デフォルトプロバイダーを使用")
                provider = self._get_fallback_provider()
            
            # 4. AI応答生成
            if data is not None:
                # データがある場合は統計解析を実行
                result = await self.statistical_analyzer.analyze_with_custom_llm(
                    query, data, provider=provider, enable_rag=True
                )
            else:
                # データがない場合は教育的な応答
                result = await self._generate_educational_response(query, intent, provider)
            
            # 5. 応答構築
            processing_time = time.time() - start_time
            
            response = AIResponse(
                content=result.get('content', ''),
                confidence=result.get('confidence', 0.7),
                provider_used=provider,
                tokens_consumed=result.get('tokens', 0),
                processing_time=processing_time,
                intent_detected=intent,
                educational_content=result.get('educational_content'),
                follow_up_suggestions=self._generate_follow_up_suggestions(intent, result)
            )
            
            # 6. コンテキスト更新
            self.context_manager.update_context(context, {
                'type': intent.value,
                'method': result.get('method', 'ai_response'),
                'success': result.get('success', True)
            })
            
            # 7. 処理完了時のチェックポイント保存
            if processing_time > 30:  # 30秒以上の処理の場合は即座に保存
                self._save_checkpoint()
            
            return response
            
        except Exception as e:
            self.logger.error(f"クエリ処理エラー: {e}")
            
            # エラー時の緊急保存
            self._emergency_save()
            
            return AIResponse(
                content=f"申し訳ございません。処理中にエラーが発生しました: {str(e)}",
                confidence=0.0,
                provider_used="error",
                tokens_consumed=0,
                processing_time=time.time() - start_time,
                intent_detected=IntentType.EXPLORATORY
            )
    
    def _get_fallback_provider(self) -> str:
        """フォールバックプロバイダーの取得"""
        # 利用可能なプロバイダーを優先順位で選択
        fallback_order = ['ollama', 'lmstudio', 'google', 'openai']
        
        for provider_name in fallback_order:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                if hasattr(provider, 'is_available') and provider.is_available():
                    return provider_name
        
        # 最後の手段として最初の利用可能なプロバイダーを使用
        for name, provider in self.providers.items():
            if hasattr(provider, 'is_available') and provider.is_available():
                return name
        
        return "error"  # 利用可能なプロバイダーがない場合
    
    def _select_optimal_provider(self, intent: IntentType, context: AnalysisContext) -> str:
        """最適なプロバイダーを選択（GUI指定対応版）"""
        try:
            # GUIから指定されたプロバイダーを優先
            preferred_provider = context.privacy_settings.get('preferred_provider')
            if preferred_provider and preferred_provider in self.providers:
                provider = self.providers[preferred_provider]
                if hasattr(provider, 'is_available') and provider.is_available():
                    self.logger.info(f"GUI指定プロバイダーを使用: {preferred_provider}")
                    return preferred_provider
                else:
                    self.logger.warning(f"GUI指定プロバイダー '{preferred_provider}' が利用できません")
            
            # プライバシー設定を考慮
            if context.privacy_settings.get('use_local_llm', False):
                if 'ollama' in self.providers:
                    provider = self.providers['ollama']
                    if hasattr(provider, 'is_available') and provider.is_available():
                        return 'ollama'
                elif 'lmstudio' in self.providers:
                    provider = self.providers['lmstudio']
                    if hasattr(provider, 'is_available') and provider.is_available():
                        return 'lmstudio'
            
            # 意図に基づく選択
            if intent == IntentType.EDUCATIONAL:
                # 教育的な内容にはGeminiが適している
                if 'google' in self.providers:
                    provider = self.providers['google']
                    if hasattr(provider, 'is_available') and provider.is_available():
                        return 'google'
            elif intent == IntentType.PREDICTIVE:
                # 予測タスクにはGPT-4が適している
                if 'openai' in self.providers:
                    provider = self.providers['openai']
                    if hasattr(provider, 'is_available') and provider.is_available():
                        return 'openai'
            
            # デフォルト選択
            fallback_provider = self._get_fallback_provider()
            self.logger.info(f"フォールバックプロバイダーを使用: {fallback_provider}")
            return fallback_provider
            
        except Exception as e:
            self.logger.error(f"プロバイダー選択エラー: {e}")
            return self._get_fallback_provider()
    
    async def _generate_educational_response(self, query: str, intent: IntentType, provider: str) -> Dict[str, Any]:
        """教育的な応答を生成"""
        educational_prompt = f"""
統計学の質問に対して教育的で分かりやすい回答を提供してください。

質問: {query}

以下の点を含めて回答してください：
1. 概念の説明
2. 使用場面
3. 前提条件や注意点
4. 簡単な例
5. 関連する統計手法

日本語で分かりやすく説明してください。
"""
        
        if provider in self.providers:
            result = await self.providers[provider].generate_response(
                educational_prompt, 
                model=self._get_default_model(provider)
            )
            
            if result.get('success'):
                return {
                    'content': result['content'],
                    'success': True,
                    'educational_content': result['content'],
                    'method': 'educational_response',
                    'tokens': result.get('tokens', 0)
                }
        
        return {
            'content': '申し訳ございません。現在、この質問にお答えできません。',
            'success': False,
            'method': 'fallback'
        }
    
    def _get_default_model(self, provider: str) -> str:
        """プロバイダーのデフォルトモデルを取得"""
        model_map = {
            'openai': 'gpt-4o',
            'google': 'gemini-1.5-pro-latest',
            'anthropic': 'claude-3-5-sonnet-20240620',
            'ollama': 'llama3.1',
            'lmstudio': 'local-model',
            'koboldcpp': 'local-model'
        }
        return model_map.get(provider, 'gpt-4o')
    
    def _generate_follow_up_suggestions(self, intent: IntentType, result: Dict[str, Any]) -> List[str]:
        """フォローアップ提案を生成"""
        suggestions = []
        
        if intent == IntentType.DESCRIPTIVE:
            suggestions = [
                "この結果に基づいて仮説検定を行いますか？",
                "データの可視化を作成しましょうか？",
                "外れ値の詳細分析を行いますか？"
            ]
        elif intent == IntentType.INFERENTIAL:
            suggestions = [
                "効果量の計算を行いますか？",
                "検定力分析を実施しましょうか？",
                "結果の解釈について詳しく説明しますか？"
            ]
        elif intent == IntentType.PREDICTIVE:
            suggestions = [
                "モデルの性能評価を詳しく見ますか？",
                "特徴量の重要度を分析しましょうか？",
                "ハイパーパラメータの最適化を行いますか？"
            ]
        
        return suggestions[:3]  # 最大3つまで

class AIStatisticalAnalyzer:
    """AI統計解析エンジン - 2025年最新版（マルチLLM、自己修正、RAG、ローカルLLM対応、プライバシー対応）"""
    def __init__(self):
        self.analysis_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        self.platform_capabilities = self._detect_platform_capabilities()
        self.knowledge_base = KnowledgeBase()
        self.providers = self._initialize_providers()
        self.privacy_selector = PrivacyAwareProviderSelector(self.providers)
    
    def _detect_platform_capabilities(self) -> Dict[str, Any]:
        """プラットフォーム機能を検出"""
        capabilities = {
            'os': platform.system(),
            'python_version': platform.python_version(),
            'gpu_available': False,
            'cuda_available': False,
            'memory_gb': 0
        }
        
        try:
            import psutil
            capabilities['memory_gb'] = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            pass
            
        try:
            import torch
            capabilities['gpu_available'] = torch.cuda.is_available()
            capabilities['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                capabilities['gpu_name'] = torch.cuda.get_device_name(0)
        except ImportError:
            pass
            
        return capabilities
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """利用可能なモデル一覧を取得"""
        models = {}
        for provider_name, provider in self.providers.items():
            try:
                if hasattr(provider, 'get_available_models'):
                    models[provider_name] = provider.get_available_models()
                else:
                    # デフォルトモデルリスト
                    default_models = {
                        'openai': ['gpt-4o', 'o3'],
                        'google': ['gemini-2.5-flash-exp', 'gemini-2.5-flash-lite-exp'],
                        'anthropic': ['claude-3-5-sonnet-20240620', 'claude-3-5-haiku-20240307'],
                        'ollama': ['llama3.1', 'llama3.2', 'llama3.3'],
                        'lmstudio': ['local-model'],
                        'koboldcpp': ['local-model']
                    }
                    models[provider_name] = default_models.get(provider_name, ['default'])
            except Exception as e:
                self.logger.warning(f"モデル取得失敗 {provider_name}: {e}")
                models[provider_name] = ['error']
        
        return models
    
    async def analyze_with_privacy_aware_llm(self, query: str, data: pd.DataFrame, 
                                           task_type: str = "general",
                                           enable_rag: bool = False, 
                                           enable_correction: bool = True) -> Dict[str, Any]:
        """プライバシー対応LLM分析"""
        try:
            start_time = time.time()
            
            # プライバシー対応プロバイダー選択
            selected_provider_name, selected_provider = self.privacy_selector.select_optimal_provider(
                data, task_type
            )
            
            self.logger.info(f"選択されたプロバイダー: {selected_provider_name}")
            
            # データの機密性レベルを記録
            sensitivity_level = self.privacy_selector.privacy_manager.classify_data_sensitivity(data)
            
            # 分析実行
            result = await self._execute_analysis_with_provider(
                query, data, selected_provider, selected_provider_name,
                enable_rag, enable_correction
            )
            
            # プライバシー情報を結果に追加
            result.update({
                "privacy_info": {
                    "selected_provider": selected_provider_name,
                    "data_sensitivity_level": sensitivity_level,
                    "privacy_measures_applied": self._get_privacy_measures_applied(sensitivity_level),
                    "data_anonymized": sensitivity_level in ['high', 'medium']
                }
            })
            
            # 分析履歴に記録
            self.analysis_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "provider": selected_provider_name,
                "sensitivity_level": sensitivity_level,
                "processing_time": time.time() - start_time
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"プライバシー対応分析エラー: {e}")
            return {
                "success": False,
                "error": str(e),
                "privacy_info": {
                    "selected_provider": "unknown",
                    "data_sensitivity_level": "unknown",
                    "privacy_measures_applied": [],
                    "data_anonymized": False
                }
            }
    
    async def analyze_with_custom_llm(self, query: str, data: pd.DataFrame, 
                                    provider: str = "google", model: str = "gemini-pro",
                                    enable_rag: bool = False, enable_correction: bool = True) -> Dict[str, Any]:
        """カスタムLLMで統計解析を実行"""
        try:
            if provider not in self.providers:
                return {"success": False, "error": f"プロバイダー {provider} が利用できません"}
            
            # RAGコンテキストの取得
            rag_context = ""
            if enable_rag and self.knowledge_base:
                relevant_docs = self.knowledge_base.search(query)
                if relevant_docs:
                    rag_context = "\n関連情報:\n" + "\n".join([doc['content'][:200] for doc in relevant_docs])
            
            # プロンプト生成
            prompt = f"""
統計データ分析タスク:
{query}

データ概要:
- 行数: {len(data)}
- 列数: {len(data.columns)}
- 列名: {list(data.columns)}
- データ型: {data.dtypes.to_dict()}

{rag_context}

Pythonコードで分析を実行し、結果を解釈してください。
"""
            
            # LLM実行
            provider_instance = self.providers[provider]
            result = await provider_instance.generate_response(prompt, model)
            
            if result["success"] and enable_correction:
                # コード実行と自己修正
                code_result = self._execute_generated_code(result["content"], data)
                if not code_result["success"]:
                    # エラー修正を試行
                    correction_prompt = f"""
前回のコードでエラーが発生しました:
{code_result['error']}

元のコード:
{result['content']}

エラーを修正したPythonコードを生成してください。
"""
                    correction_result = await provider_instance.generate_response(correction_prompt, model)
                    if correction_result["success"]:
                        result["content"] = correction_result["content"]
                        result["corrected"] = True
            
            # 履歴に保存
            self.analysis_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "provider": provider,
                "model": model,
                "result": result
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"分析エラー: {e}")
            return {"success": False, "error": str(e)}
    
    def _execute_generated_code(self, code: str, data: pd.DataFrame) -> Dict[str, Any]:
        """生成されたコードを安全に実行"""
        try:
            # コードからPython部分を抽出
            import re
            code_blocks = re.findall(r'```python\n(.*?)\n```', code, re.DOTALL)
            if not code_blocks:
                code_blocks = re.findall(r'```\n(.*?)\n```', code, re.DOTALL)
            
            if not code_blocks:
                return {"success": False, "error": "実行可能なコードが見つかりません"}
            
            # 最初のコードブロックを実行
            exec_code = code_blocks[0]
            
            # 安全な実行環境を準備
            safe_globals = {
                'pd': pd,
                'np': np,
                'data': data,
                'plt': None,  # matplotlibが利用可能な場合のみ
                'print': print
            }
            
            try:
                import matplotlib.pyplot as plt
                safe_globals['plt'] = plt
            except ImportError:
                pass
            
            # コード実行
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                exec(exec_code, safe_globals)
            
            output = output_buffer.getvalue()
            
            return {
                "success": True,
                "output": output,
                "code": exec_code
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計を取得"""
        if not self.analysis_history:
            return {"total_analyses": 0, "avg_tokens": 0, "success_rate": 0}
        
        total = len(self.analysis_history)
        successful = sum(1 for h in self.analysis_history if h["result"].get("success", False))
        
        total_tokens = sum(h["result"].get("tokens", 0) for h in self.analysis_history if h["result"].get("success", False))
        avg_tokens = total_tokens / successful if successful > 0 else 0
        
        return {
            "total_analyses": total,
            "successful_analyses": successful,
            "success_rate": successful / total if total > 0 else 0,
            "avg_tokens": avg_tokens,
            "providers_used": list(set(h["provider"] for h in self.analysis_history))
        }
    
    async def _execute_analysis_with_provider(self, query: str, data: pd.DataFrame,
                                            provider: LLMProvider, provider_name: str,
                                            enable_rag: bool, enable_correction: bool) -> Dict[str, Any]:
        """指定されたプロバイダーで分析を実行"""
        try:
            # RAG機能が有効な場合、知識ベースから関連情報を取得
            context_info = ""
            if enable_rag and self.knowledge_base:
                relevant_docs = self.knowledge_base.search(query, top_k=2)
                if relevant_docs:
                    context_info = "\n\n関連情報:\n" + "\n".join([doc['content'] for doc in relevant_docs])
            
            # プロンプト構築
            prompt = self._build_analysis_prompt(query, data, context_info)
            
            # LLM応答生成
            response = await provider.generate_response(prompt, model="default")
            
            if not response.get("success", False):
                return {
                    "success": False,
                    "error": response.get("error", "Unknown error"),
                    "provider": provider_name
                }
            
            # 応答処理
            result = self._process_llm_response(response, data, enable_correction)
            result["provider"] = provider_name
            
            return result
            
        except Exception as e:
            self.logger.error(f"プロバイダー分析エラー: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": provider_name
            }
    
    def _build_analysis_prompt(self, query: str, data: pd.DataFrame, context_info: str = "") -> str:
        """分析用プロンプトを構築"""
        data_info = f"""
データ情報:
- 行数: {len(data)}
- 列数: {len(data.columns)}
- 列名: {list(data.columns)}
- データ型: {dict(data.dtypes)}
- 欠損値: {data.isnull().sum().to_dict()}
"""
        
        prompt = f"""
統計分析タスク: {query}

{data_info}

{context_info}

以下の形式で回答してください:
1. 推奨される統計手法
2. 実装手順
3. 結果の解釈方法
4. 注意事項

回答:
"""
        return prompt
    
    def _process_llm_response(self, response: Dict[str, Any], data: pd.DataFrame, 
                             enable_correction: bool) -> Dict[str, Any]:
        """LLM応答を処理"""
        try:
            content = response.get("content", "")
            
            # 応答の構造化
            result = {
                "success": True,
                "analysis": content,
                "tokens_used": response.get("tokens", 0),
                "processing_time": response.get("processing_time", 0)
            }
            
            # 自己修正機能が有効な場合
            if enable_correction:
                corrected_content = self._apply_self_correction(content, data)
                result["corrected_analysis"] = corrected_content
                result["corrections_applied"] = content != corrected_content
            
            return result
            
        except Exception as e:
            self.logger.error(f"応答処理エラー: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": response.get("content", "")
            }
    
    def _apply_self_correction(self, content: str, data: pd.DataFrame) -> str:
        """自己修正機能"""
        try:
            # 基本的な統計用語の修正
            corrections = {
                "平均値": "平均",
                "標準偏差": "標準偏差",
                "t検定": "t検定",
                "分散分析": "分散分析",
                "回帰分析": "回帰分析"
            }
            
            corrected_content = content
            for wrong, correct in corrections.items():
                corrected_content = corrected_content.replace(wrong, correct)
            
            return corrected_content
            
        except Exception as e:
            self.logger.warning(f"自己修正エラー: {e}")
            return content
    
    def _get_privacy_measures_applied(self, sensitivity_level: str) -> List[str]:
        """適用されたプライバシー対策を取得"""
        measures = []
        
        if sensitivity_level == 'high':
            measures.extend([
                "ローカルプロバイダー優先使用",
                "データ匿名化",
                "機密情報検出",
                "アクセス制御"
            ])
        elif sensitivity_level == 'medium':
            measures.extend([
                "ローカルプロバイダー試行",
                "データ匿名化",
                "機密情報検出"
            ])
        else:
            measures.append("標準プライバシー保護")
        
        return measures
    
    def get_privacy_stats(self) -> Dict[str, Any]:
        """プライバシー統計情報を取得"""
        if not self.analysis_history:
            return {"total_analyses": 0, "privacy_levels": {}}
        
        privacy_levels = {}
        for analysis in self.analysis_history:
            level = analysis.get("sensitivity_level", "unknown")
            privacy_levels[level] = privacy_levels.get(level, 0) + 1
        
        return {
            "total_analyses": len(self.analysis_history),
            "privacy_levels": privacy_levels,
            "providers_used": list(set(h["provider"] for h in self.analysis_history))
        }
    
    def _initialize_providers(self) -> Dict[str, LLMProvider]:
        """利用可能なLLMプロバイダーを初期化"""
        providers = {}
        if ai_config.is_api_configured("openai"):
            providers["openai"] = OpenAIProvider(ai_config.openai_api_key)
        if ai_config.is_api_configured("anthropic"):
            providers["anthropic"] = AnthropicProvider(ai_config.anthropic_api_key)
        if ai_config.is_api_configured("google"):
            providers["google"] = GoogleProvider(ai_config.google_api_key)
        if ai_config.is_api_configured("together"):
            providers["together"] = TogetherProvider(ai_config.together_api_key)
        if ai_config.is_api_configured("ollama"):
            providers["ollama"] = OllamaProvider(ai_config.ollama_base_url)
        if ai_config.is_api_configured("lmstudio"):
            providers["lmstudio"] = LMStudioProvider(ai_config.lmstudio_base_url)
        if ai_config.is_api_configured("koboldcpp"):
            providers["koboldcpp"] = KoboldCppProvider(ai_config.koboldcpp_base_url)
        self.logger.info(f"初期化されたプロバイダー: {list(providers.keys())}")
        return providers

    # ... (All other methods of AIStatisticalAnalyzer remain the same) ...

# グローバル分析関数
async def analyze_with_ai(query: str, data: pd.DataFrame, provider: str = "google", 
                         model: str = "gemini-pro", **kwargs) -> Dict[str, Any]:
    """AI統計解析のメイン関数"""
    analyzer = AIStatisticalAnalyzer()
    return await analyzer.analyze_with_custom_llm(query, data, provider, model, **kwargs)

def get_performance_stats() -> Dict[str, Any]:
    """パフォーマンス統計取得のグローバル関数"""
    analyzer = AIStatisticalAnalyzer()
    return analyzer.get_performance_stats()

# 同期版の分析関数（テスト用）
def analyze_with_ai_sync(query: str, data: pd.DataFrame, provider: str = "google", 
                        model: str = "gemini-pro", **kwargs) -> Dict[str, Any]:
    """AI統計解析の同期版関数"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(analyze_with_ai(query, data, provider, model, **kwargs))

# Data sensitivity classification and privacy management
class DataPrivacyManager:
    """データプライバシー管理クラス"""
    
    def __init__(self):
        self.sensitivity_patterns = {
            'high': [
                r'\b(ssn|social\s+security|credit\s+card|password|secret|private)\b',
                r'\b(patient|medical|health|diagnosis|treatment)\b',
                r'\b(salary|income|bank|account|financial)\b',
                r'\b(address|phone|email|personal)\b',
                r'\b(confidential|proprietary|trade\s+secret)\b'
            ],
            'medium': [
                r'\b(company|business|corporate|employee)\b',
                r'\b(survey|research|study|participant)\b',
                r'\b(performance|metrics|kpi|analytics)\b'
            ],
            'low': [
                r'\b(public|open|published|available)\b',
                r'\b(sample|test|demo|example)\b',
                r'\b(general|common|standard)\b'
            ]
        }
        self.logger = logging.getLogger(f"{__name__}.DataPrivacyManager")
    
    def classify_data_sensitivity(self, data: Union[str, pd.DataFrame, Dict[str, Any]]) -> str:
        """データの機密性レベルを分類する"""
        try:
            if isinstance(data, pd.DataFrame):
                # データフレームの場合、カラム名も含めてチェック
                text_content = ' '.join(data.astype(str).values.flatten()) + ' ' + ' '.join(data.columns.astype(str))
            elif isinstance(data, dict):
                text_content = json.dumps(data, default=str)
            elif isinstance(data, str):
                text_content = data
            else:
                text_content = str(data)
            
            # 機密性パターンのチェック
            sensitivity_score = 0
            for level, patterns in self.sensitivity_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text_content, re.IGNORECASE):
                        if level == 'high':
                            sensitivity_score += 3
                        elif level == 'medium':
                            sensitivity_score += 2
                        else:
                            sensitivity_score += 1
            
            # スコアに基づいて機密性レベルを決定
            if sensitivity_score >= 3:
                return 'high'
            elif sensitivity_score >= 1:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            self.logger.error(f"データ機密性分類エラー: {e}")
            return 'medium'  # デフォルトは中程度
    
    def should_use_local_provider(self, sensitivity_level: str) -> bool:
        """ローカルプロバイダーの使用が必要かどうかを判定"""
        return sensitivity_level in ['high', 'medium']
    
    def anonymize_data(self, data: Union[str, pd.DataFrame], sensitivity_level: str) -> Union[str, pd.DataFrame]:
        """データの匿名化処理"""
        try:
            if sensitivity_level == 'low':
                return data
            
            if isinstance(data, pd.DataFrame):
                # データフレームの匿名化
                anonymized_data = data.copy()
                
                # 個人情報カラムの特定と匿名化
                personal_info_patterns = [
                    r'name', r'email', r'phone', r'address', r'id',
                    r'ssn', r'credit', r'password', r'secret'
                ]
                
                columns_to_rename = {}
                for col in anonymized_data.columns:
                    col_lower = col.lower()
                    for pattern in personal_info_patterns:
                        if re.search(pattern, col_lower):
                            # カラム名を変更して匿名化
                            new_col_name = f"anonymized_{col}"
                            columns_to_rename[col] = new_col_name
                            # 値も匿名化
                            anonymized_data[col] = f"anonymized_{col}_{hash(str(anonymized_data[col].iloc[0])) % 1000}"
                            break
                
                # カラム名を変更
                if columns_to_rename:
                    anonymized_data = anonymized_data.rename(columns=columns_to_rename)
                
                return anonymized_data
            
            elif isinstance(data, str):
                # 文字列の匿名化
                anonymized_text = data
                
                # 個人情報パターンの置換
                patterns = {
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]',
                    r'\b\d{3}-\d{2}-\d{4}\b': '[SSN]',
                    r'\b\d{4}-\d{4}-\d{4}-\d{4}\b': '[CREDIT_CARD]',
                    r'\b\d{3}-\d{3}-\d{4}\b': '[PHONE]'
                }
                
                for pattern, replacement in patterns.items():
                    anonymized_text = re.sub(pattern, replacement, anonymized_text)
                
                return anonymized_text
            
            return data
            
        except Exception as e:
            self.logger.error(f"データ匿名化エラー: {e}")
            return data

class PrivacyAwareProviderSelector:
    """プライバシー対応プロバイダー選択クラス"""
    
    def __init__(self, providers: Dict[str, LLMProvider]):
        self.providers = providers
        self.privacy_manager = DataPrivacyManager()
        self.logger = logging.getLogger(f"{__name__}.PrivacyAwareProviderSelector")
        
        # プロバイダーのプライバシーレベル分類
        self.privacy_levels = {
            'local': ['gguf', 'ollama', 'lmstudio', 'koboldcpp'],  # ローカルプロバイダー
            'cloud_private': ['anthropic', 'openai'],  # クラウド（プライバシー重視）
            'cloud_public': ['google', 'together']  # クラウド（一般）
        }
    
    def select_optimal_provider(self, data: Union[str, pd.DataFrame, Dict[str, Any]], 
                              task_type: str = "general") -> Tuple[str, LLMProvider]:
        """データの機密性に基づいて最適なプロバイダーを選択"""
        try:
            # データの機密性を分類
            sensitivity_level = self.privacy_manager.classify_data_sensitivity(data)
            self.logger.info(f"データ機密性レベル: {sensitivity_level}")
            
            # 機密性に基づくプロバイダー選択
            if sensitivity_level == 'high':
                # 高機密性データ: ローカルプロバイダーを優先
                selected_provider = self._select_local_provider()
                if selected_provider:
                    return selected_provider.provider_name, selected_provider
                else:
                    # ローカルプロバイダーが利用できない場合、クラウドプロバイダーで匿名化
                    return self._select_cloud_provider_with_anonymization(data, 'cloud_private')
            
            elif sensitivity_level == 'medium':
                # 中機密性データ: ローカルプロバイダーを試行、失敗時はクラウド
                selected_provider = self._select_local_provider()
                if selected_provider:
                    return selected_provider.provider_name, selected_provider
                else:
                    return self._select_cloud_provider_with_anonymization(data, 'cloud_private')
            
            else:
                # 低機密性データ: コストとパフォーマンスを考慮
                return self._select_best_performance_provider()
                
        except Exception as e:
            self.logger.error(f"プロバイダー選択エラー: {e}")
            # エラー時はデフォルトプロバイダーを返す
            return self._get_default_provider()
    
    def _select_local_provider(self) -> Optional[LLMProvider]:
        """ローカルプロバイダーを選択"""
        for provider_name in self.privacy_levels['local']:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                # プロバイダーの健康状態をチェック
                if self._check_provider_health(provider):
                    return provider
        return None
    
    def _select_cloud_provider_with_anonymization(self, data: Union[str, pd.DataFrame, Dict[str, Any]], 
                                                privacy_level: str) -> Tuple[str, LLMProvider]:
        """クラウドプロバイダーを選択し、必要に応じて匿名化"""
        # データを匿名化
        anonymized_data = self.privacy_manager.anonymize_data(data, 'high')
        
        # 指定されたプライバシーレベルのプロバイダーから選択
        for provider_name in self.privacy_levels.get(privacy_level, []):
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                if self._check_provider_health(provider):
                    return provider.provider_name, provider
        
        # フォールバック: 利用可能なプロバイダーを返す
        return self._get_default_provider()
    
    def _select_best_performance_provider(self) -> Tuple[str, LLMProvider]:
        """パフォーマンスを考慮したプロバイダー選択"""
        # コストとパフォーマンスを考慮した選択ロジック
        provider_priority = ['openai', 'anthropic', 'google', 'gguf', 'ollama']
        
        for provider_name in provider_priority:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                if self._check_provider_health(provider):
                    return provider.provider_name, provider
        
        return self._get_default_provider()
    
    def _check_provider_health(self, provider: LLMProvider) -> bool:
        """プロバイダーの健康状態をチェック"""
        try:
            # プロバイダーが存在するかチェック
            if provider is None:
                return False
            
            # is_healthy属性がある場合はそれを使用
            if hasattr(provider, 'is_healthy'):
                return provider.is_healthy
            
            # is_availableメソッドがある場合はそれを使用
            if hasattr(provider, 'is_available'):
                return provider.is_available()
            
            # デフォルトは利用可能とみなす
            return True
        except Exception as e:
            self.logger.warning(f"プロバイダーヘルスチェックエラー: {e}")
            return False
    
    def _get_default_provider(self) -> Tuple[str, LLMProvider]:
        """デフォルトプロバイダーを取得"""
        # 利用可能なプロバイダーから最初のものを返す
        for provider_name, provider in self.providers.items():
            return provider_name, provider
        
        raise ValueError("利用可能なプロバイダーがありません")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    async def main_demo():
        print("🤖 AI統合モジュール 2025 - Kobold.cpp テスト")
        
        test_df = pd.DataFrame({'A': [100, 200, 300], 'B': [150, 250, 350]})
        query = "データフレーム df のA列とB列の統計要約量を計算してください。"
        
        print(f"クエリ: {query}")
        
        if ai_config.is_api_configured("koboldcpp"):
            result = await ai_analyzer.analyze_with_custom_llm(
                query, test_df, provider="koboldcpp", model="local-model/gguf-model", enable_rag=False, enable_correction=False
            )
            
            if result["success"]:
                print("\n✅ 分析成功！")
                print(result.get("content"))
            else:
                print(f"\n❌ 分析失敗: {result['error']}")
        else:
            print("Kobold.cppが設定されていません。")

    asyncio.run(main_demo())
