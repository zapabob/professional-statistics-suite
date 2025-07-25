#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Latest AI Integration Module - 2025 July 25th Edition
最新AI統合モジュール - 2025年7月25日版（最新AIサービス統合）
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
from dataclasses import dataclass

# Data processing
import pandas as pd
import numpy as np
import requests

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# AI API clients - 最新版対応
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

# Local LLM support
try:
    from lmstudio import LMStudioClient
    LMSTUDIO_AVAILABLE = True
except ImportError:
    LMSTUDIO_AVAILABLE = False

try:
    import llama_cpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

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

# IntentType Enum for AI Orchestrator
class IntentType(Enum):
    """ユーザー意図の分類"""
    DESCRIPTIVE = "descriptive"
    INFERENTIAL = "inferential"
    PREDICTIVE = "predictive"
    EDUCATIONAL = "educational"
    CODE_GENERATION = "code_generation"
    VISUALIZATION = "visualization"
    STATISTICAL_ANALYSIS = "statistical_analysis"

class AIConfig:
    """AI設定管理クラス"""
    
    def __init__(self):
        self.config = {}
        self._load_from_config()
    
    def _load_from_config(self):
        """環境変数から設定を読み込み"""
        # OpenAI設定
        self.config['openai'] = {
            'api_key': os.getenv('OPENAI_API_KEY'),
            'base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
            'models': {
                'gpt4o': os.getenv('OPENAI_MODEL_GPT4O', 'gpt-4o'),
                'gpt4o_mini': os.getenv('OPENAI_MODEL_GPT4O_MINI', 'gpt-4o-mini'),
                'gpt4_turbo': os.getenv('OPENAI_MODEL_GPT4_TURBO', 'gpt-4-turbo'),
                'gpt35_turbo': os.getenv('OPENAI_MODEL_GPT35_TURBO', 'gpt-3.5-turbo')
            }
        }
        
        # Anthropic設定
        self.config['anthropic'] = {
            'api_key': os.getenv('ANTHROPIC_API_KEY'),
            'models': {
                'claude35_sonnet': os.getenv('ANTHROPIC_MODEL_CLAUDE35_SONNET', 'claude-3-5-sonnet-20241022'),
                'claude35_haiku': os.getenv('ANTHROPIC_MODEL_CLAUDE35_HAIKU', 'claude-3-5-haiku-20241022'),
                'claude3_sonnet': os.getenv('ANTHROPIC_MODEL_CLAUDE3_SONNET', 'claude-3-sonnet-20240229'),
                'claude3_haiku': os.getenv('ANTHROPIC_MODEL_CLAUDE3_HAIKU', 'claude-3-haiku-20240307')
            }
        }
        
        # Google AI設定
        self.config['google'] = {
            'api_key': os.getenv('GOOGLE_API_KEY'),
            'models': {
                'gemini15_pro': os.getenv('GOOGLE_MODEL_GEMINI15_PRO', 'gemini-1.5-pro-latest'),
                'gemini15_flash': os.getenv('GOOGLE_MODEL_GEMINI15_FLASH', 'gemini-1.5-flash-latest'),
                'gemini_pro': os.getenv('GOOGLE_MODEL_GEMINI_PRO', 'gemini-pro'),
                'gemini_pro_vision': os.getenv('GOOGLE_MODEL_GEMINI_PRO_VISION', 'gemini-pro-vision')
            }
        }
        
        # ローカルLLM設定
        self.config['local'] = {
            'ollama': {
                'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                'models': {
                    'llama3': os.getenv('OLLAMA_MODEL_LLAMA3', 'llama3'),
                    'llama3_8b': os.getenv('OLLAMA_MODEL_LLAMA3_8B', 'llama3:8b'),
                    'llama3_70b': os.getenv('OLLAMA_MODEL_LLAMA3_70B', 'llama3:70b'),
                    'phi3': os.getenv('OLLAMA_MODEL_PHI3', 'phi3'),
                    'mistral': os.getenv('OLLAMA_MODEL_MISTRAL', 'mistral'),
                    'codestral': os.getenv('OLLAMA_MODEL_CODESTRAL', 'codestral')
                }
            },
            'lmstudio': {
                'base_url': os.getenv('LMSTUDIO_BASE_URL', 'http://localhost:1234'),
                'models_dir': os.getenv('LMSTUDIO_MODELS_DIR', './models'),
                'default_model': os.getenv('LMSTUDIO_DEFAULT_MODEL', 'llama3-8b-instruct')
            },
            'gguf': {
                'models_dir': os.getenv('GGUF_MODELS_DIR', './models'),
                'default_model': os.getenv('GGUF_DEFAULT_MODEL', 'llama3-8b-instruct.Q8_0.gguf'),
                'n_ctx': int(os.getenv('GGUF_N_CTX', '4096')),
                'n_gpu_layers': int(os.getenv('GGUF_N_GPU_LAYERS', '0'))
            },
            'koboldcpp': {
                'base_url': os.getenv('KOBOLDCPP_BASE_URL', 'http://localhost:5001'),
                'default_model': os.getenv('KOBOLDCPP_DEFAULT_MODEL', 'llama3-8b-instruct')
            }
        }
    
    def is_api_configured(self, provider: str) -> bool:
        """API設定の確認"""
        if provider == 'openai':
            return bool(self.config['openai']['api_key'])
        elif provider == 'anthropic':
            return bool(self.config['anthropic']['api_key'])
        elif provider == 'google':
            return bool(self.config['google']['api_key'])
        elif provider in ['ollama', 'lmstudio', 'gguf', 'koboldcpp']:
            return True  # ローカルLLMは設定不要
        return False

class LLMProvider:
    """LLMプロバイダー基底クラス"""
    
    def __init__(self, provider_name: str, api_key: Optional[str] = None):
        self.provider_name = provider_name
        self.api_key = api_key
    
    async def generate_response(self, prompt: str, model: str, **kwargs) -> Dict[str, Any]:
        """応答生成（非同期）"""
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    """OpenAIプロバイダー（最新モデル対応）"""
    
    def __init__(self, api_key: str):
        super().__init__("openai", api_key)
        self.client = OpenAI(api_key=api_key)
    
    async def generate_response(self, prompt: str, model: str = "gpt-4o", **kwargs) -> Dict[str, Any]:
        """OpenAI応答生成"""
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            
            return {
                "content": response.choices[0].message.content,
                "provider": "openai",
                "model": model,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "success": True
            }
        except Exception as e:
            return {
                "content": f"OpenAI API エラー: {str(e)}",
                "provider": "openai",
                "model": model,
                "error": str(e),
                "success": False
            }

class AnthropicProvider(LLMProvider):
    """Anthropicプロバイダー（最新モデル対応）"""
    
    def __init__(self, api_key: str):
        super().__init__("anthropic", api_key)
        self.client = Anthropic(api_key=api_key)
    
    async def generate_response(self, prompt: str, model: str = "claude-3-5-sonnet-20241022", **kwargs) -> Dict[str, Any]:
        """Anthropic応答生成"""
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=model,
                max_tokens=kwargs.get('max_tokens', 1000),
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                "content": response.content[0].text,
                "provider": "anthropic",
                "model": model,
                "tokens_used": response.usage.input_tokens + response.usage.output_tokens if response.usage else 0,
                "success": True
            }
        except Exception as e:
            return {
                "content": f"Anthropic API エラー: {str(e)}",
                "provider": "anthropic",
                "model": model,
                "error": str(e),
                "success": False
            }

class GoogleProvider(LLMProvider):
    """Google AIプロバイダー（最新モデル対応）"""
    
    def __init__(self, api_key: str = None):
        super().__init__("google", api_key)
        if api_key:
            genai.configure(api_key=api_key)
        self.client = genai
    
    def _get_api_key(self, api_key: str = None) -> str:
        """APIキー取得"""
        if api_key:
            return api_key
        return os.getenv('GOOGLE_API_KEY', '')
    
    def is_available(self) -> bool:
        """利用可能性確認"""
        try:
            api_key = self._get_api_key()
            if not api_key:
                return False
            genai.configure(api_key=api_key)
            return True
        except Exception:
            return False
    
    async def generate_response(self, prompt: str, model: str = "gemini-1.5-pro-latest", **kwargs) -> Dict[str, Any]:
        """Google AI応答生成"""
        try:
            def _generate_sync():
                genai.configure(api_key=self._get_api_key())
                model_instance = genai.GenerativeModel(model)
                response = model_instance.generate_content(prompt)
                return response
            
            response = await asyncio.to_thread(_generate_sync)
            
            return {
                "content": response.text,
                "provider": "google",
                "model": model,
                "tokens_used": 0,  # Google AIはトークン数情報を提供しない
                "success": True
            }
        except Exception as e:
            return {
                "content": f"Google AI API エラー: {str(e)}",
                "provider": "google",
                "model": model,
                "error": str(e),
                "success": False
            }

class OllamaProvider(LLMProvider):
    """Ollamaプロバイダー（ローカルLLM）"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__("ollama")
        self.base_url = base_url
    
    async def generate_response(self, prompt: str, model: str = "llama3", **kwargs) -> Dict[str, Any]:
        """Ollama応答生成"""
        try:
            url = f"{self.base_url}/api/generate"
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                **kwargs
            }
            
            async with asyncio.timeout(30):
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as response:
                        result = await response.json()
            
            return {
                "content": result.get("response", ""),
                "provider": "ollama",
                "model": model,
                "tokens_used": result.get("eval_count", 0),
                "success": True
            }
        except Exception as e:
            return {
                "content": f"Ollama API エラー: {str(e)}",
                "provider": "ollama",
                "model": model,
                "error": str(e),
                "success": False
            }

class LMStudioProvider(LLMProvider):
    """LM Studioプロバイダー（ローカルLLM）"""
    
    def __init__(self, base_url: str = "http://localhost:1234", models_dir: str = "./models"):
        super().__init__("lmstudio")
        self.base_url = base_url
        self.models_dir = models_dir
        self.available_models = []
        self._scan_gguf_models()
    
    def _scan_gguf_models(self) -> List[str]:
        """GGUFモデルスキャン"""
        try:
            models_path = Path(self.models_dir)
            if models_path.exists():
                gguf_files = list(models_path.glob("*.gguf"))
                self.available_models = [f.stem for f in gguf_files]
            return self.available_models
        except Exception as e:
            logging.error(f"GGUFモデルスキャンエラー: {e}")
            return []
    
    async def generate_response(self, prompt: str, model: str = None, **kwargs) -> Dict[str, Any]:
        """LM Studio応答生成"""
        try:
            if not model and self.available_models:
                model = self.available_models[0]
            
            url = f"{self.base_url}/v1/chat/completions"
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                **kwargs
            }
            
            async with asyncio.timeout(30):
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as response:
                        result = await response.json()
            
            return {
                "content": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "provider": "lmstudio",
                "model": model,
                "tokens_used": result.get("usage", {}).get("total_tokens", 0),
                "success": True
            }
        except Exception as e:
            return {
                "content": f"LM Studio API エラー: {str(e)}",
                "provider": "lmstudio",
                "model": model,
                "error": str(e),
                "success": False
            }

class GGUFProvider(LLMProvider):
    """GGUF直接プロバイダー（ローカルLLM）"""
    
    def __init__(self, model_path: str = None, n_ctx: int = 4096, n_gpu_layers: int = 0):
        super().__init__("gguf")
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """モデル読み込み"""
        try:
            if self.model_path and Path(self.model_path).exists():
                self.model = llama_cpp.Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers
                )
        except Exception as e:
            logging.error(f"GGUFモデル読み込みエラー: {e}")
    
    async def generate_response(self, prompt: str, model: str = None, **kwargs) -> Dict[str, Any]:
        """GGUF直接応答生成"""
        try:
            if not self.model:
                return {
                    "content": "GGUFモデルが読み込まれていません",
                    "provider": "gguf",
                    "model": model,
                    "error": "Model not loaded",
                    "success": False
                }
            
            response = await asyncio.to_thread(
                self.model,
                prompt,
                max_tokens=kwargs.get('max_tokens', 512),
                temperature=kwargs.get('temperature', 0.3),
                **kwargs
            )
            
            return {
                "content": response['choices'][0]['text'],
                "provider": "gguf",
                "model": self.model_path,
                "tokens_used": len(response['choices'][0]['text'].split()),
                "success": True
            }
        except Exception as e:
            return {
                "content": f"GGUF API エラー: {str(e)}",
                "provider": "gguf",
                "model": model,
                "error": str(e),
                "success": False
            }

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

class LatestAIOrchestrator:
    """最新AIオーケストレーター（2025年7月25日版）"""
    
    def __init__(self):
        self.config = AIConfig()
        self.providers = {}
        self._initialize_providers()
        self._setup_power_protection()
    
    def _initialize_providers(self):
        """プロバイダー初期化"""
        # OpenAI
        if OPENAI_AVAILABLE and self.config.is_api_configured('openai'):
            self.providers['openai'] = OpenAIProvider(self.config.config['openai']['api_key'])
        
        # Anthropic
        if ANTHROPIC_AVAILABLE and self.config.is_api_configured('anthropic'):
            self.providers['anthropic'] = AnthropicProvider(self.config.config['anthropic']['api_key'])
        
        # Google AI
        if GOOGLE_AI_AVAILABLE and self.config.is_api_configured('google'):
            self.providers['google'] = GoogleProvider(self.config.config['google']['api_key'])
        
        # ローカルLLM
        if LMSTUDIO_AVAILABLE:
            self.providers['lmstudio'] = LMStudioProvider(
                self.config.config['local']['lmstudio']['base_url'],
                self.config.config['local']['lmstudio']['models_dir']
            )
        
        if LLAMA_CPP_AVAILABLE:
            self.providers['gguf'] = GGUFProvider(
                self.config.config['local']['gguf']['default_model'],
                self.config.config['local']['gguf']['n_ctx'],
                self.config.config['local']['gguf']['n_gpu_layers']
            )
        
        self.providers['ollama'] = OllamaProvider(
            self.config.config['local']['ollama']['base_url']
        )
    
    def _setup_power_protection(self):
        """電源断保護設定"""
        def signal_handler(signum, frame):
            logging.info("🛡️ 電源断保護: 緊急保存を実行中...")
            self._emergency_save()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _emergency_save(self):
        """緊急保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path("emergency_backups")
            backup_dir.mkdir(exist_ok=True)
            
            backup_data = {
                "timestamp": timestamp,
                "providers": list(self.providers.keys()),
                "config": self.config.config
            }
            
            backup_file = backup_dir / f"emergency_backup_{timestamp}.json"
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"🛡️ 緊急保存完了: {backup_file}")
        except Exception as e:
            logging.error(f"緊急保存エラー: {e}")
    
    async def analyze_query(self, query: str, data: Optional[pd.DataFrame] = None, 
                           preferred_provider: str = None) -> AIResponse:
        """クエリ分析（最新AI統合）"""
        start_time = time.time()
        
        # プロバイダー選択
        provider = self._select_optimal_provider(query, preferred_provider)
        
        # プロンプト構築
        prompt = self._build_analysis_prompt(query, data)
        
        # AI応答生成
        response = await self.providers[provider].generate_response(prompt, **self._get_model_params(provider))
        
        processing_time = time.time() - start_time
        
        return AIResponse(
            content=response.get('content', ''),
            confidence=0.9 if response.get('success') else 0.1,
            provider_used=provider,
            tokens_consumed=response.get('tokens_used', 0),
            processing_time=processing_time,
            intent_detected=self._classify_intent(query),
            educational_content=self._generate_educational_content(query, response.get('content', '')),
            follow_up_suggestions=self._generate_follow_up_suggestions(query, response.get('content', ''))
        )
    
    def _select_optimal_provider(self, query: str, preferred_provider: str = None) -> str:
        """最適プロバイダー選択"""
        if preferred_provider and preferred_provider in self.providers:
            return preferred_provider
        
        # 統計分析クエリの場合はローカルLLMを優先
        if self._is_statistical_query(query):
            for provider in ['gguf', 'lmstudio', 'ollama']:
                if provider in self.providers:
                    return provider
        
        # デフォルト優先順位
        priority_order = ['openai', 'anthropic', 'google', 'gguf', 'lmstudio', 'ollama']
        
        for provider in priority_order:
            if provider in self.providers:
                return provider
        
        return list(self.providers.keys())[0] if self.providers else 'openai'
    
    def _is_statistical_query(self, query: str) -> bool:
        """統計分析クエリ判定"""
        statistical_keywords = [
            't検定', 'anova', '回帰', '相関', '分散', '平均', '中央値',
            't-test', 'regression', 'correlation', 'variance', 'mean', 'median'
        ]
        return any(keyword in query.lower() for keyword in statistical_keywords)
    
    def _build_analysis_prompt(self, query: str, data: Optional[pd.DataFrame] = None) -> str:
        """分析プロンプト構築"""
        prompt = f"""
あなたは統計分析の専門家です。以下のクエリに対して詳細で正確な回答を提供してください。

クエリ: {query}

"""
        
        if data is not None:
            prompt += f"""
データ情報:
- 行数: {len(data)}
- 列数: {len(data.columns)}
- 列名: {list(data.columns)}
- データ型: {data.dtypes.to_dict()}
- 欠損値: {data.isnull().sum().to_dict()}

"""
        
        prompt += """
回答形式:
1. 分析手法の説明
2. 実装方法
3. 結果の解釈
4. 注意点や制限事項

日本語で回答してください。
"""
        
        return prompt
    
    def _get_model_params(self, provider: str) -> Dict[str, Any]:
        """モデルパラメータ取得"""
        if provider == 'openai':
            return {
                'model': self.config.config['openai']['models']['gpt4o'],
                'max_tokens': 2000,
                'temperature': 0.3
            }
        elif provider == 'anthropic':
            return {
                'model': self.config.config['anthropic']['models']['claude35_sonnet'],
                'max_tokens': 2000
            }
        elif provider == 'google':
            return {
                'model': self.config.config['google']['models']['gemini15_pro']
            }
        else:
            return {
                'max_tokens': 1000,
                'temperature': 0.3
            }
    
    def _classify_intent(self, query: str) -> IntentType:
        """意図分類"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['説明', '解説', '教えて', 'what', 'explain']):
            return IntentType.EDUCATIONAL
        elif any(word in query_lower for word in ['コード', '実装', 'code', 'implement']):
            return IntentType.CODE_GENERATION
        elif any(word in query_lower for word in ['グラフ', '可視化', 'plot', 'visualize']):
            return IntentType.VISUALIZATION
        elif any(word in query_lower for word in ['分析', '統計', 'analysis', 'statistics']):
            return IntentType.STATISTICAL_ANALYSIS
        elif any(word in query_lower for word in ['予測', 'prediction', 'forecast']):
            return IntentType.PREDICTIVE
        else:
            return IntentType.DESCRIPTIVE
    
    def _generate_educational_content(self, query: str, response: str) -> str:
        """教育コンテンツ生成"""
        return f"📚 学習ポイント:\n{response[:200]}..."
    
    def _generate_follow_up_suggestions(self, query: str, response: str) -> List[str]:
        """フォローアップ提案生成"""
        suggestions = [
            "より詳細な分析を行いますか？",
            "可視化を追加しますか？",
            "他の統計手法も試してみますか？"
        ]
        return suggestions[:3]
    
    def get_available_providers(self) -> List[str]:
        """利用可能プロバイダー取得"""
        return list(self.providers.keys())
    
    def get_provider_status(self) -> Dict[str, bool]:
        """プロバイダー状態取得"""
        return {provider: True for provider in self.providers.keys()}

# 非同期HTTPクライアント用
import aiohttp

# 使用例
async def main():
    """メイン関数"""
    orchestrator = LatestAIOrchestrator()
    
    # 利用可能プロバイダー確認
    providers = orchestrator.get_available_providers()
    print(f"利用可能プロバイダー: {providers}")
    
    # テストクエリ
    query = "t検定について詳しく教えてください"
    response = await orchestrator.analyze_query(query)
    
    print(f"プロバイダー: {response.provider_used}")
    print(f"応答: {response.content}")
    print(f"処理時間: {response.processing_time:.2f}秒")

if __name__ == "__main__":
    asyncio.run(main()) 