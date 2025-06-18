#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Integration Module
AI統合モジュール - OpenAI, Google AI Studio, Anthropic, 画像処理, 自然言語処理
最新版API対応 (2024年対応)
"""

import asyncio
import base64
import re
import json
import os
import platform
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import traceback
import logging

# Data processing
import pandas as pd
import numpy as np

# AI API clients (オプション)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Image processing (オプション)
try:
    from PIL import Image
    import cv2
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# IMAGE_PROCESSING_AVAILABLE の定義
IMAGE_PROCESSING_AVAILABLE = PIL_AVAILABLE and (PYTESSERACT_AVAILABLE or EASYOCR_AVAILABLE)

# Configuration
try:
    from config import ai_config
except ImportError:
    # デフォルト設定
    class MockConfig:
        def __init__(self):
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            self.mcp_server_url = "ws://localhost:8080"
            self.mcp_enabled = False
            self.ocr_languages = ["eng", "jpn"]
            self.tesseract_cmd = None
        
        def is_api_configured(self, provider: str) -> bool:
            if provider == "openai":
                return bool(self.openai_api_key)
            elif provider == "google":
                return bool(self.google_api_key)
            elif provider == "anthropic":
                return bool(self.anthropic_api_key)
            return False
    
    ai_config = MockConfig()

# プロンプトテンプレート（マルチプラットフォーム対応）
STATISTICAL_ANALYSIS_PROMPTS = {
    "natural_language_query": "分析要求: {user_query}\nプラットフォーム情報: {platform_info}",
    "image_data_extraction": "画像からデータを抽出してください。GPU最適化: {gpu_optimization}",
    "code_generation": """
データサイエンスの専門家として、以下の要求に基づいて実行可能なPythonコードを生成してください。

データ情報:
{data_info}

システム情報:
{platform_info}

ユーザー要求: {user_query}

要件:
1. pandasとnumpyを使用
2. 必要に応じてmatplotlib/seabornでビジュアライゼーション
3. エラーハンドリングを含める
4. 日本語コメントを追加
5. 利用可能なGPU最適化を活用（{gpu_platform}）

```python
# 生成されたコード
```
"""
}

class AIStatisticalAnalyzer:
    """AI統計解析エンジン - 最新API対応版"""
    
    def __init__(self):
        self.analysis_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    async def analyze_natural_language_query(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """自然言語クエリで統計解析（マルチプラットフォーム対応）"""
        try:
            # プラットフォーム情報の準備
            platform_info = self._get_platform_context()
            
            # AI API が利用可能な場合の優先順位で実行
            if OPENAI_AVAILABLE and ai_config.is_api_configured("openai"):
                result = await self._analyze_with_openai(query, data, platform_info)
            elif ANTHROPIC_AVAILABLE and ai_config.is_api_configured("anthropic"):
                result = await self._analyze_with_anthropic(query, data, platform_info)
            elif GOOGLE_AI_AVAILABLE and ai_config.is_api_configured("google"):
                result = await self._analyze_with_google(query, data, platform_info)
            else:
                # ローカル解析
                result = self._analyze_locally(query, data)
            
            # 分析履歴に記録
            self.analysis_history.append({
                "query": query,
                "timestamp": pd.Timestamp.now(),
                "result": result,
                "platform_info": platform_info
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"分析エラー: {str(e)}")
            return {
                "success": False, 
                "error": str(e), 
                "traceback": traceback.format_exc(),
                "platform_info": self.platform_capabilities
            }
    
    def _get_platform_context(self) -> Dict[str, Any]:
        """プラットフォームコンテキストを取得"""
        context = {
            'platform': self.platform_capabilities['platform'],
            'gpu_platform': self.platform_capabilities['gpu_platform'],
            'gpu_acceleration': self.platform_capabilities['gpu_acceleration'],
            'optimization_level': self.platform_capabilities['optimization_level']
        }
        
        # プラットフォーム固有の推奨事項
        if self.platform_capabilities['gpu_platform'] == 'mps':
            context['recommendations'] = [
                "Apple Siliconの統合メモリアーキテクチャを活用",
                "PyTorchのMPSバックエンドを使用可能",
                "Metal Performance Shadersによる最適化が可能"
            ]
        elif self.platform_capabilities['gpu_platform'] == 'cuda':
            context['recommendations'] = [
                "NVIDIA CUDAによる高速GPU計算",
                "Tensor Coresを活用した混合精度計算",
                "CuPyによる高速数値計算が利用可能"
            ]
        elif self.platform_capabilities['gpu_platform'] == 'rocm':
            context['recommendations'] = [
                "AMD ROCmによるGPU加速",
                "OpenCLバックエンドでの計算最適化",
                "AMD GPU特化の最適化が可能"
            ]
        else:
            context['recommendations'] = [
                "CPUベースの高速化（NumBA、マルチプロセシング）",
                "メモリ効率の最適化",
                "並列処理による性能向上"
            ]
        
        return context

    async def _analyze_with_openai(self, query: str, data: pd.DataFrame, platform_info: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI APIで分析 - マルチプラットフォーム対応"""
        try:
            client = openai.OpenAI(api_key=ai_config.openai_api_key)
            
            data_info = f"""
- 形状: {data.shape}
- 列: {list(data.columns)}
- 型: {data.dtypes.to_dict()}
- 欠損値: {data.isnull().sum().to_dict()}
"""
            
            prompt = STATISTICAL_ANALYSIS_PROMPTS["code_generation"].format(
                data_info=data_info,
                platform_info=json.dumps(platform_info, ensure_ascii=False, indent=2),
                user_query=query,
                gpu_platform=platform_info['gpu_platform']
            )
            
            # モデル選択（プラットフォームに応じて）
            model = "gpt-4o"  # 最新の安定版モデル
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"あなたは統計分析とデータサイエンスの専門家です。{platform_info['gpu_platform']}プラットフォームに最適化された実行可能で安全なPythonコードを生成してください。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4096
            )
            
            ai_response = response.choices[0].message.content or ""
            
            # コード抽出
            code_pattern = r'```python\n(.*?)\n```'
            code_matches = re.findall(code_pattern, ai_response, re.DOTALL)
            
            return {
                "success": True,
                "provider": "openai",
                "model": model,
                "response": ai_response,
                "extracted_code": code_matches[0] if code_matches else None,
                "data_info": data_info,
                "platform_optimization": platform_info,
                "gpu_accelerated": platform_info['gpu_acceleration']
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI分析エラー: {str(e)}")
            return {"success": False, "error": str(e), "provider": "openai"}
    
    async def _analyze_with_anthropic(self, query: str, data: pd.DataFrame, platform_info: Dict[str, Any]) -> Dict[str, Any]:
        """Anthropic APIで分析 - マルチプラットフォーム対応"""
        try:
            client = anthropic.Anthropic(api_key=ai_config.anthropic_api_key)
            
            data_info = f"""
- 形状: {data.shape}
- 列: {list(data.columns)}
- 型: {data.dtypes.to_dict()}
- 欠損値: {data.isnull().sum().to_dict()}
"""
            
            prompt = STATISTICAL_ANALYSIS_PROMPTS["code_generation"].format(
                data_info=data_info,
                platform_info=json.dumps(platform_info, ensure_ascii=False, indent=2),
                user_query=query,
                gpu_platform=platform_info['gpu_platform']
            )
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                temperature=0.1,
                system="あなたは統計分析とデータサイエンスの専門家です。実行可能で安全なPythonコードを生成してください。",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            ai_response = response.content[0].text if response.content else ""
            
            # コード抽出
            code_pattern = r'```python\n(.*?)\n```'
            code_matches = re.findall(code_pattern, ai_response, re.DOTALL)
            
            return {
                "success": True,
                "provider": "anthropic",
                "model": "claude-3-5-sonnet-20241022",
                "response": ai_response,
                "extracted_code": code_matches[0] if code_matches else None,
                "data_info": data_info,
                "platform_optimization": platform_info,
                "gpu_accelerated": platform_info['gpu_acceleration']
            }
            
        except Exception as e:
            self.logger.error(f"Anthropic分析エラー: {str(e)}")
            return {"success": False, "error": str(e), "provider": "anthropic"}
    
    async def _analyze_with_google(self, query: str, data: pd.DataFrame, platform_info: Dict[str, Any]) -> Dict[str, Any]:
        """Google AI Studioで分析"""
        try:
            genai.configure(api_key=ai_config.google_api_key)
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            data_info = f"""
- 形状: {data.shape}
- 列: {list(data.columns)}
- 型: {data.dtypes.to_dict()}
- 欠損値: {data.isnull().sum().to_dict()}
"""
            
            prompt = STATISTICAL_ANALYSIS_PROMPTS["code_generation"].format(
                data_info=data_info,
                platform_info=json.dumps(platform_info, ensure_ascii=False, indent=2),
                user_query=query,
                gpu_platform=platform_info['gpu_platform']
            )
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=4096,
                    temperature=0.1
                )
            )
            
            ai_response = response.text if response.text else ""
            
            # コード抽出
            code_pattern = r'```python\n(.*?)\n```'
            code_matches = re.findall(code_pattern, ai_response, re.DOTALL)
            
            return {
                "success": True,
                "provider": "google",
                "model": "gemini-1.5-pro",
                "response": ai_response,
                "extracted_code": code_matches[0] if code_matches else None,
                "data_info": data_info,
                "platform_optimization": platform_info,
                "gpu_accelerated": platform_info['gpu_acceleration']
            }
            
        except Exception as e:
            self.logger.error(f"Google AI分析エラー: {str(e)}")
            return {"success": False, "error": str(e), "provider": "google"}
    
    def _analyze_locally(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """ローカル分析（AIなし）"""
        try:
            # 基本的な統計情報
            data_info = {
                "shape": data.shape,
                "columns": list(data.columns),
                "dtypes": data.dtypes.to_dict(),
                "missing_values": data.isnull().sum().to_dict(),
                "basic_stats": data.describe().to_dict() if data.select_dtypes(include=[np.number]).shape[1] > 0 else {}
            }
            
            # クエリキーワード解析
            query_lower = query.lower()
            suggested_analysis = []
            
            if any(word in query_lower for word in ['相関', 'correlation', '関係']):
                suggested_analysis.append("correlation_analysis")
            if any(word in query_lower for word in ['回帰', 'regression', '予測']):
                suggested_analysis.append("regression_analysis")
            if any(word in query_lower for word in ['検定', 'test', 't検定', 'ttest']):
                suggested_analysis.append("statistical_test")
            if any(word in query_lower for word in ['可視化', 'plot', 'graph', 'chart']):
                suggested_analysis.append("visualization")
            
            return {
                "success": True,
                "provider": "local",
                "query": query,
                "data_info": data_info,
                "suggested_analysis": suggested_analysis,
                "message": "AI APIが利用できないため、ローカル分析を実行しました。データ情報と推奨分析手法を提供します。"
            }
            
        except Exception as e:
            self.logger.error(f"ローカル分析エラー: {str(e)}")
            return {"success": False, "error": str(e), "provider": "local"}

    async def extract_data_from_image(self, image_path: str) -> Dict[str, Any]:
        """画像からデータを抽出"""
        if not IMAGE_PROCESSING_AVAILABLE:
            return {"success": False, "error": "画像処理ライブラリが利用できません"}
        
        try:
            # 画像読み込み
            image = Image.open(image_path)
            
            # OCR実行（優先順位: EasyOCR > Tesseract）
            extracted_text = ""
            if EASYOCR_AVAILABLE:
                reader = easyocr.Reader(ai_config.ocr_languages)
                results = reader.readtext(str(image_path))
                extracted_text = "\n".join([result[1] for result in results])
            elif PYTESSERACT_AVAILABLE:
                if ai_config.tesseract_cmd:
                    pytesseract.pytesseract.tesseract_cmd = ai_config.tesseract_cmd
                extracted_text = pytesseract.image_to_string(image, lang='+'.join(ai_config.ocr_languages))
            
            # テキストからデータパターンを探索
            data_patterns = self._extract_data_patterns(extracted_text)
            
            return {
                "success": True,
                "image_path": image_path,
                "extracted_text": extracted_text,
                "data_patterns": data_patterns,
                "image_size": image.size
            }
            
        except Exception as e:
            self.logger.error(f"画像データ抽出エラー: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _extract_data_patterns(self, text: str) -> Dict[str, Any]:
        """テキストからデータパターンを抽出"""
        patterns = {
            "numbers": re.findall(r'-?\d+\.?\d*', text),
            "percentages": re.findall(r'\d+\.?\d*%', text),
            "dates": re.findall(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', text),
            "tables": []
        }
        
        # 簡単な表形式データの検出
        lines = text.split('\n')
        for line in lines:
            if '\t' in line or ',' in line:
                cells = re.split(r'[\t,]', line.strip())
                if len(cells) > 1:
                    patterns["tables"].append(cells)
        
        return patterns

    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """分析履歴を取得"""
        return self.analysis_history
    
    def clear_analysis_history(self):
        """分析履歴をクリア"""
        self.analysis_history.clear()

# グローバルインスタンス
ai_analyzer = AIStatisticalAnalyzer()

def analyze_with_ai(query: str, data: pd.DataFrame) -> Dict[str, Any]:
    """AI分析のシンプルなインターフェース"""
    try:
        import asyncio
        
        # イベントループの取得または作成
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 非同期関数を実行
        return loop.run_until_complete(ai_analyzer.analyze_natural_language_query(query, data))
    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_image_data(image_path: str) -> Dict[str, Any]:
    """画像データ抽出のシンプルなインターフェース"""
    try:
        import asyncio
        
        # イベントループの取得または作成
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # 非同期関数を実行
        return loop.run_until_complete(ai_analyzer.extract_data_from_image(image_path))
    except Exception as e:
        return {"success": False, "error": str(e)}
