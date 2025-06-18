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

# プロンプトテンプレート
STATISTICAL_ANALYSIS_PROMPTS = {
    "natural_language_query": "分析要求: {user_query}",
    "image_data_extraction": "画像からデータを抽出してください。",
    "code_generation": """
データサイエンスの専門家として、以下の要求に基づいて実行可能なPythonコードを生成してください。

データ情報:
{data_info}

ユーザー要求: {user_query}

要件:
1. pandasとnumpyを使用
2. 必要に応じてmatplotlib/seabornでビジュアライゼーション
3. エラーハンドリングを含める
4. 日本語コメントを追加

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
        """自然言語クエリで統計解析"""
        try:
            # AI API が利用可能な場合の優先順位で実行
            if OPENAI_AVAILABLE and ai_config.is_api_configured("openai"):
                result = await self._analyze_with_openai(query, data)
            elif ANTHROPIC_AVAILABLE and ai_config.is_api_configured("anthropic"):
                result = await self._analyze_with_anthropic(query, data)
            elif GOOGLE_AI_AVAILABLE and ai_config.is_api_configured("google"):
                result = await self._analyze_with_google(query, data)
            else:
                # ローカル解析
                result = self._analyze_locally(query, data)
            
            # 分析履歴に記録
            self.analysis_history.append({
                "query": query,
                "timestamp": pd.Timestamp.now(),
                "result": result
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"分析エラー: {str(e)}")
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    
    async def _analyze_with_openai(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """OpenAI APIで分析 - 最新API対応"""
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
                user_query=query
            )
            
            response = client.chat.completions.create(
                model="gpt-4o",  # 最新の安定版モデル
                messages=[
                    {"role": "system", "content": "あなたは統計分析とデータサイエンスの専門家です。実行可能で安全なPythonコードを生成してください。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=4096
            )
            
            ai_response = response.choices[0].message.content or ""
            code = self._extract_python_code(ai_response)
            
            return {
                "success": True,
                "ai_response": ai_response,
                "python_code": code,
                "provider": "OpenAI",
                "model": "gpt-4o"
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI API エラー: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _analyze_with_google(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Google AI Studioで分析 - 最新API対応"""
        try:
            if not GOOGLE_AI_AVAILABLE:
                return {"success": False, "error": "Google AI APIが利用できません"}
                
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
                user_query=query
            )
            
            response = model.generate_content(prompt)
            ai_response = response.text if response.text else ""
            code = self._extract_python_code(ai_response)
            
            return {
                "success": True,
                "ai_response": ai_response,
                "python_code": code,
                "provider": "Google",
                "model": "gemini-1.5-pro"
            }
            
        except Exception as e:
            self.logger.error(f"Google AI API エラー: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _analyze_with_anthropic(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Anthropic APIで分析 - 最新API対応"""
        try:
            if not ANTHROPIC_AVAILABLE:
                return {"success": False, "error": "Anthropic APIが利用できません"}
                
            client = anthropic.Anthropic(api_key=ai_config.anthropic_api_key)
            model = "claude-3-5-sonnet-20241022"  # 最新版
            
            data_info = f"""
- 形状: {data.shape}
- 列: {list(data.columns)}
- 型: {data.dtypes.to_dict()}
- 欠損値: {data.isnull().sum().to_dict()}
"""
            
            prompt = STATISTICAL_ANALYSIS_PROMPTS["code_generation"].format(
                data_info=data_info,
                user_query=query
            )
            
            response = client.messages.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4096,
                system="あなたは統計分析とデータサイエンスの専門家です。実行可能で安全なPythonコードを生成してください。"
            )
            
            ai_response = ""
            if response.content and len(response.content) > 0:
                content_block = response.content[0]
                if hasattr(content_block, 'text'):
                    ai_response = getattr(content_block, 'text')
                else:
                    ai_response = str(content_block)
            
            code = self._extract_python_code(ai_response)
            
            return {
                "success": True,
                "ai_response": ai_response,
                "python_code": code,
                "provider": "Anthropic",
                "model": model
            }
            
        except Exception as e:
            self.logger.error(f"Anthropic API エラー: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _analyze_locally(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """ローカル解析（ルールベース）"""
        try:
            code = self._generate_code_from_query(query, data)
            return {
                "success": True,
                "ai_response": f"ローカル解析でクエリ「{query}」を処理しました。",
                "python_code": code,
                "provider": "Local",
                "model": "Rule-based"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_code_from_query(self, query: str, data: pd.DataFrame) -> str:
        """クエリからPythonコード生成（ルールベース）"""
        query_lower = query.lower()
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if "相関" in query or "correlation" in query_lower:
            return f"""
# 相関分析
import matplotlib.pyplot as plt
import seaborn as sns

try:
    correlation_matrix = data[{numeric_cols}].corr()
    print("相関行列:")
    print(correlation_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
except Exception as e:
    print(f"相関分析エラー: {{e}}")
"""
        
        elif "回帰" in query or "regression" in query_lower:
            if len(numeric_cols) >= 2:
                return f"""
# 線形回帰分析
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

try:
    X = data[['{numeric_cols[0]}']]
    y = data['{numeric_cols[1]}']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    print(f"R² Score: {{r2:.4f}}")
    print(f"係数: {{model.coef_[0]:.4f}}")
    print(f"切片: {{model.intercept_:.4f}}")
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, alpha=0.5, label='実際の値')
    plt.scatter(X_test, y_pred, alpha=0.5, label='予測値')
    plt.legend()
    plt.show()
except Exception as e:
    print(f"回帰分析エラー: {{e}}")
"""
            else:
                return "# エラー: 回帰分析には少なくとも2つの数値列が必要です"
        
        elif "記述統計" in query or "descriptive" in query_lower:
            return """
# 記述統計
try:
    print("データの形状:", data.shape)
    print("\\n基本統計量:")
    print(data.describe())

    print("\\n欠損値:")
    print(data.isnull().sum())

    print("\\nデータ型:")
    print(data.dtypes)
except Exception as e:
    print(f"記述統計エラー: {e}")
"""
        
        else:
            return """
# 基本的なデータ探索
try:
    print("データ概要:")
    print(f"行数: {len(data)}")
    print(f"列数: {len(data.columns)}")
    print("\\n最初の5行:")
    print(data.head())

    print("\\n基本統計量:")
    print(data.describe())
except Exception as e:
    print(f"データ探索エラー: {e}")
"""
    
    async def analyze_image_data(self, image_path: str, context: str = "") -> Dict[str, Any]:
        """画像からデータ抽出・分析"""
        try:
            # OCRでテキスト抽出（利用可能な場合）
            ocr_text = ""
            if IMAGE_PROCESSING_AVAILABLE:
                try:
                    if PYTESSERACT_AVAILABLE:
                        img = Image.open(image_path)
                        ocr_text = pytesseract.image_to_string(img, lang='eng+jpn')
                    elif EASYOCR_AVAILABLE:
                        reader = easyocr.Reader(['en', 'ja'])
                        result = reader.readtext(image_path)
                        ocr_text = '\n'.join([item[1] for item in result])
                except Exception as e:
                    self.logger.warning(f"OCRエラー: {e}")
            
            # AI APIで画像分析
            if OPENAI_AVAILABLE and ai_config.is_api_configured("openai"):
                return await self._analyze_image_with_openai(image_path, context, ocr_text)
            elif GOOGLE_AI_AVAILABLE and ai_config.is_api_configured("google"):
                return await self._analyze_image_with_google(image_path, context, ocr_text)
            elif ANTHROPIC_AVAILABLE and ai_config.is_api_configured("anthropic"):
                return await self._analyze_image_with_anthropic(image_path, context, ocr_text)
            else:
                return {
                    "success": True,
                    "ai_response": "画像分析にはAI APIの設定が必要です。",
                    "python_code": "",
                    "ocr_text": ocr_text,
                    "provider": "Local",
                    "model": "OCR-only"
                }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _analyze_image_with_openai(self, image_path: str, context: str, ocr_text: str) -> Dict[str, Any]:
        """OpenAI Vision APIで画像分析"""
        try:
            client = openai.OpenAI(api_key=ai_config.openai_api_key)
            
            # 画像をBase64エンコード
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            prompt = "この画像に含まれるデータ（表、グラフ、チャートなど）を抽出し、Pandasで使用できるPythonコードを生成してください。"
            if context:
                prompt += f"\n追加コンテキスト: {context}"
            
            response = client.chat.completions.create(
                model="gpt-4o",  # Vision対応モデル
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096
            )
            
            ai_response = response.choices[0].message.content or ""
            code = self._extract_python_code(ai_response)
            
            return {
                "success": True,
                "ai_response": ai_response,
                "python_code": code,
                "ocr_text": ocr_text,
                "provider": "OpenAI",
                "model": "gpt-4o"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _analyze_image_with_google(self, image_path: str, context: str, ocr_text: str) -> Dict[str, Any]:
        """Google Gemini Vision APIで画像分析"""
        try:
            if not GOOGLE_AI_AVAILABLE:
                return {"success": False, "error": "Google AI APIが利用できません"}
                
            genai.configure(api_key=ai_config.google_api_key)
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            img = Image.open(image_path)
            prompt = "この画像に含まれるデータを抽出し、Pandasで使用できるPythonコードを生成してください。"
            if context:
                prompt += f"\n追加コンテキスト: {context}"
            
            response = model.generate_content([prompt, img])
            ai_response = response.text if response.text else ""
            code = self._extract_python_code(ai_response)
            
            return {
                "success": True,
                "ai_response": ai_response,
                "python_code": code,
                "ocr_text": ocr_text,
                "provider": "Google",
                "model": "gemini-1.5-pro"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _analyze_image_with_anthropic(self, image_path: str, context: str, ocr_text: str) -> Dict[str, Any]:
        """Anthropic Claude Vision APIで画像分析"""
        try:
            if not ANTHROPIC_AVAILABLE:
                return {"success": False, "error": "Anthropic APIが利用できません"}
                
            client = anthropic.Anthropic(api_key=ai_config.anthropic_api_key)
            
            # 画像をBase64エンコード
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            prompt = "この画像に含まれるデータを抽出し、Pandasで使用できるPythonコードを生成してください。"
            if context:
                prompt += f"\n追加コンテキスト: {context}"
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            )
            
            ai_response = ""
            if response.content and len(response.content) > 0:
                content_block = response.content[0]
                if hasattr(content_block, 'text'):
                    ai_response = getattr(content_block, 'text')
                else:
                    ai_response = str(content_block)
            
            code = self._extract_python_code(ai_response)
            
            return {
                "success": True,
                "ai_response": ai_response,
                "python_code": code,
                "ocr_text": ocr_text,
                "provider": "Anthropic",
                "model": "claude-3-5-sonnet-20241022"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _extract_python_code(self, text: str) -> str:
        """テキストからPythonコードを抽出"""
        if not text:
            return ""
            
        # ```python ... ``` ブロックを抽出
        pattern = r'```python\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return '\n'.join(matches)
        
        # ``` ... ``` ブロックを抽出（python指定なし）
        pattern = r'```\s*\n(.*?)\n```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return '\n'.join(matches)
        
        return ""
    
    def execute_generated_code(self, code: str, data: pd.DataFrame) -> Dict[str, Any]:
        """生成されたコードを安全に実行"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score
            
            # 安全な実行環境
            safe_globals = {
                "pd": pd,
                "np": np,
                "data": data,
                "df": data,
                "plt": plt,
                "sns": sns,
                "LinearRegression": LinearRegression,
                "train_test_split": train_test_split,
                "r2_score": r2_score,
                "__builtins__": {
                    "len": len, "str": str, "int": int, "float": float, "print": print,
                    "range": range, "enumerate": enumerate, "zip": zip, "list": list,
                    "dict": dict, "tuple": tuple, "set": set
                }
            }
            
            local_vars = {}
            exec(code, safe_globals, local_vars)
            
            return {
                "success": True,
                "result": local_vars,
                "output": "コード実行完了"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """分析履歴を取得"""
        return self.analysis_history
    
    def clear_history(self) -> None:
        """分析履歴をクリア"""
        self.analysis_history.clear()

# グローバルインスタンス
ai_analyzer = AIStatisticalAnalyzer()

# ユーティリティ関数
def check_ai_availability() -> Dict[str, Any]:
    """AI API の利用可能性をチェック"""
    return {
        "openai": OPENAI_AVAILABLE and ai_config.is_api_configured("openai"),
        "google": GOOGLE_AI_AVAILABLE and ai_config.is_api_configured("google"),
        "anthropic": ANTHROPIC_AVAILABLE and ai_config.is_api_configured("anthropic"),
        "image_processing": IMAGE_PROCESSING_AVAILABLE,
        "pytesseract": PYTESSERACT_AVAILABLE,
        "easyocr": EASYOCR_AVAILABLE
    }

async def quick_analyze(query: str, data: pd.DataFrame) -> Dict[str, Any]:
    """クイック分析関数"""
    return await ai_analyzer.analyze_natural_language_query(query, data)
