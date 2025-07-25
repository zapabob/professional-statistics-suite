#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local LLM Statistical Assistant
ローカルLLM統計補助システム

LMStudioのPythonライブラリを使用してGGUFファイルを推論し、
統計分析のサポートを行うシステム
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import asyncio
from dataclasses import dataclass
from datetime import datetime

# データ処理
import pandas as pd
import numpy as np

# LMStudioライブラリ
try:
    from lmstudio import LMStudioClient
    LMSTUDIO_AVAILABLE = True
except ImportError:
    try:
        from lmstudio.client import LMStudioClient
        LMSTUDIO_AVAILABLE = True
    except ImportError:
        LMSTUDIO_AVAILABLE = False
        print("⚠️ LMStudioライブラリが利用できません")

# 既存のAI統合システム
from ai_integration import AIOrchestrator, AnalysisContext, IntentType

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GGUFModelConfig:
    """GGUFモデル設定"""
    model_path: str
    model_name: str
    context_size: int = 4096
    temperature: float = 0.3
    max_tokens: int = 512
    top_p: float = 0.9
    repeat_penalty: float = 1.1

@dataclass
class StatisticalQuery:
    """統計クエリ"""
    query: str
    data_info: Optional[Dict[str, Any]] = None
    context: Optional[str] = None
    user_expertise: str = "intermediate"
    language: str = "ja"

@dataclass
class StatisticalResponse:
    """統計応答"""
    answer: str
    confidence: float
    suggested_methods: List[str]
    processing_time: float
    tokens_used: int
    educational_content: Optional[str] = None
    code_example: Optional[str] = None

class LocalLLMStatisticalAssistant:
    """ローカルLLM統計補助システム"""
    
    def __init__(self, model_config: GGUFModelConfig):
        self.model_config = model_config
        self.client = None
        self.is_initialized = False
        
        # 統計分析用のプロンプトテンプレート
        self.statistical_prompts = {
            "descriptive": self._get_descriptive_prompt(),
            "inferential": self._get_inferential_prompt(),
            "predictive": self._get_predictive_prompt(),
            "educational": self._get_educational_prompt()
        }
        
        # 利用可能な統計手法データベース
        self.statistical_methods = self._load_statistical_methods()
    
    def initialize(self) -> bool:
        """LMStudioクライアントを初期化"""
        try:
            if not LMSTUDIO_AVAILABLE:
                logger.error("LMStudioライブラリが利用できません")
                return False
            
            # モデルファイルの存在確認
            if not Path(self.model_config.model_path).exists():
                logger.error(f"GGUFモデルファイルが見つかりません: {self.model_config.model_path}")
                return False
            
            # LMStudioクライアントを初期化（実際のAPIエンドポイントに合わせて調整）
            try:
                self.client = LMStudioClient()
                logger.info(f"✅ LMStudioクライアント初期化成功")
            except Exception as e:
                logger.warning(f"⚠️ LMStudioクライアント初期化エラー: {e}")
                # クライアントが初期化できなくても、モデルファイルは存在するので続行
                pass
            
            logger.info(f"📁 モデルパス: {self.model_config.model_path}")
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ LMStudioクライアント初期化失敗: {e}")
            return False
    
    async def analyze_statistical_query(self, query: StatisticalQuery) -> StatisticalResponse:
        """統計クエリを分析"""
        if not self.is_initialized:
            return StatisticalResponse(
                answer="システムが初期化されていません",
                confidence=0.0,
                suggested_methods=[],
                processing_time=0.0,
                tokens_used=0
            )
        
        try:
            start_time = time.time()
            
            # クエリの意図を分類
            intent = self._classify_query_intent(query.query)
            
            # 適切なプロンプトを生成
            prompt = self._build_statistical_prompt(query, intent)
            
            # LMStudioで推論実行
            response = await self._generate_response(prompt)
            
            processing_time = time.time() - start_time
            
            # 応答を解析
            parsed_response = self._parse_statistical_response(response)
            
            return StatisticalResponse(
                answer=parsed_response.get('answer', response),
                confidence=parsed_response.get('confidence', 0.7),
                suggested_methods=parsed_response.get('suggested_methods', []),
                educational_content=parsed_response.get('educational_content'),
                code_example=parsed_response.get('code_example'),
                processing_time=processing_time,
                tokens_used=parsed_response.get('tokens_used', 0)
            )
            
        except Exception as e:
            logger.error(f"統計クエリ分析エラー: {e}")
            return StatisticalResponse(
                answer=f"エラーが発生しました: {str(e)}",
                confidence=0.0,
                suggested_methods=[],
                processing_time=0.0,
                tokens_used=0
            )
    
    async def _generate_response(self, prompt: str) -> str:
        """LMStudioで応答生成"""
        try:
            if self.client:
                # LMStudioクライアントで推論実行
                response = await self.client.chat.completions.create(
                    model=self.model_config.model_name,
                    messages=[
                        {"role": "system", "content": "あなたは統計分析の専門家です。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.model_config.temperature,
                    max_tokens=self.model_config.max_tokens,
                    top_p=self.model_config.top_p,
                    repeat_penalty=self.model_config.repeat_penalty
                )
                
                return response.choices[0].message.content
            else:
                # クライアントが利用できない場合はモック応答
                logger.warning("LMStudioクライアントが利用できないため、モック応答を返します")
                return f"モック応答: {prompt[:100]}... (LMStudioクライアントが利用できません)"
            
        except Exception as e:
            logger.error(f"LMStudio推論エラー: {e}")
            return f"推論エラー: {str(e)}"
    
    def _classify_query_intent(self, query: str) -> str:
        """クエリの意図を分類"""
        query_lower = query.lower()
        
        # 記述統計
        if any(word in query_lower for word in ['平均', '中央値', '分散', '標準偏差', '分布', '要約']):
            return "descriptive"
        
        # 推論統計
        elif any(word in query_lower for word in ['検定', 't検定', 'カイ二乗', '相関', '回帰', '有意']):
            return "inferential"
        
        # 予測分析
        elif any(word in query_lower for word in ['予測', '予測', 'モデル', '機械学習', '分類']):
            return "predictive"
        
        # 教育的内容
        else:
            return "educational"
    
    def _build_statistical_prompt(self, query: StatisticalQuery, intent: str) -> str:
        """統計分析用プロンプトを構築"""
        base_prompt = self.statistical_prompts.get(intent, self.statistical_prompts["educational"])
        
        # データ情報を追加
        data_context = ""
        if query.data_info:
            data_context = f"""
データ情報:
- 行数: {query.data_info.get('rows', 'N/A')}
- 列数: {query.data_info.get('columns', 'N/A')}
- 列名: {query.data_info.get('column_names', [])}
- データ型: {query.data_info.get('dtypes', {})}
"""
        
        # コンテキストを追加
        context_info = ""
        if query.context:
            context_info = f"\n分析コンテキスト: {query.context}"
        
        # ユーザーレベルに応じた説明レベルを設定
        expertise_level = {
            "beginner": "初心者向けに詳しく説明してください",
            "intermediate": "中級者向けに説明してください",
            "advanced": "上級者向けに専門的に説明してください"
        }.get(query.user_expertise, "中級者向けに説明してください")
        
        prompt = f"""
{base_prompt}

{data_context}
{context_info}

ユーザーの質問: {query.query}
{expertise_level}

回答は以下の形式で提供してください:
1. 直接的な回答
2. 推奨される統計手法
3. 教育的内容（必要に応じて）
4. コード例（必要に応じて）
"""
        
        return prompt
    
    def _get_descriptive_prompt(self) -> str:
        """記述統計用プロンプト"""
        return """
あなたは統計分析の専門家です。記述統計に関する質問に答えてください。

記述統計では以下の点を考慮してください:
- データの要約統計量（平均、中央値、分散、標準偏差など）
- データの分布の可視化
- 外れ値の検出
- データの品質評価
"""
    
    def _get_inferential_prompt(self) -> str:
        """推論統計用プロンプト"""
        return """
あなたは統計分析の専門家です。推論統計に関する質問に答えてください。

推論統計では以下の点を考慮してください:
- 適切な検定手法の選択
- 仮説の設定
- 有意水準の設定
- 結果の解釈
- 仮定の確認
"""
    
    def _get_predictive_prompt(self) -> str:
        """予測分析用プロンプト"""
        return """
あなたは統計分析の専門家です。予測分析に関する質問に答えてください。

予測分析では以下の点を考慮してください:
- 適切なモデルの選択
- 特徴量エンジニアリング
- モデルの評価指標
- 過学習の回避
- 結果の解釈
"""
    
    def _get_educational_prompt(self) -> str:
        """教育的内容用プロンプト"""
        return """
あなたは統計分析の専門家です。統計学の教育的内容を提供してください。

教育的内容では以下の点を考慮してください:
- 概念の分かりやすい説明
- 具体例の提供
- 実践的な応用例
- よくある誤解の説明
"""
    
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
            logger.error(f"応答解析エラー: {e}")
            return {
                'answer': response,
                'confidence': 0.5,
                'suggested_methods': [],
                'tokens_used': len(response.split())
            }
    
    def _load_statistical_methods(self) -> Dict[str, Dict[str, Any]]:
        """統計手法データベースを読み込み"""
        return {
            "t_test": {
                "name": "t検定",
                "description": "2群の平均値の差を検定",
                "assumptions": ["正規分布", "等分散性", "独立性"],
                "use_cases": ["2群の比較", "前後比較"]
            },
            "chi_square": {
                "name": "カイ二乗検定",
                "description": "カテゴリカルデータの独立性を検定",
                "assumptions": ["独立性", "期待度数"],
                "use_cases": ["分割表の分析", "適合度検定"]
            },
            "correlation": {
                "name": "相関分析",
                "description": "2変数間の関係性を分析",
                "assumptions": ["線形関係", "正規分布"],
                "use_cases": ["関係性の探索", "予測モデル"]
            },
            "regression": {
                "name": "回帰分析",
                "description": "従属変数を独立変数で予測",
                "assumptions": ["線形性", "独立性", "等分散性", "正規性"],
                "use_cases": ["予測モデル", "因果関係の探索"]
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        return {
            "model_path": self.model_config.model_path,
            "model_name": self.model_config.model_name,
            "context_size": self.model_config.context_size,
            "temperature": self.model_config.temperature,
            "max_tokens": self.model_config.max_tokens,
            "is_initialized": self.is_initialized
        }

class StatisticalAssistantDemo:
    """統計補助システムのデモ"""
    
    def __init__(self):
        self.assistant = None
    
    def setup_assistant(self, model_path: str) -> bool:
        """統計補助システムをセットアップ"""
        try:
            # GGUFモデル設定
            model_config = GGUFModelConfig(
                model_path=model_path,
                model_name=Path(model_path).stem,
                context_size=4096,
                temperature=0.3,
                max_tokens=512
            )
            
            # 統計補助システムを初期化
            self.assistant = LocalLLMStatisticalAssistant(model_config)
            
            if self.assistant.initialize():
                print("✅ 統計補助システム初期化成功")
                return True
            else:
                print("❌ 統計補助システム初期化失敗")
                return False
                
        except Exception as e:
            print(f"❌ セットアップエラー: {e}")
            return False
    
    async def run_demo(self):
        """デモを実行"""
        if not self.assistant:
            print("❌ 統計補助システムが初期化されていません")
            return
        
        print("\n🚀 ローカルLLM統計補助システム デモ")
        print("=" * 50)
        
        # デモクエリ
        demo_queries = [
            StatisticalQuery(
                query="データの平均値と標準偏差を計算する方法を教えてください",
                user_expertise="beginner"
            ),
            StatisticalQuery(
                query="2群の平均値の差を検定するにはどの手法を使いますか？",
                user_expertise="intermediate"
            ),
            StatisticalQuery(
                query="相関分析と回帰分析の違いを説明してください",
                user_expertise="intermediate"
            ),
            StatisticalQuery(
                query="データが正規分布していない場合の対処法を教えてください",
                user_expertise="advanced"
            )
        ]
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n📊 デモクエリ {i}: {query.query}")
            print("-" * 40)
            
            response = await self.assistant.analyze_statistical_query(query)
            
            print(f"✅ 回答生成成功")
            print(f"⏱️  処理時間: {response.processing_time:.2f}秒")
            print(f"🔢 トークン使用量: {response.tokens_used}")
            print(f"📊 信頼度: {response.confidence:.2f}")
            
            if response.suggested_methods:
                print(f"📋 推奨手法: {', '.join(response.suggested_methods)}")
            
            print(f"📝 回答:\n{response.answer}")
            
            if response.educational_content:
                print(f"📚 教育的内容:\n{response.educational_content}")
            
            if response.code_example:
                print(f"💻 コード例:\n{response.code_example}")

async def main():
    """メイン関数"""
    print("🧠 ローカルLLM統計補助システム")
    print("=" * 50)
    
    # 動的にGGUFモデルファイルを検索
    model_paths = [
        "./models/",
        "../models/",
        "../../models/",
        "~/models/",
        "~/Downloads/"
    ]
    
    available_model = None
    for base_path in model_paths:
        search_path = Path(base_path).expanduser()
        if search_path.exists():
            # .ggufファイルを検索
            gguf_files = list(search_path.glob('*.gguf'))
            if gguf_files:
                # サイズでソート（小さいモデルを優先）
                gguf_files.sort(key=lambda x: x.stat().st_size)
                available_model = str(gguf_files[0])
                break
    
    if not available_model:
        print("❌ GGUFモデルファイルが見つかりません")
        print("💡 以下のパスにGGUFファイルを配置してください:")
        for path in model_paths:
            print(f"   - {path}")
        print("\n🔧 AIサポートなしでも統計分析は利用可能です")
        return
    
    print(f"✅ GGUFモデルを発見: {available_model}")
    
    # デモを実行
    demo = StatisticalAssistantDemo()
    if demo.setup_assistant(available_model):
        await demo.run_demo()
    else:
        print("❌ デモの実行に失敗しました")

if __name__ == "__main__":
    asyncio.run(main()) 