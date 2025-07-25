#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Contextual Information Retrieval System
コンテキスト対応情報検索システム

このモジュールは以下の機能を提供します:
- データ特性に基づく関連性スコアリング
- クエリ拡張機能
- キャッシュシステム
- コンテキスト対応知識検索
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import pickle
from pathlib import Path
import threading
import time

# RAG関連ライブラリ
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# 統計ライブラリ
try:
    from scipy.spatial.distance import cosine
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# 設定とライセンス
try:
    from config import check_feature_permission
    if not check_feature_permission('advanced_ai'):
        raise ImportError("Advanced AI features require Professional edition or higher")
except ImportError:
    def check_feature_permission(feature):
        return True

# 既存モジュール
try:
    from ai_integration import AnalysisContext, DataCharacteristics
    from statistical_method_advisor import StatisticalMethod, DataType
except ImportError:
    # フォールバック定義
    class AnalysisContext:
        def __init__(self):
            self.user_expertise_level = "intermediate"
            self.analysis_history = []
    
    class DataCharacteristics:
        def __init__(self):
            self.n_rows = 0
            self.n_columns = 0
            self.column_types = {}
    
    class StatisticalMethod:
        pass
    
    class DataType:
        pass

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalContext:
    """検索コンテキスト"""
    user_expertise_level: str
    data_characteristics: DataCharacteristics
    analysis_history: List[str]
    current_query: str
    preferred_methods: List[str] = field(default_factory=list)
    recent_topics: List[str] = field(default_factory=list)
    session_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalResult:
    """検索結果"""
    content: str
    relevance_score: float
    source: str
    method_related: List[str] = field(default_factory=list)
    assumptions_covered: List[str] = field(default_factory=list)
    educational_content: Optional[str] = None
    code_examples: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class CacheEntry:
    """キャッシュエントリ"""
    query_hash: str
    results: List[RetrievalResult]
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)

class ContextualRetriever:
    """コンテキスト対応情報検索システム"""
    
    def __init__(self, cache_dir: str = "cache", max_cache_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_lock = threading.Lock()
        
        # RAGシステム初期化
        self._initialize_rag_system()
        
        # 統計知識ベース初期化
        self._initialize_statistical_knowledge()
        
        # キャッシュ読み込み
        self._load_cache()
        
        logger.info("ContextualRetriever 初期化完了")
    
    def _initialize_rag_system(self):
        """RAGシステムの初期化"""
        if RAG_AVAILABLE:
            try:
                # 軽量な多言語モデルを使用
                self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self.index = None
                self.documents = []
                self.document_embeddings = []
                logger.info("RAGシステム初期化完了")
            except Exception as e:
                logger.warning(f"RAGシステム初期化エラー: {e}")
                self.embedding_model = None
        else:
            logger.warning("RAGライブラリが利用できません")
            self.embedding_model = None
    
    def _initialize_statistical_knowledge(self):
        """統計知識ベースの初期化"""
        self.statistical_knowledge = {
            "methods": {
                "t_test": {
                    "description": "2群の平均値の差を検定する手法",
                    "assumptions": ["正規性", "等分散性", "独立性"],
                    "use_cases": ["A/Bテスト", "治療効果の比較"],
                    "alternatives": ["Mann-Whitney U検定", "Wilcoxon検定"],
                    "code_example": "scipy.stats.ttest_ind(group1, group2)",
                    "educational_content": "t検定は、2つの群の平均値に差があるかを検定する手法です..."
                },
                "anova": {
                    "description": "3群以上の平均値の差を検定する手法",
                    "assumptions": ["正規性", "等分散性", "独立性"],
                    "use_cases": ["複数治療の比較", "実験条件の効果"],
                    "alternatives": ["Kruskal-Wallis検定", "Friedman検定"],
                    "code_example": "scipy.stats.f_oneway(*groups)",
                    "educational_content": "ANOVAは、3群以上の平均値の差を同時に検定する手法です..."
                },
                "regression": {
                    "description": "変数間の関係性を分析する手法",
                    "assumptions": ["線形性", "独立性", "等分散性", "正規性"],
                    "use_cases": ["予測モデル", "因果関係の分析"],
                    "alternatives": ["ロバスト回帰", "非線形回帰"],
                    "code_example": "sklearn.linear_model.LinearRegression()",
                    "educational_content": "回帰分析は、説明変数と目的変数の関係性を分析する手法です..."
                }
            },
            "assumptions": {
                "normality": {
                    "description": "データが正規分布に従うこと",
                    "tests": ["Shapiro-Wilk検定", "Kolmogorov-Smirnov検定"],
                    "remediation": ["データ変換", "ノンパラメトリック手法"],
                    "educational_content": "正規性の仮定は、多くの統計手法で重要な前提条件です..."
                },
                "homoscedasticity": {
                    "description": "分散が等しいこと",
                    "tests": ["Levene検定", "Bartlett検定"],
                    "remediation": ["Welch検定", "ロバスト手法"],
                    "educational_content": "等分散性の仮定は、群間比較で重要な条件です..."
                }
            }
        }
    
    def search_relevant_methods(self, query: str, context: RetrievalContext) -> List[RetrievalResult]:
        """関連する統計手法を検索"""
        cache_key = self._generate_cache_key(query, context)
        
        # キャッシュチェック
        cached_results = self._get_from_cache(cache_key)
        if cached_results:
            return cached_results
        
        # 新しい検索実行
        results = self._perform_contextual_search(query, context)
        
        # キャッシュに保存
        self._save_to_cache(cache_key, results)
        
        return results
    
    def _perform_contextual_search(self, query: str, context: RetrievalContext) -> List[RetrievalResult]:
        """コンテキスト対応検索の実行"""
        results = []
        
        # 1. クエリ拡張
        expanded_queries = self._expand_query(query, context)
        
        # 2. 統計知識ベース検索
        knowledge_results = self._search_statistical_knowledge(expanded_queries, context)
        results.extend(knowledge_results)
        
        # 3. RAGシステム検索（利用可能な場合）
        if self.embedding_model:
            rag_results = self._search_rag_system(expanded_queries, context)
            results.extend(rag_results)
        
        # 4. 関連性スコアリング
        scored_results = self._score_relevance(results, context)
        
        # 5. 結果の並び替えとフィルタリング
        final_results = self._rank_and_filter_results(scored_results, context)
        
        return final_results
    
    def _expand_query(self, query: str, context: RetrievalContext) -> List[str]:
        """クエリ拡張"""
        expanded_queries = [query]
        
        # データ特性に基づく拡張
        if context.data_characteristics:
            data_expansions = self._expand_based_on_data_characteristics(query, context.data_characteristics)
            expanded_queries.extend(data_expansions)
        
        # ユーザー履歴に基づく拡張
        if context.analysis_history:
            history_expansions = self._expand_based_on_history(query, context.analysis_history)
            expanded_queries.extend(history_expansions)
        
        # 専門レベルに基づく拡張
        expertise_expansions = self._expand_based_on_expertise(query, context.user_expertise_level)
        expanded_queries.extend(expertise_expansions)
        
        return list(set(expanded_queries))  # 重複除去
    
    def _expand_based_on_data_characteristics(self, query: str, data_chars: DataCharacteristics) -> List[str]:
        """データ特性に基づくクエリ拡張"""
        expansions = []
        
        # サンプルサイズに基づく拡張
        if data_chars.n_rows < 30:
            expansions.extend(["小サンプル", "ノンパラメトリック", "ブートストラップ"])
        elif data_chars.n_rows > 1000:
            expansions.extend(["大サンプル", "統計的検出力", "効果量"])
        
        # 変数タイプに基づく拡張
        categorical_vars = [k for k, v in data_chars.column_types.items() if v == 'categorical']
        continuous_vars = [k for k, v in data_chars.column_types.items() if v == 'continuous']
        
        if categorical_vars:
            expansions.extend(["カテゴリカル", "カイ二乗検定", "分割表"])
        if continuous_vars:
            expansions.extend(["連続変数", "相関", "回帰"])
        
        return expansions
    
    def _expand_based_on_history(self, query: str, history: List[str]) -> List[str]:
        """履歴に基づくクエリ拡張"""
        expansions = []
        
        # 最近使用された手法を抽出
        recent_methods = []
        for item in history[-5:]:  # 最近5件
            if "t検定" in item:
                recent_methods.append("t検定")
            elif "ANOVA" in item or "分散分析" in item:
                recent_methods.append("ANOVA")
            elif "回帰" in item:
                recent_methods.append("回帰分析")
        
        # 関連する手法を追加
        for method in recent_methods:
            if method == "t検定":
                expansions.extend(["Mann-Whitney", "Wilcoxon", "効果量"])
            elif method == "ANOVA":
                expansions.extend(["Kruskal-Wallis", "多重比較", "効果量"])
            elif method == "回帰分析":
                expansions.extend(["多重回帰", "変数選択", "診断"])
        
        return expansions
    
    def _expand_based_on_expertise(self, query: str, expertise_level: str) -> List[str]:
        """専門レベルに基づくクエリ拡張"""
        expansions = []
        
        if expertise_level == "novice":
            expansions.extend(["基本的な", "初心者向け", "分かりやすい"])
        elif expertise_level == "intermediate":
            expansions.extend(["中級", "実用的", "応用"])
        elif expertise_level == "expert":
            expansions.extend(["高度な", "専門的", "最新の"])
        
        return expansions
    
    def _search_statistical_knowledge(self, queries: List[str], context: RetrievalContext) -> List[RetrievalResult]:
        """統計知識ベース検索"""
        results = []
        
        for query in queries:
            for method_name, method_info in self.statistical_knowledge["methods"].items():
                # クエリと手法の関連性を計算
                relevance_score = self._calculate_method_relevance(query, method_info, context)
                
                if relevance_score > 0.3:  # 閾値
                    result = RetrievalResult(
                        content=method_info["description"],
                        relevance_score=relevance_score,
                        source=f"statistical_knowledge_{method_name}",
                        method_related=[method_name],
                        assumptions_covered=method_info.get("assumptions", []),
                        educational_content=method_info.get("educational_content"),
                        code_examples=[method_info.get("code_example", "")]
                    )
                    results.append(result)
        
        return results
    
    def _calculate_method_relevance(self, query: str, method_info: Dict[str, Any], context: RetrievalContext) -> float:
        """手法の関連性スコア計算"""
        score = 0.0
        
        # クエリと手法名の一致度
        query_lower = query.lower()
        method_name = method_info.get("description", "").lower()
        
        # キーワードマッチング
        keywords = ["t検定", "anova", "回帰", "相関", "カイ二乗", "分散分析"]
        for keyword in keywords:
            if keyword in query_lower and keyword in method_name:
                score += 0.4
        
        # データ特性との適合度
        if context.data_characteristics:
            data_score = self._calculate_data_method_fit(method_info, context.data_characteristics)
            score += data_score * 0.3
        
        # ユーザー履歴との関連性
        if context.analysis_history:
            history_score = self._calculate_history_relevance(method_info, context.analysis_history)
            score += history_score * 0.2
        
        # 専門レベルとの適合度
        expertise_score = self._calculate_expertise_fit(method_info, context.user_expertise_level)
        score += expertise_score * 0.1
        
        return min(score, 1.0)
    
    def _calculate_data_method_fit(self, method_info: Dict[str, Any], data_chars: DataCharacteristics) -> float:
        """データと手法の適合度計算"""
        score = 0.0
        
        # サンプルサイズの適合度
        if data_chars.n_rows < 30:
            # 小サンプル向け手法
            if "ノンパラメトリック" in method_info.get("alternatives", []):
                score += 0.3
        elif data_chars.n_rows > 100:
            # 大サンプル向け手法
            if "効果量" in method_info.get("description", ""):
                score += 0.3
        
        # 変数タイプの適合度
        categorical_count = sum(1 for v in data_chars.column_types.values() if v == 'categorical')
        continuous_count = sum(1 for v in data_chars.column_types.values() if v == 'continuous')
        
        if categorical_count > 0 and "カテゴリカル" in method_info.get("description", ""):
            score += 0.3
        if continuous_count > 0 and "連続" in method_info.get("description", ""):
            score += 0.3
        
        return score
    
    def _calculate_history_relevance(self, method_info: Dict[str, Any], history: List[str]) -> float:
        """履歴との関連性計算"""
        score = 0.0
        
        method_name = method_info.get("description", "").lower()
        
        for item in history:
            item_lower = item.lower()
            if any(keyword in item_lower for keyword in ["t検定", "anova", "回帰"]):
                if any(keyword in method_name for keyword in ["t検定", "anova", "回帰"]):
                    score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_expertise_fit(self, method_info: Dict[str, Any], expertise_level: str) -> float:
        """専門レベルとの適合度計算"""
        if expertise_level == "novice":
            return 0.8 if "基本的" in method_info.get("description", "") else 0.3
        elif expertise_level == "expert":
            return 0.8 if "高度" in method_info.get("description", "") else 0.3
        else:
            return 0.6  # intermediate
    
    def _search_rag_system(self, queries: List[str], context: RetrievalContext) -> List[RetrievalResult]:
        """RAGシステム検索"""
        if not self.embedding_model or not self.documents:
            return []
        
        results = []
        
        for query in queries:
            # クエリの埋め込み
            query_embedding = self.embedding_model.encode([query])[0]
            
            # FAISSインデックス検索
            if self.index is not None:
                D, I = self.index.search(query_embedding.reshape(1, -1), k=5)
                
                for i, (distance, idx) in enumerate(zip(D[0], I[0])):
                    if idx < len(self.documents):
                        relevance_score = 1.0 - distance
                        
                        result = RetrievalResult(
                            content=self.documents[idx],
                            relevance_score=relevance_score,
                            source=f"rag_system_{idx}",
                            method_related=[],
                            assumptions_covered=[],
                            educational_content=None,
                            code_examples=[]
                        )
                        results.append(result)
        
        return results
    
    def _score_relevance(self, results: List[RetrievalResult], context: RetrievalContext) -> List[RetrievalResult]:
        """関連性スコアリング"""
        for result in results:
            # 基本スコアに追加の重み付け
            additional_score = 0.0
            
            # ユーザーの好みする手法との一致
            if any(method in result.method_related for method in context.preferred_methods):
                additional_score += 0.2
            
            # 最近のトピックとの関連性
            if any(topic in result.content for topic in context.recent_topics):
                additional_score += 0.1
            
            # セッションメタデータとの関連性
            if context.session_metadata.get("analysis_type") in result.content:
                additional_score += 0.1
            
            result.relevance_score = min(result.relevance_score + additional_score, 1.0)
        
        return results
    
    def _rank_and_filter_results(self, results: List[RetrievalResult], context: RetrievalContext) -> List[RetrievalResult]:
        """結果の並び替えとフィルタリング"""
        # 関連性スコアで並び替え
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 重複除去
        seen_contents = set()
        filtered_results = []
        
        for result in results:
            content_hash = hashlib.md5(result.content.encode()).hexdigest()
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                filtered_results.append(result)
        
        # 上位結果のみ返す
        return filtered_results[:10]
    
    def _generate_cache_key(self, query: str, context: RetrievalContext) -> str:
        """キャッシュキーの生成"""
        # クエリとコンテキストのハッシュ
        context_str = f"{query}_{context.user_expertise_level}_{context.data_characteristics.n_rows}_{context.data_characteristics.n_columns}"
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[RetrievalResult]]:
        """キャッシュからの取得"""
        with self.cache_lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                # キャッシュの有効期限チェック（24時間）
                if datetime.now() - entry.timestamp < timedelta(hours=24):
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                    return entry.results
                else:
                    # 期限切れのエントリを削除
                    del self.cache[cache_key]
        
        return None
    
    def _save_to_cache(self, cache_key: str, results: List[RetrievalResult]):
        """キャッシュへの保存"""
        with self.cache_lock:
            # キャッシュサイズ制限チェック
            if len(self.cache) >= self.max_cache_size:
                # 最も古いエントリを削除
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
                del self.cache[oldest_key]
            
            entry = CacheEntry(
                query_hash=cache_key,
                results=results,
                timestamp=datetime.now()
            )
            self.cache[cache_key] = entry
    
    def _load_cache(self):
        """キャッシュの読み込み"""
        cache_file = self.cache_dir / "contextual_retriever_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"キャッシュを読み込みました: {len(self.cache)} エントリ")
            except Exception as e:
                logger.warning(f"キャッシュ読み込みエラー: {e}")
    
    def save_cache(self):
        """キャッシュの保存"""
        cache_file = self.cache_dir / "contextual_retriever_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info(f"キャッシュを保存しました: {len(self.cache)} エントリ")
        except Exception as e:
            logger.error(f"キャッシュ保存エラー: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """キャッシュ統計の取得"""
        with self.cache_lock:
            total_entries = len(self.cache)
            total_accesses = sum(entry.access_count for entry in self.cache.values())
            avg_accesses = total_accesses / total_entries if total_entries > 0 else 0
            
            return {
                "total_entries": total_entries,
                "total_accesses": total_accesses,
                "average_accesses": avg_accesses,
                "cache_size_mb": sum(len(pickle.dumps(entry)) for entry in self.cache.values()) / (1024 * 1024)
            }
    
    def clear_cache(self):
        """キャッシュのクリア"""
        with self.cache_lock:
            self.cache.clear()
        logger.info("キャッシュをクリアしました")

def main():
    """テスト用メイン関数"""
    # テストデータの作成
    data_chars = DataCharacteristics()
    data_chars.n_rows = 100
    data_chars.n_columns = 5
    data_chars.column_types = {
        "group": "categorical",
        "score": "continuous",
        "age": "continuous"
    }
    
    context = RetrievalContext(
        user_expertise_level="intermediate",
        data_characteristics=data_chars,
        analysis_history=["t検定", "ANOVA", "回帰分析"],
        current_query="2群の平均値の差を検定したい",
        preferred_methods=["t検定", "Mann-Whitney"],
        recent_topics=["検定", "比較"]
    )
    
    # ContextualRetrieverのテスト
    retriever = ContextualRetriever()
    results = retriever.search_relevant_methods("2群の平均値の差を検定したい", context)
    
    print("検索結果:")
    for i, result in enumerate(results[:3]):
        print(f"{i+1}. {result.content} (スコア: {result.relevance_score:.3f})")
        print(f"   手法: {result.method_related}")
        print(f"   仮定: {result.assumptions_covered}")
        print()

if __name__ == "__main__":
    main() 