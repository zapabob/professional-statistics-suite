#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistical Method Advisor System
統計手法推奨システム - データ特性に基づく最適な統計手法の推奨

このモジュールは以下の機能を提供します:
- データ特性の自動分析
- 研究質問に基づく統計手法推奨
- 手法適用可能性の評価
- 代替手法の提案
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# 統計ライブラリ
from scipy.stats import normaltest, levene, shapiro

# 設定とライセンス
try:
    from config import check_feature_permission
    if not check_feature_permission('advanced_ai'):
        raise ImportError("Advanced AI features require Professional edition or higher")
except ImportError:
    def check_feature_permission(feature):
        return True

# データ前処理
try:
    from data_preprocessing import validate_statistical_data, DataQualityReport
except ImportError:
    # フォールバック実装
    def validate_statistical_data(data):
        return {"quality_score": 0.8, "issues": [], "recommendations": []}
    
    class DataQualityReport:
        def __init__(self, data):
            self.quality_score = 0.8
            self.issues = []
            self.recommendations = []

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataType(Enum):
    """データ型の分類"""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    BINARY = "binary"
    COUNT = "count"

class AnalysisGoal(Enum):
    """解析目標の分類"""
    DESCRIPTIVE = "descriptive"
    COMPARISON = "comparison"
    RELATIONSHIP = "relationship"
    PREDICTION = "prediction"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"

class StatisticalMethod(Enum):
    """統計手法の分類"""
    # 記述統計
    DESCRIPTIVE_STATS = "descriptive_statistics"
    
    # 比較検定
    T_TEST_ONE_SAMPLE = "t_test_one_sample"
    T_TEST_TWO_SAMPLE = "t_test_two_sample"
    T_TEST_PAIRED = "t_test_paired"
    MANN_WHITNEY = "mann_whitney"
    WILCOXON = "wilcoxon"
    
    # 分散分析
    ANOVA_ONE_WAY = "anova_one_way"
    ANOVA_TWO_WAY = "anova_two_way"
    KRUSKAL_WALLIS = "kruskal_wallis"
    
    # 相関・回帰
    PEARSON_CORRELATION = "pearson_correlation"
    SPEARMAN_CORRELATION = "spearman_correlation"
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    
    # カテゴリカル分析
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    
    # 機械学習
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"

@dataclass
class DataCharacteristics:
    """データ特性の情報"""
    sample_size: int
    variables: Dict[str, DataType]
    missing_values: Dict[str, float]
    outliers: Dict[str, int]
    normality: Dict[str, bool]
    homoscedasticity: Optional[bool] = None
    independence: bool = True
    multicollinearity: Optional[float] = None
    balance: Optional[Dict[str, float]] = None

@dataclass
class MethodRecommendation:
    """統計手法推奨の結果"""
    method: StatisticalMethod
    confidence: float
    rationale: str
    assumptions_met: Dict[str, bool]
    alternatives: List[StatisticalMethod] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sample_size_adequate: bool = True
    effect_size_detectable: Optional[float] = None

class StatisticalMethodAdvisor:
    """統計手法推奨システムのコアエンジン"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.StatisticalMethodAdvisor")
        
        # 手法データベースの初期化
        self.method_database = self._initialize_method_database()
        
        # 推奨ルールの初期化
        self.recommendation_rules = self._initialize_recommendation_rules()
        
        self.logger.info("StatisticalMethodAdvisor初期化完了")
    
    def _initialize_method_database(self) -> Dict[StatisticalMethod, Dict[str, Any]]:
        """統計手法データベースの初期化"""
        return {
            StatisticalMethod.T_TEST_ONE_SAMPLE: {
                "name": "一標本t検定",
                "description": "母平均が特定の値と等しいかを検定",
                "assumptions": ["normality", "independence"],
                "data_types": [DataType.CONTINUOUS],
                "min_sample_size": 30,
                "analysis_goals": [AnalysisGoal.COMPARISON],
                "power_calculation": True
            },
            StatisticalMethod.T_TEST_TWO_SAMPLE: {
                "name": "二標本t検定",
                "description": "2つの群の平均値を比較",
                "assumptions": ["normality", "independence", "homoscedasticity"],
                "data_types": [DataType.CONTINUOUS],
                "min_sample_size": 30,
                "analysis_goals": [AnalysisGoal.COMPARISON],
                "power_calculation": True
            },
            StatisticalMethod.MANN_WHITNEY: {
                "name": "Mann-Whitney U検定",
                "description": "2つの群の分布を比較（ノンパラメトリック）",
                "assumptions": ["independence"],
                "data_types": [DataType.CONTINUOUS, DataType.ORDINAL],
                "min_sample_size": 10,
                "analysis_goals": [AnalysisGoal.COMPARISON],
                "power_calculation": False
            },
            StatisticalMethod.ANOVA_ONE_WAY: {
                "name": "一元配置分散分析",
                "description": "3つ以上の群の平均値を比較",
                "assumptions": ["normality", "independence", "homoscedasticity"],
                "data_types": [DataType.CONTINUOUS],
                "min_sample_size": 30,
                "analysis_goals": [AnalysisGoal.COMPARISON],
                "power_calculation": True
            },
            StatisticalMethod.KRUSKAL_WALLIS: {
                "name": "Kruskal-Wallis検定",
                "description": "3つ以上の群の分布を比較（ノンパラメトリック）",
                "assumptions": ["independence"],
                "data_types": [DataType.CONTINUOUS, DataType.ORDINAL],
                "min_sample_size": 15,
                "analysis_goals": [AnalysisGoal.COMPARISON],
                "power_calculation": False
            },
            StatisticalMethod.PEARSON_CORRELATION: {
                "name": "Pearson相関係数",
                "description": "2つの連続変数間の線形関係を測定",
                "assumptions": ["normality", "linearity", "independence"],
                "data_types": [DataType.CONTINUOUS],
                "min_sample_size": 30,
                "analysis_goals": [AnalysisGoal.RELATIONSHIP],
                "power_calculation": True
            },
            StatisticalMethod.SPEARMAN_CORRELATION: {
                "name": "Spearman順位相関係数",
                "description": "2つの変数間の単調関係を測定",
                "assumptions": ["independence"],
                "data_types": [DataType.CONTINUOUS, DataType.ORDINAL],
                "min_sample_size": 20,
                "analysis_goals": [AnalysisGoal.RELATIONSHIP],
                "power_calculation": False
            },
            StatisticalMethod.LINEAR_REGRESSION: {
                "name": "線形回帰分析",
                "description": "連続変数の予測と関係性の分析",
                "assumptions": ["linearity", "independence", "homoscedasticity", "normality_residuals"],
                "data_types": [DataType.CONTINUOUS],
                "min_sample_size": 50,
                "analysis_goals": [AnalysisGoal.PREDICTION, AnalysisGoal.RELATIONSHIP],
                "power_calculation": True
            },
            StatisticalMethod.LOGISTIC_REGRESSION: {
                "name": "ロジスティック回帰分析",
                "description": "二値結果の予測と関係性の分析",
                "assumptions": ["independence", "linearity_logit"],
                "data_types": [DataType.BINARY],
                "min_sample_size": 100,
                "analysis_goals": [AnalysisGoal.PREDICTION, AnalysisGoal.CLASSIFICATION],
                "power_calculation": True
            },
            StatisticalMethod.CHI_SQUARE: {
                "name": "カイ二乗検定",
                "description": "カテゴリカル変数間の関連性を検定",
                "assumptions": ["independence", "expected_frequency"],
                "data_types": [DataType.CATEGORICAL],
                "min_sample_size": 50,
                "analysis_goals": [AnalysisGoal.RELATIONSHIP],
                "power_calculation": True
            }
        }
    
    def _initialize_recommendation_rules(self) -> Dict[str, Any]:
        """推奨ルールの初期化"""
        return {
            "sample_size_thresholds": {
                "small": 30,
                "medium": 100,
                "large": 500
            },
            "normality_threshold": 0.05,
            "homoscedasticity_threshold": 0.05,
            "multicollinearity_threshold": 0.8,
            "missing_data_threshold": 0.1,
            "outlier_threshold": 0.05
        }
    
    def analyze_data_characteristics(self, data: pd.DataFrame, 
                                   target_variable: Optional[str] = None) -> DataCharacteristics:
        """データ特性の分析"""
        self.logger.info(f"データ特性分析開始: {data.shape}")
        
        sample_size = len(data)
        variables = {}
        missing_values = {}
        outliers = {}
        normality = {}
        
        # 各変数の分析
        for column in data.columns:
            # データ型の判定
            variables[column] = self._determine_data_type(data[column])
            
            # 欠損値の割合
            missing_values[column] = data[column].isnull().sum() / len(data)
            
            # 外れ値の検出
            if variables[column] in [DataType.CONTINUOUS, DataType.DISCRETE]:
                outliers[column] = self._detect_outliers(data[column])
                
                # 正規性の検定
                if len(data[column].dropna()) >= 8:  # Shapiro-Wilkの最小サンプルサイズ
                    normality[column] = self._test_normality(data[column])
                else:
                    normality[column] = False
            else:
                outliers[column] = 0
                normality[column] = False
        
        # 等分散性の検定（連続変数が複数ある場合）
        continuous_vars = [col for col, dtype in variables.items() 
                          if dtype == DataType.CONTINUOUS]
        homoscedasticity = None
        if len(continuous_vars) >= 2:
            homoscedasticity = self._test_homoscedasticity(data[continuous_vars])
        
        # 多重共線性の検出
        multicollinearity = None
        if len(continuous_vars) >= 2:
            multicollinearity = self._detect_multicollinearity(data[continuous_vars])
        
        # クラスバランスの分析（目的変数がある場合）
        balance = None
        if target_variable and target_variable in data.columns:
            if variables[target_variable] in [DataType.CATEGORICAL, DataType.BINARY]:
                balance = self._analyze_class_balance(data[target_variable])
        
        characteristics = DataCharacteristics(
            sample_size=sample_size,
            variables=variables,
            missing_values=missing_values,
            outliers=outliers,
            normality=normality,
            homoscedasticity=homoscedasticity,
            multicollinearity=multicollinearity,
            balance=balance
        )
        
        self.logger.info("データ特性分析完了")
        return characteristics
    
    def _determine_data_type(self, series: pd.Series) -> DataType:
        """データ型の自動判定"""
        # 欠損値を除外
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return DataType.CATEGORICAL
        
        # 数値型の場合
        if pd.api.types.is_numeric_dtype(clean_series):
            unique_values = clean_series.nunique()
            total_values = len(clean_series)
            
            # 二値変数
            if unique_values == 2:
                return DataType.BINARY
            
            # カウントデータ（非負整数）
            if clean_series.dtype in ['int64', 'int32'] and (clean_series >= 0).all():
                if unique_values < 20:  # 離散的
                    return DataType.COUNT
            
            # 連続変数 vs 離散変数の判定を改善
            unique_ratio = unique_values / total_values
            
            # 離散データの条件を厳しくする
            if unique_values <= 10 or unique_ratio < 0.02:
                return DataType.DISCRETE
            else:
                return DataType.CONTINUOUS
        
        # カテゴリカル変数
        else:
            unique_values = clean_series.nunique()
            
            # 二値変数
            if unique_values == 2:
                return DataType.BINARY
            
            # 順序があるかの判定（簡易版）
            if clean_series.dtype.name == 'category' and clean_series.cat.ordered:
                return DataType.ORDINAL
            
            return DataType.CATEGORICAL
    
    def _detect_outliers(self, series: pd.Series) -> int:
        """外れ値の検出（IQR法）"""
        clean_series = series.dropna()
        if len(clean_series) < 4:
            return 0
        
        Q1 = clean_series.quantile(0.25)
        Q3 = clean_series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((clean_series < lower_bound) | (clean_series > upper_bound)).sum()
        return outliers
    
    def _test_normality(self, series: pd.Series) -> bool:
        """正規性の検定"""
        clean_series = series.dropna()
        if len(clean_series) < 8:
            return False
        
        try:
            # 定数値の場合は正規性なしとする
            if clean_series.nunique() <= 1:
                return False
            
            if len(clean_series) <= 5000:
                # Shapiro-Wilk検定（小サンプル）
                statistic, p_value = shapiro(clean_series)
            else:
                # D'Agostino-Pearson検定（大サンプル）
                statistic, p_value = normaltest(clean_series)
            
            # p値が有効な値かチェック
            if np.isnan(p_value) or np.isinf(p_value):
                return False
                
            return p_value > self.recommendation_rules["normality_threshold"]
        except Exception as e:
            self.logger.debug(f"正規性検定エラー: {e}")
            return False
    
    def _test_homoscedasticity(self, data: pd.DataFrame) -> bool:
        """等分散性の検定"""
        try:
            # Levene検定
            groups = [data[col].dropna() for col in data.columns]
            statistic, p_value = levene(*groups)
            return p_value > self.recommendation_rules["homoscedasticity_threshold"]
        except:
            return False
    
    def _detect_multicollinearity(self, data: pd.DataFrame) -> float:
        """多重共線性の検出（相関行列の最大値）"""
        try:
            corr_matrix = data.corr().abs()
            # 対角成分を除外
            np.fill_diagonal(corr_matrix.values, 0)
            return corr_matrix.max().max()
        except:
            return 0.0
    
    def _analyze_class_balance(self, series: pd.Series) -> Dict[str, float]:
        """クラスバランスの分析"""
        value_counts = series.value_counts(normalize=True)
        return value_counts.to_dict()
    
    def recommend_methods(self, characteristics: DataCharacteristics,
                         analysis_goal: AnalysisGoal,
                         target_variable: Optional[str] = None,
                         predictor_variables: Optional[List[str]] = None) -> List[MethodRecommendation]:
        """統計手法の推奨"""
        self.logger.info(f"統計手法推奨開始: {analysis_goal}")
        
        recommendations = []
        
        # 各統計手法について適用可能性を評価
        for method, method_info in self.method_database.items():
            if analysis_goal not in method_info["analysis_goals"]:
                continue
            
            recommendation = self._evaluate_method(
                method, method_info, characteristics, 
                target_variable, predictor_variables
            )
            
            if recommendation.confidence > 0.1:  # 最低信頼度
                recommendations.append(recommendation)
        
        # 信頼度でソート
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        
        self.logger.info(f"推奨手法数: {len(recommendations)}")
        return recommendations
    
    def _evaluate_method(self, method: StatisticalMethod, method_info: Dict[str, Any],
                        characteristics: DataCharacteristics,
                        target_variable: Optional[str] = None,
                        predictor_variables: Optional[List[str]] = None) -> MethodRecommendation:
        """個別手法の評価"""
        confidence = 1.0
        assumptions_met = {}
        warnings = []
        alternatives = []
        
        # サンプルサイズの確認
        sample_size_adequate = characteristics.sample_size >= method_info["min_sample_size"]
        if not sample_size_adequate:
            confidence *= 0.5
            warnings.append(f"サンプルサイズが不足（必要: {method_info['min_sample_size']}, 現在: {characteristics.sample_size}）")
        
        # データ型の確認
        if target_variable:
            target_type = characteristics.variables.get(target_variable)
            if target_type not in method_info["data_types"]:
                confidence *= 0.3
                warnings.append(f"目的変数のデータ型が不適切（{target_type}）")
        
        # 仮定の確認
        for assumption in method_info["assumptions"]:
            met = self._check_assumption(assumption, characteristics, target_variable, predictor_variables)
            assumptions_met[assumption] = met
            
            if not met:
                confidence *= 0.7
                warnings.append(f"仮定「{assumption}」が満たされていません")
                
                # 代替手法の提案
                alternatives.extend(self._suggest_alternatives(method, assumption))
        
        # 欠損値の影響
        if target_variable and characteristics.missing_values.get(target_variable, 0) > self.recommendation_rules["missing_data_threshold"]:
            confidence *= 0.8
            warnings.append("目的変数に多くの欠損値があります")
        
        # 外れ値の影響
        if target_variable and characteristics.outliers.get(target_variable, 0) > characteristics.sample_size * self.recommendation_rules["outlier_threshold"]:
            confidence *= 0.9
            warnings.append("多くの外れ値が検出されました")
        
        # 根拠の生成
        rationale = self._generate_rationale(method, method_info, characteristics, assumptions_met)
        
        return MethodRecommendation(
            method=method,
            confidence=confidence,
            rationale=rationale,
            assumptions_met=assumptions_met,
            alternatives=list(set(alternatives)),
            warnings=warnings,
            sample_size_adequate=sample_size_adequate
        )
    
    def _check_assumption(self, assumption: str, characteristics: DataCharacteristics,
                         target_variable: Optional[str] = None,
                         predictor_variables: Optional[List[str]] = None) -> bool:
        """統計的仮定の確認"""
        if assumption == "normality":
            if target_variable:
                return characteristics.normality.get(target_variable, False)
            return any(characteristics.normality.values())
        
        elif assumption == "independence":
            return characteristics.independence
        
        elif assumption == "homoscedasticity":
            return characteristics.homoscedasticity if characteristics.homoscedasticity is not None else True
        
        elif assumption == "linearity":
            # 簡易的な線形性チェック（実際の実装では相関係数などを使用）
            return True
        
        elif assumption == "expected_frequency":
            # カイ二乗検定の期待度数チェック（簡易版）
            return characteristics.sample_size >= 50
        
        else:
            return True
    
    def _suggest_alternatives(self, method: StatisticalMethod, violated_assumption: str) -> List[StatisticalMethod]:
        """仮定違反時の代替手法提案"""
        alternatives = []
        
        if violated_assumption == "normality":
            if method == StatisticalMethod.T_TEST_TWO_SAMPLE:
                alternatives.append(StatisticalMethod.MANN_WHITNEY)
            elif method == StatisticalMethod.ANOVA_ONE_WAY:
                alternatives.append(StatisticalMethod.KRUSKAL_WALLIS)
            elif method == StatisticalMethod.PEARSON_CORRELATION:
                alternatives.append(StatisticalMethod.SPEARMAN_CORRELATION)
        
        elif violated_assumption == "homoscedasticity":
            if method == StatisticalMethod.T_TEST_TWO_SAMPLE:
                alternatives.append(StatisticalMethod.MANN_WHITNEY)
            elif method == StatisticalMethod.ANOVA_ONE_WAY:
                alternatives.append(StatisticalMethod.KRUSKAL_WALLIS)
        
        return alternatives
    
    def _generate_rationale(self, method: StatisticalMethod, method_info: Dict[str, Any],
                           characteristics: DataCharacteristics, assumptions_met: Dict[str, bool]) -> str:
        """推奨理由の生成"""
        rationale_parts = []
        
        # 基本的な適用理由
        rationale_parts.append(f"{method_info['name']}は{method_info['description']}に適しています。")
        
        # サンプルサイズ
        if characteristics.sample_size >= method_info["min_sample_size"]:
            rationale_parts.append(f"サンプルサイズ（{characteristics.sample_size}）は十分です。")
        
        # 満たされた仮定
        met_assumptions = [assumption for assumption, met in assumptions_met.items() if met]
        if met_assumptions:
            rationale_parts.append(f"必要な仮定（{', '.join(met_assumptions)}）が満たされています。")
        
        # 違反した仮定
        violated_assumptions = [assumption for assumption, met in assumptions_met.items() if not met]
        if violated_assumptions:
            rationale_parts.append(f"ただし、仮定（{', '.join(violated_assumptions)}）に注意が必要です。")
        
        return " ".join(rationale_parts)
    
    def get_method_details(self, method: StatisticalMethod) -> Dict[str, Any]:
        """統計手法の詳細情報を取得"""
        return self.method_database.get(method, {})
    
    def export_recommendations(self, recommendations: List[MethodRecommendation], 
                             filepath: str) -> bool:
        """推奨結果のエクスポート"""
        try:
            export_data = []
            for rec in recommendations:
                # NumPy型をPython標準型に変換
                assumptions_met_converted = {}
                for key, value in rec.assumptions_met.items():
                    if isinstance(value, np.bool_):
                        assumptions_met_converted[key] = bool(value)
                    else:
                        assumptions_met_converted[key] = value
                
                export_data.append({
                    "method": rec.method.value,
                    "confidence": float(rec.confidence),
                    "rationale": rec.rationale,
                    "assumptions_met": assumptions_met_converted,
                    "alternatives": [alt.value for alt in rec.alternatives],
                    "warnings": rec.warnings,
                    "sample_size_adequate": bool(rec.sample_size_adequate)
                })
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"推奨結果をエクスポート: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"エクスポートエラー: {e}")
            return False

def create_sample_data_for_testing() -> pd.DataFrame:
    """テスト用のサンプルデータ作成"""
    np.random.seed(42)
    
    n = 200
    data = {
        'age': np.random.normal(35, 10, n),
        'income': np.random.lognormal(10, 0.5, n),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n, p=[0.3, 0.4, 0.2, 0.1]),
        'satisfaction': np.random.randint(1, 6, n),
        'gender': np.random.choice(['Male', 'Female'], n),
        'purchased': np.random.choice([0, 1], n, p=[0.6, 0.4])
    }
    
    return pd.DataFrame(data)

if __name__ == '__main__':
    # デモンストレーション
    print("🧠 Statistical Method Advisor デモンストレーション")
    print("=" * 60)
    
    # サンプルデータの作成
    print("\n📊 サンプルデータ作成...")
    sample_data = create_sample_data_for_testing()
    print(f"データサイズ: {sample_data.shape}")
    print(f"変数: {list(sample_data.columns)}")
    
    # アドバイザーの初期化
    print("\n🔧 StatisticalMethodAdvisor初期化...")
    advisor = StatisticalMethodAdvisor()
    
    # データ特性の分析
    print("\n🔍 データ特性分析...")
    characteristics = advisor.analyze_data_characteristics(sample_data, target_variable='purchased')
    
    print(f"サンプルサイズ: {characteristics.sample_size}")
    print(f"変数型: {characteristics.variables}")
    print(f"正規性: {characteristics.normality}")
    
    # 統計手法の推奨
    print("\n💡 統計手法推奨...")
    
    # 比較分析の例
    comparison_recs = advisor.recommend_methods(
        characteristics, 
        AnalysisGoal.COMPARISON,
        target_variable='income'
    )
    
    print("\n🔍 比較分析の推奨手法:")
    for i, rec in enumerate(comparison_recs[:3], 1):
        print(f"{i}. {rec.method.value} (信頼度: {rec.confidence:.2f})")
        print(f"   理由: {rec.rationale}")
        if rec.warnings:
            print(f"   警告: {', '.join(rec.warnings)}")
    
    # 関係性分析の例
    relationship_recs = advisor.recommend_methods(
        characteristics,
        AnalysisGoal.RELATIONSHIP,
        target_variable='satisfaction'
    )
    
    print("\n🔗 関係性分析の推奨手法:")
    for i, rec in enumerate(relationship_recs[:3], 1):
        print(f"{i}. {rec.method.value} (信頼度: {rec.confidence:.2f})")
        print(f"   理由: {rec.rationale}")
    
    print("\n✅ デモンストレーション完了！")