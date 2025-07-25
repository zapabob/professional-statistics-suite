#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automated Assumption Validation System
自動統計的仮定検証システム

このモジュールは以下の機能を提供します:
- 統計手法の仮定の自動検証
- 仮定違反の重要度評価
- 代替手法の提案
- 詳細な診断レポート生成
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# 統計ライブラリ
from scipy import stats
from scipy.stats import (
    shapiro, normaltest, levene, chi2_contingency
)

# statsmodelsからDurbin-Watson検定をインポート
try:
    from statsmodels.stats.diagnostic import durbin_watson
except ImportError:
    # フォールバック実装
    def durbin_watson(residuals):
        """Durbin-Watson統計量の簡易計算"""
        diff = np.diff(residuals)
        return np.sum(diff**2) / np.sum(residuals**2)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# プロット関連（オプション）
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

# 設定とライセンス
try:
    from config import check_feature_permission
    if not check_feature_permission('advanced_ai'):
        raise ImportError("Advanced AI features require Professional edition or higher")
except ImportError:
    def check_feature_permission(feature):
        return True

# 統計手法アドバイザー
try:
    from statistical_method_advisor import StatisticalMethod, DataType
except ImportError:
    # フォールバック定義
    class StatisticalMethod(Enum):
        T_TEST_TWO_SAMPLE = "t_test_two_sample"
        ANOVA_ONE_WAY = "anova_one_way"
        LINEAR_REGRESSION = "linear_regression"
    
    class DataType(Enum):
        CONTINUOUS = "continuous"
        CATEGORICAL = "categorical"

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssumptionType(Enum):
    """統計的仮定の種類"""
    NORMALITY = "normality"
    HOMOSCEDASTICITY = "homoscedasticity"
    INDEPENDENCE = "independence"
    LINEARITY = "linearity"
    NO_MULTICOLLINEARITY = "no_multicollinearity"
    EXPECTED_FREQUENCY = "expected_frequency"
    RANDOM_SAMPLING = "random_sampling"
    ADDITIVITY = "additivity"

class ViolationSeverity(Enum):
    """仮定違反の重要度"""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

@dataclass
class AssumptionTest:
    """仮定検証テストの結果"""
    assumption: AssumptionType
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    is_violated: bool = False
    severity: ViolationSeverity = ViolationSeverity.NONE
    interpretation: str = ""
    recommendation: str = ""

@dataclass
class ValidationResult:
    """仮定検証の総合結果"""
    method: StatisticalMethod
    assumptions_tested: List[AssumptionTest]
    overall_validity: bool
    severity_summary: Dict[ViolationSeverity, int]
    alternative_methods: List[StatisticalMethod] = field(default_factory=list)
    diagnostic_plots: List[str] = field(default_factory=list)
    detailed_report: str = ""

class AssumptionValidator:
    """自動統計的仮定検証システム"""
    
    def __init__(self, alpha: float = 0.05, plot_diagnostics: bool = True):
        self.logger = logging.getLogger(f"{__name__}.AssumptionValidator")
        self.alpha = alpha  # 有意水準
        self.plot_diagnostics = plot_diagnostics
        
        # 仮定検証ルールの初期化
        self.validation_rules = self._initialize_validation_rules()
        
        # 重要度判定基準
        self.severity_thresholds = {
            ViolationSeverity.MILD: 0.01,
            ViolationSeverity.MODERATE: 0.001,
            ViolationSeverity.SEVERE: 0.0001,
            ViolationSeverity.CRITICAL: 0.00001
        }
        
        self.logger.info("AssumptionValidator初期化完了")
    
    def _initialize_validation_rules(self) -> Dict[StatisticalMethod, List[AssumptionType]]:
        """統計手法ごとの仮定検証ルール"""
        return {
            StatisticalMethod.T_TEST_TWO_SAMPLE: [
                AssumptionType.NORMALITY,
                AssumptionType.HOMOSCEDASTICITY,
                AssumptionType.INDEPENDENCE
            ],
            StatisticalMethod.ANOVA_ONE_WAY: [
                AssumptionType.NORMALITY,
                AssumptionType.HOMOSCEDASTICITY,
                AssumptionType.INDEPENDENCE
            ],
            StatisticalMethod.LINEAR_REGRESSION: [
                AssumptionType.LINEARITY,
                AssumptionType.INDEPENDENCE,
                AssumptionType.HOMOSCEDASTICITY,
                AssumptionType.NORMALITY,
                AssumptionType.NO_MULTICOLLINEARITY
            ]
        }
    
    def validate_assumptions(self, data: pd.DataFrame, method: StatisticalMethod,
                           target_variable: str, predictor_variables: List[str] = None,
                           group_variable: str = None) -> ValidationResult:
        """統計手法の仮定を包括的に検証"""
        self.logger.info(f"仮定検証開始: {method.value}")
        
        # 必要な仮定を取得
        required_assumptions = self.validation_rules.get(method, [])
        
        # 各仮定をテスト
        assumption_tests = []
        for assumption in required_assumptions:
            test_result = self._test_assumption(
                assumption, data, target_variable, 
                predictor_variables, group_variable, method
            )
            assumption_tests.append(test_result)
        
        # 総合評価
        overall_validity = all(not test.is_violated for test in assumption_tests)
        
        # 重要度サマリー
        severity_summary = {severity: 0 for severity in ViolationSeverity}
        for test in assumption_tests:
            severity_summary[test.severity] += 1
        
        # 代替手法の提案
        alternative_methods = self._suggest_alternatives(method, assumption_tests)
        
        # 診断プロット生成
        diagnostic_plots = []
        if self.plot_diagnostics:
            diagnostic_plots = self._generate_diagnostic_plots(
                data, method, target_variable, predictor_variables, group_variable
            )
        
        # 詳細レポート生成
        detailed_report = self._generate_detailed_report(
            method, assumption_tests, overall_validity
        )
        
        result = ValidationResult(
            method=method,
            assumptions_tested=assumption_tests,
            overall_validity=overall_validity,
            severity_summary=severity_summary,
            alternative_methods=alternative_methods,
            diagnostic_plots=diagnostic_plots,
            detailed_report=detailed_report
        )
        
        self.logger.info(f"仮定検証完了: 有効性={overall_validity}")
        return result
    
    def _test_assumption(self, assumption: AssumptionType, data: pd.DataFrame,
                        target_variable: str, predictor_variables: List[str] = None,
                        group_variable: str = None, method: StatisticalMethod = None) -> AssumptionTest:
        """個別の仮定をテスト"""
        
        if assumption == AssumptionType.NORMALITY:
            return self._test_normality(data, target_variable, group_variable)
        
        elif assumption == AssumptionType.HOMOSCEDASTICITY:
            return self._test_homoscedasticity(data, target_variable, group_variable, predictor_variables)
        
        elif assumption == AssumptionType.INDEPENDENCE:
            return self._test_independence(data, target_variable)
        
        elif assumption == AssumptionType.LINEARITY:
            return self._test_linearity(data, target_variable, predictor_variables)
        
        elif assumption == AssumptionType.NO_MULTICOLLINEARITY:
            return self._test_multicollinearity(data, predictor_variables)
        
        elif assumption == AssumptionType.EXPECTED_FREQUENCY:
            return self._test_expected_frequency(data, target_variable, group_variable)
        
        else:
            # デフォルトの仮定テスト
            return AssumptionTest(
                assumption=assumption,
                test_name="Not Implemented",
                statistic=0.0,
                p_value=1.0,
                is_violated=False,
                severity=ViolationSeverity.NONE,
                interpretation="このテストは実装されていません",
                recommendation="手動で確認してください"
            )
    
    def _test_normality(self, data: pd.DataFrame, target_variable: str, 
                       group_variable: str = None) -> AssumptionTest:
        """正規性の検定"""
        try:
            if group_variable:
                # グループ別の正規性検定
                groups = data.groupby(group_variable)[target_variable]
                p_values = []
                statistics = []
                
                for name, group in groups:
                    clean_group = group.dropna()
                    if len(clean_group) >= 8:
                        if len(clean_group) <= 5000:
                            stat, p_val = shapiro(clean_group)
                        else:
                            stat, p_val = normaltest(clean_group)
                        statistics.append(stat)
                        p_values.append(p_val)
                
                if p_values:
                    # 最小p値を使用（最も厳しい結果）
                    min_p_value = min(p_values)
                    avg_statistic = np.mean(statistics)
                    test_name = "Shapiro-Wilk (grouped)"
                else:
                    min_p_value = 1.0
                    avg_statistic = 0.0
                    test_name = "Normality test (insufficient data)"
            
            else:
                # 単一変数の正規性検定
                clean_data = data[target_variable].dropna()
                
                if len(clean_data) < 8:
                    return AssumptionTest(
                        assumption=AssumptionType.NORMALITY,
                        test_name="Normality test (insufficient data)",
                        statistic=0.0,
                        p_value=1.0,
                        is_violated=False,
                        severity=ViolationSeverity.NONE,
                        interpretation="サンプルサイズが不足しています",
                        recommendation="より多くのデータを収集してください"
                    )
                
                if len(clean_data) <= 5000:
                    avg_statistic, min_p_value = shapiro(clean_data)
                    test_name = "Shapiro-Wilk"
                else:
                    avg_statistic, min_p_value = normaltest(clean_data)
                    test_name = "D'Agostino-Pearson"
            
            # 仮定違反の判定
            is_violated = min_p_value < self.alpha
            severity = self._determine_severity(min_p_value)
            
            # 解釈と推奨事項
            if is_violated:
                interpretation = f"正規性が棄却されました (p={min_p_value:.4f})"
                recommendation = "ノンパラメトリック検定または変数変換を検討してください"
            else:
                interpretation = f"正規性が支持されました (p={min_p_value:.4f})"
                recommendation = "パラメトリック検定を使用できます"
            
            return AssumptionTest(
                assumption=AssumptionType.NORMALITY,
                test_name=test_name,
                statistic=avg_statistic,
                p_value=min_p_value,
                is_violated=is_violated,
                severity=severity,
                interpretation=interpretation,
                recommendation=recommendation
            )
            
        except Exception as e:
            self.logger.error(f"正規性検定エラー: {e}")
            return AssumptionTest(
                assumption=AssumptionType.NORMALITY,
                test_name="Normality test (error)",
                statistic=0.0,
                p_value=1.0,
                is_violated=False,
                severity=ViolationSeverity.NONE,
                interpretation=f"検定実行エラー: {e}",
                recommendation="データを確認してください"
            )
    
    def _test_homoscedasticity(self, data: pd.DataFrame, target_variable: str,
                              group_variable: str = None, predictor_variables: List[str] = None) -> AssumptionTest:
        """等分散性の検定"""
        try:
            if group_variable:
                # グループ間の等分散性検定
                groups = [group[target_variable].dropna() for name, group in data.groupby(group_variable)]
                
                if len(groups) < 2:
                    return AssumptionTest(
                        assumption=AssumptionType.HOMOSCEDASTICITY,
                        test_name="Homoscedasticity test (insufficient groups)",
                        statistic=0.0,
                        p_value=1.0,
                        is_violated=False,
                        severity=ViolationSeverity.NONE,
                        interpretation="グループが不足しています",
                        recommendation="複数のグループが必要です"
                    )
                
                # Levene検定を使用
                statistic, p_value = levene(*groups)
                test_name = "Levene's test"
                
            elif predictor_variables:
                # 回帰分析での等分散性検定（残差分析）
                X = data[predictor_variables].dropna()
                y = data[target_variable].dropna()
                
                # 共通のインデックスを取得
                common_index = X.index.intersection(y.index)
                X = X.loc[common_index]
                y = y.loc[common_index]
                
                if len(X) < 10:
                    return AssumptionTest(
                        assumption=AssumptionType.HOMOSCEDASTICITY,
                        test_name="Homoscedasticity test (insufficient data)",
                        statistic=0.0,
                        p_value=1.0,
                        is_violated=False,
                        severity=ViolationSeverity.NONE,
                        interpretation="データが不足しています",
                        recommendation="より多くのデータが必要です"
                    )
                
                # 線形回帰を実行
                model = LinearRegression()
                model.fit(X, y)
                residuals = y - model.predict(X)
                
                # Breusch-Pagan検定の簡易版
                # 残差の二乗を予測値で回帰
                fitted_values = model.predict(X)
                residuals_squared = residuals ** 2
                
                bp_model = LinearRegression()
                bp_model.fit(fitted_values.reshape(-1, 1), residuals_squared)
                bp_predictions = bp_model.predict(fitted_values.reshape(-1, 1))
                
                # F統計量の計算
                ss_reg = np.sum((bp_predictions - np.mean(residuals_squared)) ** 2)
                ss_res = np.sum((residuals_squared - bp_predictions) ** 2)
                
                df_reg = 1
                df_res = len(residuals_squared) - 2
                
                if ss_res > 0:
                    f_statistic = (ss_reg / df_reg) / (ss_res / df_res)
                    p_value = 1 - stats.f.cdf(f_statistic, df_reg, df_res)
                else:
                    f_statistic = 0.0
                    p_value = 1.0
                
                statistic = f_statistic
                test_name = "Breusch-Pagan test (simplified)"
                
            else:
                return AssumptionTest(
                    assumption=AssumptionType.HOMOSCEDASTICITY,
                    test_name="Homoscedasticity test (no groups or predictors)",
                    statistic=0.0,
                    p_value=1.0,
                    is_violated=False,
                    severity=ViolationSeverity.NONE,
                    interpretation="グループ変数または予測変数が必要です",
                    recommendation="適切な変数を指定してください"
                )
            
            # 仮定違反の判定
            is_violated = p_value < self.alpha
            severity = self._determine_severity(p_value)
            
            # 解釈と推奨事項
            if is_violated:
                interpretation = f"等分散性が棄却されました (p={p_value:.4f})"
                recommendation = "Welchのt検定や重み付き最小二乗法を検討してください"
            else:
                interpretation = f"等分散性が支持されました (p={p_value:.4f})"
                recommendation = "標準的な検定を使用できます"
            
            return AssumptionTest(
                assumption=AssumptionType.HOMOSCEDASTICITY,
                test_name=test_name,
                statistic=statistic,
                p_value=p_value,
                is_violated=is_violated,
                severity=severity,
                interpretation=interpretation,
                recommendation=recommendation
            )
            
        except Exception as e:
            self.logger.error(f"等分散性検定エラー: {e}")
            return AssumptionTest(
                assumption=AssumptionType.HOMOSCEDASTICITY,
                test_name="Homoscedasticity test (error)",
                statistic=0.0,
                p_value=1.0,
                is_violated=False,
                severity=ViolationSeverity.NONE,
                interpretation=f"検定実行エラー: {e}",
                recommendation="データを確認してください"
            )
    
    def _test_independence(self, data: pd.DataFrame, target_variable: str) -> AssumptionTest:
        """独立性の検定（簡易版）"""
        try:
            clean_data = data[target_variable].dropna()
            
            if len(clean_data) < 20:
                return AssumptionTest(
                    assumption=AssumptionType.INDEPENDENCE,
                    test_name="Independence test (insufficient data)",
                    statistic=0.0,
                    p_value=1.0,
                    is_violated=False,
                    severity=ViolationSeverity.NONE,
                    interpretation="データが不足しています",
                    recommendation="より多くのデータが必要です"
                )
            
            # Durbin-Watson検定（時系列の自己相関）
            # データが時系列順に並んでいると仮定
            dw_statistic = durbin_watson(clean_data)
            
            # DW統計量の解釈（2に近いほど独立性が高い）
            if 1.5 <= dw_statistic <= 2.5:
                is_violated = False
                severity = ViolationSeverity.NONE
                interpretation = f"独立性が支持されました (DW={dw_statistic:.3f})"
                recommendation = "データは独立していると考えられます"
            else:
                is_violated = True
                if dw_statistic < 1.5:
                    severity = ViolationSeverity.MODERATE
                    interpretation = f"正の自己相関が検出されました (DW={dw_statistic:.3f})"
                else:
                    severity = ViolationSeverity.MILD
                    interpretation = f"負の自己相関が検出されました (DW={dw_statistic:.3f})"
                recommendation = "時系列分析手法や一般化最小二乗法を検討してください"
            
            # p値の近似計算（簡易版）
            p_value = 2 * min(abs(dw_statistic - 2), 2 - abs(dw_statistic - 2)) / 2
            
            return AssumptionTest(
                assumption=AssumptionType.INDEPENDENCE,
                test_name="Durbin-Watson test",
                statistic=dw_statistic,
                p_value=p_value,
                is_violated=is_violated,
                severity=severity,
                interpretation=interpretation,
                recommendation=recommendation
            )
            
        except Exception as e:
            self.logger.error(f"独立性検定エラー: {e}")
            return AssumptionTest(
                assumption=AssumptionType.INDEPENDENCE,
                test_name="Independence test (error)",
                statistic=0.0,
                p_value=1.0,
                is_violated=False,
                severity=ViolationSeverity.NONE,
                interpretation=f"検定実行エラー: {e}",
                recommendation="データを確認してください"
            )
    
    def _test_linearity(self, data: pd.DataFrame, target_variable: str,
                       predictor_variables: List[str]) -> AssumptionTest:
        """線形性の検定"""
        try:
            if not predictor_variables:
                return AssumptionTest(
                    assumption=AssumptionType.LINEARITY,
                    test_name="Linearity test (no predictors)",
                    statistic=0.0,
                    p_value=1.0,
                    is_violated=False,
                    severity=ViolationSeverity.NONE,
                    interpretation="予測変数が指定されていません",
                    recommendation="予測変数を指定してください"
                )
            
            X = data[predictor_variables].dropna()
            y = data[target_variable].dropna()
            
            # 共通のインデックスを取得
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            if len(X) < 10:
                return AssumptionTest(
                    assumption=AssumptionType.LINEARITY,
                    test_name="Linearity test (insufficient data)",
                    statistic=0.0,
                    p_value=1.0,
                    is_violated=False,
                    severity=ViolationSeverity.NONE,
                    interpretation="データが不足しています",
                    recommendation="より多くのデータが必要です"
                )
            
            # 線形回帰モデル
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            linear_predictions = linear_model.predict(X)
            linear_mse = mean_squared_error(y, linear_predictions)
            
            # 多項式回帰モデル（2次）
            from sklearn.preprocessing import PolynomialFeatures
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            
            poly_model = LinearRegression()
            poly_model.fit(X_poly, y)
            poly_predictions = poly_model.predict(X_poly)
            poly_mse = mean_squared_error(y, poly_predictions)
            
            # F検定による線形性の検定
            n = len(y)
            p_linear = X.shape[1]
            p_poly = X_poly.shape[1]
            
            if poly_mse > 0 and p_poly > p_linear:
                f_statistic = ((linear_mse - poly_mse) / (p_poly - p_linear)) / (poly_mse / (n - p_poly))
                p_value = 1 - stats.f.cdf(f_statistic, p_poly - p_linear, n - p_poly)
            else:
                f_statistic = 0.0
                p_value = 1.0
            
            # 仮定違反の判定
            is_violated = p_value < self.alpha
            severity = self._determine_severity(p_value)
            
            # 解釈と推奨事項
            if is_violated:
                interpretation = f"非線形性が検出されました (p={p_value:.4f})"
                recommendation = "多項式回帰や非線形回帰を検討してください"
            else:
                interpretation = f"線形性が支持されました (p={p_value:.4f})"
                recommendation = "線形回帰を使用できます"
            
            return AssumptionTest(
                assumption=AssumptionType.LINEARITY,
                test_name="Linearity F-test",
                statistic=f_statistic,
                p_value=p_value,
                is_violated=is_violated,
                severity=severity,
                interpretation=interpretation,
                recommendation=recommendation
            )
            
        except Exception as e:
            self.logger.error(f"線形性検定エラー: {e}")
            return AssumptionTest(
                assumption=AssumptionType.LINEARITY,
                test_name="Linearity test (error)",
                statistic=0.0,
                p_value=1.0,
                is_violated=False,
                severity=ViolationSeverity.NONE,
                interpretation=f"検定実行エラー: {e}",
                recommendation="データを確認してください"
            )
    
    def _test_multicollinearity(self, data: pd.DataFrame, predictor_variables: List[str]) -> AssumptionTest:
        """多重共線性の検定"""
        try:
            if not predictor_variables or len(predictor_variables) < 2:
                return AssumptionTest(
                    assumption=AssumptionType.NO_MULTICOLLINEARITY,
                    test_name="Multicollinearity test (insufficient predictors)",
                    statistic=0.0,
                    p_value=1.0,
                    is_violated=False,
                    severity=ViolationSeverity.NONE,
                    interpretation="予測変数が不足しています",
                    recommendation="複数の予測変数が必要です"
                )
            
            X = data[predictor_variables].dropna()
            
            if len(X) < len(predictor_variables) + 5:
                return AssumptionTest(
                    assumption=AssumptionType.NO_MULTICOLLINEARITY,
                    test_name="Multicollinearity test (insufficient data)",
                    statistic=0.0,
                    p_value=1.0,
                    is_violated=False,
                    severity=ViolationSeverity.NONE,
                    interpretation="データが不足しています",
                    recommendation="より多くのデータが必要です"
                )
            
            # 相関行列の計算
            corr_matrix = X.corr()
            
            # 最大相関係数を取得（対角成分を除く）
            corr_values = corr_matrix.values
            np.fill_diagonal(corr_values, 0)
            max_correlation = np.max(np.abs(corr_values))
            
            # VIF（分散拡大要因）の計算
            from sklearn.linear_model import LinearRegression
            vif_values = []
            
            for i, var in enumerate(predictor_variables):
                # 他の変数でこの変数を予測
                other_vars = [v for j, v in enumerate(predictor_variables) if j != i]
                if len(other_vars) > 0:
                    X_others = X[other_vars]
                    y_var = X[var]
                    
                    model = LinearRegression()
                    model.fit(X_others, y_var)
                    r_squared = model.score(X_others, y_var)
                    
                    if r_squared < 0.999:  # 完全な多重共線性を避ける
                        vif = 1 / (1 - r_squared)
                    else:
                        vif = float('inf')
                    
                    vif_values.append(vif)
            
            max_vif = max(vif_values) if vif_values else 1.0
            
            # 多重共線性の判定
            # VIF > 10 または 相関係数 > 0.8 で問題とする
            is_violated = max_vif > 10 or max_correlation > 0.8
            
            if max_vif > 100:
                severity = ViolationSeverity.CRITICAL
            elif max_vif > 50:
                severity = ViolationSeverity.SEVERE
            elif max_vif > 10:
                severity = ViolationSeverity.MODERATE
            elif max_correlation > 0.8:
                severity = ViolationSeverity.MILD
            else:
                severity = ViolationSeverity.NONE
            
            # 解釈と推奨事項
            if is_violated:
                interpretation = f"多重共線性が検出されました (最大VIF={max_vif:.2f}, 最大相関={max_correlation:.3f})"
                recommendation = "変数選択、主成分分析、またはリッジ回帰を検討してください"
            else:
                interpretation = f"多重共線性は問題ありません (最大VIF={max_vif:.2f}, 最大相関={max_correlation:.3f})"
                recommendation = "現在の変数セットを使用できます"
            
            # p値の近似（VIFベース）
            p_value = max(0.001, 1 / max_vif) if max_vif > 1 else 1.0
            
            return AssumptionTest(
                assumption=AssumptionType.NO_MULTICOLLINEARITY,
                test_name="VIF and Correlation test",
                statistic=max_vif,
                p_value=p_value,
                is_violated=is_violated,
                severity=severity,
                interpretation=interpretation,
                recommendation=recommendation
            )
            
        except Exception as e:
            self.logger.error(f"多重共線性検定エラー: {e}")
            return AssumptionTest(
                assumption=AssumptionType.NO_MULTICOLLINEARITY,
                test_name="Multicollinearity test (error)",
                statistic=0.0,
                p_value=1.0,
                is_violated=False,
                severity=ViolationSeverity.NONE,
                interpretation=f"検定実行エラー: {e}",
                recommendation="データを確認してください"
            )
    
    def _test_expected_frequency(self, data: pd.DataFrame, target_variable: str,
                                group_variable: str) -> AssumptionTest:
        """期待度数の検定（カイ二乗検定用）"""
        try:
            if not group_variable:
                return AssumptionTest(
                    assumption=AssumptionType.EXPECTED_FREQUENCY,
                    test_name="Expected frequency test (no group variable)",
                    statistic=0.0,
                    p_value=1.0,
                    is_violated=False,
                    severity=ViolationSeverity.NONE,
                    interpretation="グループ変数が指定されていません",
                    recommendation="グループ変数を指定してください"
                )
            
            # クロス集計表の作成
            contingency_table = pd.crosstab(data[target_variable], data[group_variable])
            
            # カイ二乗検定の実行
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # 期待度数の最小値
            min_expected = np.min(expected)
            
            # 期待度数が5未満のセルの割合
            cells_below_5 = np.sum(expected < 5) / expected.size
            
            # 仮定違反の判定
            # 期待度数が5未満のセルが20%を超える場合は問題
            is_violated = cells_below_5 > 0.2 or min_expected < 1
            
            if min_expected < 1:
                severity = ViolationSeverity.SEVERE
            elif cells_below_5 > 0.5:
                severity = ViolationSeverity.MODERATE
            elif cells_below_5 > 0.2:
                severity = ViolationSeverity.MILD
            else:
                severity = ViolationSeverity.NONE
            
            # 解釈と推奨事項
            if is_violated:
                interpretation = f"期待度数が不十分です (最小期待度数={min_expected:.2f}, 5未満の割合={cells_below_5:.1%})"
                recommendation = "Fisherの正確検定やカテゴリの統合を検討してください"
            else:
                interpretation = f"期待度数は十分です (最小期待度数={min_expected:.2f}, 5未満の割合={cells_below_5:.1%})"
                recommendation = "カイ二乗検定を使用できます"
            
            return AssumptionTest(
                assumption=AssumptionType.EXPECTED_FREQUENCY,
                test_name="Expected frequency test",
                statistic=min_expected,
                p_value=cells_below_5,  # 違反割合をp値として使用
                is_violated=is_violated,
                severity=severity,
                interpretation=interpretation,
                recommendation=recommendation
            )
            
        except Exception as e:
            self.logger.error(f"期待度数検定エラー: {e}")
            return AssumptionTest(
                assumption=AssumptionType.EXPECTED_FREQUENCY,
                test_name="Expected frequency test (error)",
                statistic=0.0,
                p_value=1.0,
                is_violated=False,
                severity=ViolationSeverity.NONE,
                interpretation=f"検定実行エラー: {e}",
                recommendation="データを確認してください"
            )
    
    def _determine_severity(self, p_value: float) -> ViolationSeverity:
        """p値に基づく違反重要度の判定"""
        if p_value >= self.alpha:
            return ViolationSeverity.NONE
        
        for severity, threshold in self.severity_thresholds.items():
            if p_value >= threshold:
                return severity
        
        return ViolationSeverity.CRITICAL
    
    def _suggest_alternatives(self, method: StatisticalMethod, 
                            assumption_tests: List[AssumptionTest]) -> List[StatisticalMethod]:
        """仮定違反に基づく代替手法の提案"""
        alternatives = []
        
        # 違反した仮定を特定
        violated_assumptions = [test.assumption for test in assumption_tests if test.is_violated]
        
        if method == StatisticalMethod.T_TEST_TWO_SAMPLE:
            if AssumptionType.NORMALITY in violated_assumptions:
                alternatives.append("mann_whitney")
            if AssumptionType.HOMOSCEDASTICITY in violated_assumptions:
                alternatives.append("welch_t_test")
        
        elif method == StatisticalMethod.ANOVA_ONE_WAY:
            if AssumptionType.NORMALITY in violated_assumptions:
                alternatives.append("kruskal_wallis")
            if AssumptionType.HOMOSCEDASTICITY in violated_assumptions:
                alternatives.append("welch_anova")
        
        elif method == StatisticalMethod.LINEAR_REGRESSION:
            if AssumptionType.LINEARITY in violated_assumptions:
                alternatives.append("polynomial_regression")
                alternatives.append("gam")
            if AssumptionType.HOMOSCEDASTICITY in violated_assumptions:
                alternatives.append("weighted_least_squares")
            if AssumptionType.NO_MULTICOLLINEARITY in violated_assumptions:
                alternatives.append("ridge_regression")
                alternatives.append("lasso_regression")
        
        return alternatives
    
    def _generate_diagnostic_plots(self, data: pd.DataFrame, method: StatisticalMethod,
                                  target_variable: str, predictor_variables: List[str] = None,
                                  group_variable: str = None) -> List[str]:
        """診断プロットの生成"""
        plot_files = []
        
        try:
            if not self.plot_diagnostics or not PLOTTING_AVAILABLE:
                return plot_files
            
            # プロットの保存ディレクトリ
            plot_dir = Path("diagnostic_plots")
            plot_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 正規性診断プロット
            if method in [StatisticalMethod.T_TEST_TWO_SAMPLE, StatisticalMethod.ANOVA_ONE_WAY]:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # ヒストグラム
                data[target_variable].hist(bins=30, ax=axes[0])
                axes[0].set_title('Distribution of Target Variable')
                axes[0].set_xlabel(target_variable)
                
                # Q-Qプロット
                stats.probplot(data[target_variable].dropna(), dist="norm", plot=axes[1])
                axes[1].set_title('Q-Q Plot')
                
                plot_file = plot_dir / f"normality_diagnostic_{timestamp}.png"
                plt.savefig(plot_file)
                plt.close()
                plot_files.append(str(plot_file))
            
            # 回帰診断プロット
            elif method == StatisticalMethod.LINEAR_REGRESSION and predictor_variables:
                X = data[predictor_variables].dropna()
                y = data[target_variable].dropna()
                
                common_index = X.index.intersection(y.index)
                X = X.loc[common_index]
                y = y.loc[common_index]
                
                if len(X) > 5:
                    model = LinearRegression()
                    model.fit(X, y)
                    predictions = model.predict(X)
                    residuals = y - predictions
                    
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # 残差プロット
                    axes[0, 0].scatter(predictions, residuals)
                    axes[0, 0].axhline(y=0, color='r', linestyle='--')
                    axes[0, 0].set_title('Residuals vs Fitted')
                    axes[0, 0].set_xlabel('Fitted Values')
                    axes[0, 0].set_ylabel('Residuals')
                    
                    # 正規Q-Qプロット
                    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
                    axes[0, 1].set_title('Normal Q-Q Plot of Residuals')
                    
                    # スケール-ロケーションプロット
                    standardized_residuals = np.sqrt(np.abs(residuals / np.std(residuals)))
                    axes[1, 0].scatter(predictions, standardized_residuals)
                    axes[1, 0].set_title('Scale-Location Plot')
                    axes[1, 0].set_xlabel('Fitted Values')
                    axes[1, 0].set_ylabel('√|Standardized Residuals|')
                    
                    # 残差ヒストグラム
                    axes[1, 1].hist(residuals, bins=20)
                    axes[1, 1].set_title('Residuals Distribution')
                    axes[1, 1].set_xlabel('Residuals')
                    
                    plt.tight_layout()
                    plot_file = plot_dir / f"regression_diagnostic_{timestamp}.png"
                    plt.savefig(plot_file)
                    plt.close()
                    plot_files.append(str(plot_file))
            
        except Exception as e:
            self.logger.error(f"診断プロット生成エラー: {e}")
        
        return plot_files
    
    def _generate_detailed_report(self, method: StatisticalMethod, 
                                assumption_tests: List[AssumptionTest],
                                overall_validity: bool) -> str:
        """詳細レポートの生成"""
        report_lines = []
        
        report_lines.append("統計手法仮定検証レポート")
        report_lines.append("=" * 50)
        report_lines.append(f"検証対象手法: {method.value}")
        report_lines.append(f"検証日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"総合判定: {'有効' if overall_validity else '要注意'}")
        report_lines.append("")
        
        report_lines.append("個別仮定検証結果:")
        report_lines.append("-" * 30)
        
        for test in assumption_tests:
            report_lines.append(f"仮定: {test.assumption.value}")
            report_lines.append(f"  検定: {test.test_name}")
            report_lines.append(f"  統計量: {test.statistic:.4f}")
            report_lines.append(f"  p値: {test.p_value:.4f}")
            report_lines.append(f"  違反: {'はい' if test.is_violated else 'いいえ'}")
            report_lines.append(f"  重要度: {test.severity.value}")
            report_lines.append(f"  解釈: {test.interpretation}")
            report_lines.append(f"  推奨: {test.recommendation}")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def export_validation_report(self, result: ValidationResult, filepath: str) -> bool:
        """検証レポートのエクスポート"""
        try:
            export_data = {
                "method": result.method.value,
                "validation_timestamp": datetime.now().isoformat(),
                "overall_validity": bool(result.overall_validity),
                "severity_summary": {k.value: int(v) for k, v in result.severity_summary.items()},
                "assumptions_tested": [
                    {
                        "assumption": test.assumption.value,
                        "test_name": test.test_name,
                        "statistic": float(test.statistic),
                        "p_value": float(test.p_value),
                        "is_violated": bool(test.is_violated),
                        "severity": test.severity.value,
                        "interpretation": test.interpretation,
                        "recommendation": test.recommendation
                    }
                    for test in result.assumptions_tested
                ],
                "alternative_methods": result.alternative_methods,
                "diagnostic_plots": result.diagnostic_plots,
                "detailed_report": result.detailed_report
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"検証レポートをエクスポート: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"レポートエクスポートエラー: {e}")
            return False

def create_sample_data_for_assumption_testing() -> pd.DataFrame:
    """仮定検証テスト用のサンプルデータ作成"""
    np.random.seed(42)
    
    n = 100
    
    # 正規分布データ
    normal_data = np.random.normal(50, 10, n)
    
    # 非正規分布データ（指数分布）
    non_normal_data = np.random.exponential(2, n)
    
    # グループ変数
    groups = np.random.choice(['A', 'B', 'C'], n)
    
    # 予測変数（多重共線性あり）
    x1 = np.random.normal(0, 1, n)
    x2 = 2 * x1 + np.random.normal(0, 0.1, n)  # x1と高い相関
    x3 = np.random.normal(0, 1, n)
    
    # 目的変数（線形関係）
    y_linear = 2 * x1 + 3 * x3 + np.random.normal(0, 1, n)
    
    # 目的変数（非線形関係）
    y_nonlinear = x1**2 + x3 + np.random.normal(0, 1, n)
    
    return pd.DataFrame({
        'normal_var': normal_data,
        'non_normal_var': non_normal_data,
        'group': groups,
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'y_linear': y_linear,
        'y_nonlinear': y_nonlinear
    })

if __name__ == '__main__':
    # デモンストレーション
    print("🔍 Assumption Validator デモンストレーション")
    print("=" * 60)
    
    # サンプルデータの作成
    print("\n📊 サンプルデータ作成...")
    sample_data = create_sample_data_for_assumption_testing()
    print(f"データサイズ: {sample_data.shape}")
    print(f"変数: {list(sample_data.columns)}")
    
    # バリデーターの初期化
    print("\n🔧 AssumptionValidator初期化...")
    validator = AssumptionValidator(alpha=0.05, plot_diagnostics=False)
    
    # t検定の仮定検証
    print("\n🧪 t検定の仮定検証...")
    t_test_result = validator.validate_assumptions(
        data=sample_data,
        method=StatisticalMethod.T_TEST_TWO_SAMPLE,
        target_variable='normal_var',
        group_variable='group'
    )
    
    print(f"総合有効性: {t_test_result.overall_validity}")
    print(f"重要度サマリー: {t_test_result.severity_summary}")
    
    for test in t_test_result.assumptions_tested:
        print(f"\n仮定: {test.assumption.value}")
        print(f"  検定: {test.test_name}")
        print(f"  p値: {test.p_value:.4f}")
        print(f"  違反: {test.is_violated}")
        print(f"  重要度: {test.severity.value}")
    
    # 線形回帰の仮定検証
    print("\n📈 線形回帰の仮定検証...")
    regression_result = validator.validate_assumptions(
        data=sample_data,
        method=StatisticalMethod.LINEAR_REGRESSION,
        target_variable='y_linear',
        predictor_variables=['x1', 'x2', 'x3']
    )
    
    print(f"総合有効性: {regression_result.overall_validity}")
    print(f"代替手法: {regression_result.alternative_methods}")
    
    for test in regression_result.assumptions_tested:
        print(f"\n仮定: {test.assumption.value}")
        print(f"  検定: {test.test_name}")
        print(f"  統計量: {test.statistic:.4f}")
        print(f"  違反: {test.is_violated}")
        print(f"  推奨: {test.recommendation}")
    
    print("\n✅ デモンストレーション完了！")