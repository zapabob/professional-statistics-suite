#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistical Power Analysis Module
統計的検出力分析モジュール

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
License: MIT
"""

import numpy as np
from typing import Dict, Any
from builtins import str as builtin_str
import statsmodels.stats.power as smp
import math

# プロフェッショナル機能インポート
try:
    from professional_utils import professional_logger, performance_monitor
    PROFESSIONAL_LOGGING = True
except ImportError:
    PROFESSIONAL_LOGGING = False

class PowerAnalysisEngine:
    """
    統計的検出力分析とサンプルサイズ計算を行うクラス
    """

    def __init__(self):
        if PROFESSIONAL_LOGGING:
            professional_logger.info("PowerAnalysisEngine 初期化")

    @performance_monitor.monitor_function if PROFESSIONAL_LOGGING else lambda x: x
    def calculate_sample_size_t_test(self, effect_size: float, alpha: float = 0.05, power: float = 0.8, alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        t検定のサンプルサイズを計算する。

        Args:
            effect_size (float): Cohen's d (効果量)。
            alpha (float): 有意水準 (Type I error rate)。デフォルトは0.05。
            power (float): 検出力 (1 - Type II error rate)。デフォルトは0.8。
            alternative (str): 検定のタイプ ('two-sided', 'one-sided')。デフォルトは'two-sided'。

        Returns:
            Dict[str, Any]: 計算結果を含む辞書。
        """
        try:
            # 両側検定か片側検定かによって、power analysisの関数を使い分ける
            if alternative == 'two-sided':
                nobs = smp.TTestIndPower().solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    ratio=1.0,  # 2群のサンプルサイズが等しいと仮定
                    alternative=alternative
                )
            elif alternative == 'one-sided':
                nobs = smp.TTestIndPower().solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    ratio=1.0,
                    alternative=alternative
                )
            else:
                raise ValueError("alternativeは'two-sided'または'one-sided'である必要があります。")

            if np.isnan(nobs) or nobs <= 0:
                raise ValueError("計算されたサンプルサイズが不正です。効果量、有意水準、検出力を確認してください。")

            return {
                "success": True,
                "method": "t-test",
                "effect_size": effect_size,
                "alpha": alpha,
                "power": power,
                "alternative": alternative,
                "sample_size_per_group": math.ceil(nobs),
                "total_sample_size": math.ceil(nobs) * 2
            }
        except Exception as e:
            if PROFESSIONAL_LOGGING:
                professional_logger.error(f"t検定のサンプルサイズ計算エラー: {e}")
            return {"success": False, "error": builtin_str(e)}

    @performance_monitor.monitor_function if PROFESSIONAL_LOGGING else lambda x: x
    def calculate_power_t_test(self, effect_size: float, nobs_per_group: int, alpha: float = 0.05, alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        t検定の検出力を計算する。

        Args:
            effect_size (float): Cohen's d (効果量)。
            nobs_per_group (int): 1群あたりのサンプルサイズ。
            alpha (float): 有意水準 (Type I error rate)。デフォルトは0.05。
            alternative (str): 検定のタイプ ('two-sided', 'one-sided')。デフォルトは'two-sided'。

        Returns:
            Dict[str, Any]: 計算結果を含む辞書。
        """
        try:
            power = smp.TTestIndPower().solve_power(
                effect_size=effect_size,
                nobs1=nobs_per_group,
                alpha=alpha,
                ratio=1.0,
                alternative=alternative
            )
            return {
                "success": True,
                "method": "t-test",
                "effect_size": effect_size,
                "nobs_per_group": nobs_per_group,
                "alpha": alpha,
                "alternative": alternative,
                "power": power
            }
        except Exception as e:
            if PROFESSIONAL_LOGGING:
                professional_logger.error(f"t検定の検出力計算エラー: {e}")
            return {"success": False, "error": builtin_str(e)}

    @performance_monitor.monitor_function if PROFESSIONAL_LOGGING else lambda x: x
    def calculate_sample_size_anova(self, k_groups: int, effect_size: float, alpha: float = 0.05, power: float = 0.8) -> Dict[str, Any]:
        """
        一元配置分散分析 (ANOVA) のサンプルサイズを計算する。

        Args:
            k_groups (int): グループ数。
            effect_size (float): Cohen's f (効果量)。
            alpha (float): 有意水準。デフォルトは0.05。
            power (float): 検出力。デフォルトは0.8。

        Returns:
            Dict[str, Any]: 計算結果を含む辞書。
        """
        try:
            nobs = smp.FTestAnovaPower().solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                k_groups=k_groups
            )
            if np.isnan(nobs) or nobs <= 0:
                raise ValueError("計算されたサンプルサイズが不正です。効果量、有意水準、検出力を確認してください。")

            return {
                "success": True,
                "method": "ANOVA",
                "k_groups": k_groups,
                "effect_size": effect_size,
                "alpha": alpha,
                "power": power,
                "sample_size_per_group": math.ceil(nobs),
                "total_sample_size": math.ceil(nobs) * k_groups
            }
        except Exception as e:
            if PROFESSIONAL_LOGGING:
                professional_logger.error(f"ANOVAのサンプルサイズ計算エラー: {e}")
            return {"success": False, "error": builtin_str(e)}

    @performance_monitor.monitor_function if PROFESSIONAL_LOGGING else lambda x: x
    def calculate_power_anova(self, k_groups: int, effect_size: float, nobs_per_group: int, alpha: float = 0.05) -> Dict[str, Any]:
        """
        一元配置分散分析 (ANOVA) の検出力を計算する。

        Args:
            k_groups (int): グループ数。
            effect_size (float): Cohen's f (効果量)。
            nobs_per_group (int): 1群あたりのサンプルサイズ。
            alpha (float): 有意水準。デフォルトは0.05。

        Returns:
            Dict[str, Any]: 計算結果を含む辞書。
        """
        try:
            power = smp.FTestAnovaPower().solve_power(
                effect_size=effect_size,
                nobs=nobs_per_group,
                alpha=alpha,
                k_groups=k_groups
            )
            return {
                "success": True,
                "method": "ANOVA",
                "k_groups": k_groups,
                "effect_size": effect_size,
                "nobs_per_group": nobs_per_group,
                "alpha": alpha,
                "power": power
            }
        except Exception as e:
            if PROFESSIONAL_LOGGING:
                professional_logger.error(f"ANOVAの検出力計算エラー: {e}")
            return {"success": False, "error": builtin_str(e)}

    @performance_monitor.monitor_function if PROFESSIONAL_LOGGING else lambda x: x
    def estimate_effect_size_from_t(self, t_statistic: float, df: int) -> Dict[str, Any]:
        """
        t統計量と自由度からCohen's dを推定する。

        Args:
            t_statistic (float): t統計量。
            df (int): 自由度。

        Returns:
            Dict[str, Any]: 推定された効果量と信頼区間を含む辞書。
        """
        try:
            # Cohen's dの推定
            cohens_d = t_statistic / math.sqrt(df / 2) # 2群のサンプルサイズが等しいと仮定した場合の近似

            # Cohen's dの信頼区間 (近似)
            # より正確な信頼区間は非心t分布を使用する必要があるが、ここでは簡易的な方法
            # 簡易的な標準誤差の計算 (Hedges' gの標準誤差の近似)
            # dの標準誤差 = sqrt((n1+n2)/(n1*n2) + d^2 / (2*(n1+n2-2)))
            # ここではn1=n2=nと仮定し、df = 2n-2 -> n = df/2 + 1
            n_per_group = df / 2 + 1
            if n_per_group < 2:
                raise ValueError("自由度が小さすぎます。")
            
            se_d = math.sqrt( (2/n_per_group) + (cohens_d**2) / (2 * (n_per_group * 2 - 2)) )
            
            # 95%信頼区間
            lower_ci = cohens_d - 1.96 * se_d
            upper_ci = cohens_d + 1.96 * se_d

            return {
                "success": True,
                "effect_size_type": "Cohen's d",
                "effect_size": cohens_d,
                "confidence_interval_95": (lower_ci, upper_ci),
                "interpretation": self._interpret_cohens_d(cohens_d)
            }
        except Exception as e:
            if PROFESSIONAL_LOGGING:
                professional_logger.error(f"t統計量からの効果量推定エラー: {e}")
            return {"success": False, "error": builtin_str(e)}

    @performance_monitor.monitor_function if PROFESSIONAL_LOGGING else lambda x: x
    def estimate_effect_size_from_f(self, f_statistic: float, df1: int, df2: int) -> Dict[str, Any]:
        """
        F統計量と自由度からCohen's fを推定する。

        Args:
            f_statistic (float): F統計量。
            df1 (int): 分子自由度。
            df2 (int): 分母自由度。

        Returns:
            Dict[str, Any]: 推定された効果量を含む辞書。
        """
        try:
            # Cohen's fの推定
            # f = sqrt(F * df1 / N)
            # N = df2 + df1 + 1 (近似)
            total_n = df1 + df2 + 1
            cohens_f = math.sqrt((f_statistic * df1) / total_n)

            # Eta squared (η²) の推定
            eta_squared = (f_statistic * df1) / (f_statistic * df1 + df2)

            return {
                "success": True,
                "effect_size_type": "Cohen's f",
                "effect_size_cohens_f": cohens_f,
                "effect_size_eta_squared": eta_squared,
                "interpretation_eta_squared": self._interpret_eta_squared(eta_squared)
            }
        except Exception as e:
            if PROFESSIONAL_LOGGING:
                professional_logger.error(f"F統計量からの効果量推定エラー: {e}")
            return {"success": False, "error": builtin_str(e)}

    def _interpret_cohens_d(self, d: float) -> str:
        """Cohen's dの解釈"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible (無視できる)"
        elif abs_d < 0.5:
            return "small (小さい)"
        elif abs_d < 0.8:
            return "medium (中程度)"
        else:
            return "large (大きい)"

    def _interpret_eta_squared(self, eta_sq: float) -> str:
        """η²の解釈"""
        if eta_sq < 0.01:
            return "negligible (無視できる)"
        elif eta_sq < 0.06:
            return "small (小さい)"
        elif eta_sq < 0.14:
            return "medium (中程度)"
        else:
            return "large (大きい)"

# インスタンス作成
power_analyzer = PowerAnalysisEngine()
