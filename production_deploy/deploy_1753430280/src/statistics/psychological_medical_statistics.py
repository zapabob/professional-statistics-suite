#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Psychological and Medical Statistics Module
心理医療統計分析モジュール

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
License: MIT
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, kruskal, wilcoxon
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest, proportions_chisquare
from statsmodels.stats.power import TTestPower, FTestAnovaPower
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class PsychologicalMedicalStats:
    """心理医療統計分析クラス"""
    
    def __init__(self):
        """初期化"""
        self.analysis_results = {}
        
    def clinical_trial_analysis(self, data: pd.DataFrame, group_col: str, outcome_col: str, 
                              baseline_col: str = None, time_col: str = None) -> Dict[str, Any]:
        """
        臨床試験分析
        
        Args:
            data: データフレーム
            group_col: 治療群変数名
            outcome_col: アウトカム変数名
            baseline_col: ベースライン値変数名（オプション）
            time_col: 時間変数名（オプション）
        """
        try:
            results = {}
            
            # 記述統計
            desc_stats = data.groupby(group_col)[outcome_col].agg([
                'count', 'mean', 'std', 'median', 'min', 'max'
            ]).round(4)
            results['descriptive'] = desc_stats
            
            # グループ間比較
            groups = data.groupby(group_col)[outcome_col].apply(lambda x: x.dropna()).to_dict()
            group_names = list(groups.keys())
            
            if len(group_names) == 2:
                # 2群比較
                group1 = np.array(groups[group_names[0]])
                group2 = np.array(groups[group_names[1]])
                
                # t検定
                t_stat, t_p = stats.ttest_ind(group1, group2)
                
                # Mann-Whitney U検定（ノンパラメトリック）
                mw_stat, mw_p = mannwhitneyu(group1, group2, alternative='two-sided')
                
                # 効果量（Cohen's d）
                cohens_d = self._calculate_cohens_d(group1, group2)
                
                results['group_comparison'] = {
                    't_test': {'statistic': t_stat, 'p_value': t_p},
                    'mann_whitney': {'statistic': mw_stat, 'p_value': mw_p},
                    'effect_size': {'cohens_d': cohens_d, 'interpretation': self._interpret_cohens_d(cohens_d)}
                }
            
            elif len(group_names) > 2:
                # 多群比較
                group_data = list(groups.values())
                
                # ANOVA
                f_stat, f_p = stats.f_oneway(*group_data)
                
                # Kruskal-Wallis検定（ノンパラメトリック）
                kw_stat, kw_p = kruskal(*group_data)
                
                # 効果量（eta squared）
                grand_mean = data[outcome_col].mean()
                ss_between = sum([len(g) * (np.mean(g) - grand_mean)**2 for g in group_data])
                ss_total = sum([(x - grand_mean)**2 for x in data[outcome_col].dropna()])
                eta_squared = ss_between / ss_total
                
                results['group_comparison'] = {
                    'anova': {'f_statistic': f_stat, 'p_value': f_p},
                    'kruskal_wallis': {'statistic': kw_stat, 'p_value': kw_p},
                    'effect_size': {'eta_squared': eta_squared, 'interpretation': self._interpret_eta_squared(eta_squared)}
                }
            
            # ベースライン調整分析
            if baseline_col and baseline_col in data.columns:
                baseline_results = self._baseline_adjusted_analysis(data, group_col, outcome_col, baseline_col)
                results['baseline_adjusted'] = baseline_results
            
            # 時系列分析
            if time_col and time_col in data.columns:
                time_results = self._time_series_analysis(data, group_col, outcome_col, time_col)
                results['time_series'] = time_results
            
            return {"success": True, "results": results}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _baseline_adjusted_analysis(self, data: pd.DataFrame, group_col: str, outcome_col: str, 
                                  baseline_col: str) -> Dict[str, Any]:
        """ベースライン調整分析"""
        try:
            # ベースライン調整値の計算
            data['baseline_adjusted'] = data[outcome_col] - data[baseline_col]
            
            # 調整後のグループ比較
            groups = data.groupby(group_col)['baseline_adjusted'].apply(lambda x: x.dropna()).to_dict()
            group_names = list(groups.keys())
            
            if len(group_names) == 2:
                group1 = np.array(groups[group_names[0]])
                group2 = np.array(groups[group_names[1]])
                
                # t検定
                t_stat, t_p = stats.ttest_ind(group1, group2)
                
                # 効果量
                cohens_d = self._calculate_cohens_d(group1, group2)
                
                return {
                    't_test': {'statistic': t_stat, 'p_value': t_p},
                    'effect_size': {'cohens_d': cohens_d, 'interpretation': self._interpret_cohens_d(cohens_d)}
                }
            
            return {"error": "ベースライン調整分析は2群比較のみ対応"}
            
        except Exception as e:
            return {"error": str(e)}
    
    def _time_series_analysis(self, data: pd.DataFrame, group_col: str, outcome_col: str, 
                             time_col: str) -> Dict[str, Any]:
        """時系列分析"""
        try:
            # 時系列データの整理
            time_data = data.pivot_table(
                index=time_col, 
                columns=group_col, 
                values=outcome_col, 
                aggfunc='mean'
            )
            
            # 各時点でのグループ比較
            time_comparisons = {}
            for time_point in time_data.index:
                group_values = [time_data.loc[time_point, col] for col in time_data.columns if pd.notna(time_data.loc[time_point, col])]
                if len(group_values) == 2:
                    t_stat, t_p = stats.ttest_ind(group_values[0], group_values[1])
                    time_comparisons[time_point] = {'t_statistic': t_stat, 'p_value': t_p}
            
            return {
                'time_comparisons': time_comparisons,
                'time_series_data': time_data.to_dict()
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def survival_analysis(self, data: pd.DataFrame, time_col: str, event_col: str, 
                         group_col: str = None) -> Dict[str, Any]:
        """
        生存時間分析
        
        Args:
            data: データフレーム
            time_col: 生存時間変数名
            event_col: イベント変数名（1=イベント発生、0=打ち切り）
            group_col: グループ変数名（オプション）
        """
        try:
            from lifelines import KaplanMeierFitter, LogRankTest
            
            results = {}
            
            # 全体の生存曲線
            kmf = KaplanMeierFitter()
            kmf.fit(data[time_col], data[event_col])
            
            results['overall'] = {
                'median_survival': kmf.median_survival_time_,
                'mean_survival': kmf.mean_survival_time_,
                'survival_curve': kmf.survival_function_.to_dict()
            }
            
            # グループ別生存分析
            if group_col:
                group_results = {}
                groups = data.groupby(group_col)
                
                for group_name, group_data in groups:
                    kmf_group = KaplanMeierFitter()
                    kmf_group.fit(group_data[time_col], group_data[event_col])
                    
                    group_results[group_name] = {
                        'median_survival': kmf_group.median_survival_time_,
                        'mean_survival': kmf_group.mean_survival_time_,
                        'survival_curve': kmf_group.survival_function_.to_dict()
                    }
                
                results['by_group'] = group_results
                
                # Log-rank検定
                if len(groups) == 2:
                    group_names = list(groups.groups.keys())
                    group1_data = groups.get_group(group_names[0])
                    group2_data = groups.get_group(group_names[1])
                    
                    lr_test = LogRankTest(group1_data[time_col], group2_data[time_col],
                                        group1_data[event_col], group2_data[event_col])
                    
                    results['log_rank_test'] = {
                        'statistic': lr_test.test_statistic,
                        'p_value': lr_test.p_value
                    }
            
            return {"success": True, "results": results}
            
        except ImportError:
            return {"success": False, "error": "lifelinesライブラリが必要です"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def repeated_measures_analysis(self, data: pd.DataFrame, subject_col: str, time_col: str, 
                                 value_col: str, group_col: str = None) -> Dict[str, Any]:
        """
        反復測定分析
        
        Args:
            data: データフレーム
            subject_col: 被験者ID変数名
            time_col: 時間/条件変数名
            value_col: 測定値変数名
            group_col: グループ変数名（オプション）
        """
        try:
            results = {}
            
            # データの整理
            pivot_data = data.pivot_table(
                index=subject_col,
                columns=time_col,
                values=value_col,
                aggfunc='mean'
            )
            
            # 記述統計
            desc_stats = pivot_data.describe()
            results['descriptive'] = desc_stats.to_dict()
            
            # 球面性検定
            sphericity_result = self._mauchly_sphericity_test(pivot_data)
            results['sphericity_test'] = sphericity_result
            
            # 反復測定ANOVA
            if group_col:
                # 混合要因ANOVA
                formula = f"{value_col} ~ C({time_col}) + C({group_col}) + C({time_col}):C({group_col})"
                model = ols(formula, data=data).fit()
                anova_table = anova_lm(model, typ=2)
                results['mixed_anova'] = anova_table.to_dict()
            else:
                # 一元反復測定ANOVA
                formula = f"{value_col} ~ C({time_col})"
                model = ols(formula, data=data).fit()
                anova_table = anova_lm(model, typ=2)
                results['repeated_anova'] = anova_table.to_dict()
            
            # 事後検定（時間効果）
            if len(pivot_data.columns) > 2:
                post_hoc = self._repeated_measures_post_hoc(data, time_col, value_col)
                results['post_hoc'] = post_hoc
            
            return {"success": True, "results": results}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _mauchly_sphericity_test(self, pivot_data: pd.DataFrame) -> Dict[str, Any]:
        """Mauchly球面性検定"""
        try:
            # 差分行列の計算
            diff_matrix = pivot_data.diff(axis=1).dropna(axis=1)
            
            # 共分散行列の計算
            cov_matrix = diff_matrix.cov()
            
            # Mauchly's W統計量の計算
            n = len(pivot_data)
            p = len(diff_matrix.columns)
            
            # 行列式の計算
            det_cov = np.linalg.det(cov_matrix.values)
            trace_cov = np.trace(cov_matrix.values)
            
            # Mauchly's W
            w = det_cov / ((trace_cov / p) ** p)
            
            # 自由度
            df = p * (p - 1) / 2
            
            # カイ二乗統計量
            chi_square = -(n - 1) * np.log(w)
            
            # p値（カイ二乗分布）
            p_value = 1 - stats.chi2.cdf(chi_square, df)
            
            return {
                "statistic": w,
                "chi_square": chi_square,
                "df": df,
                "p_value": p_value,
                "sphericity_assumption": p_value > 0.05
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _repeated_measures_post_hoc(self, data: pd.DataFrame, time_col: str, value_col: str) -> Dict[str, Any]:
        """反復測定事後検定"""
        try:
            # ペアワイズ比較
            time_points = data[time_col].unique()
            comparisons = []
            
            for i in range(len(time_points)):
                for j in range(i+1, len(time_points)):
                    time1_data = data[data[time_col] == time_points[i]][value_col]
                    time2_data = data[data[time_col] == time_points[j]][value_col]
                    
                    # 対応ありt検定
                    t_stat, p_value = stats.ttest_rel(time1_data, time2_data)
                    
                    comparisons.append({
                        'time1': time_points[i],
                        'time2': time_points[j],
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'mean_diff': time1_data.mean() - time2_data.mean()
                    })
            
            # Bonferroni補正
            n_comparisons = len(comparisons)
            for comp in comparisons:
                comp['p_adj_bonferroni'] = min(comp['p_value'] * n_comparisons, 1.0)
                comp['significant_bonferroni'] = comp['p_adj_bonferroni'] < 0.05
            
            return {
                'method': 'Bonferroni',
                'comparisons': comparisons,
                'n_comparisons': n_comparisons
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def power_analysis(self, effect_size: float, alpha: float = 0.05, power: float = 0.8, 
                      test_type: str = 't_test') -> Dict[str, Any]:
        """
        検出力分析
        
        Args:
            effect_size: 効果量
            alpha: 有意水準
            power: 検出力
            test_type: 検定タイプ
        """
        try:
            results = {}
            
            if test_type == 't_test':
                power_analysis = TTestPower()
                
                # 必要なサンプルサイズ
                n_needed = power_analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    alternative='two-sided'
                )
                
                # 実際の検出力
                actual_power = power_analysis.power(
                    effect_size=effect_size,
                    nobs=n_needed,
                    alpha=alpha,
                    alternative='two-sided'
                )
                
                results = {
                    'required_sample_size': int(n_needed),
                    'actual_power': actual_power,
                    'effect_size': effect_size,
                    'alpha': alpha,
                    'target_power': power
                }
            
            elif test_type == 'anova':
                power_analysis = FTestAnovaPower()
                
                # 必要なサンプルサイズ
                n_needed = power_analysis.solve_power(
                    effect_size=effect_size,
                    alpha=alpha,
                    power=power,
                    k_groups=3  # デフォルト3群
                )
                
                results = {
                    'required_sample_size_per_group': int(n_needed),
                    'total_sample_size': int(n_needed * 3),
                    'effect_size': effect_size,
                    'alpha': alpha,
                    'target_power': power
                }
            
            return {"success": True, "results": results}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def reliability_analysis(self, data: pd.DataFrame, items: List[str]) -> Dict[str, Any]:
        """
        信頼性分析（Cronbach's α）
        
        Args:
            data: データフレーム
            items: 項目名のリスト
        """
        try:
            # 項目データの抽出
            item_data = data[items].dropna()
            
            if len(item_data) < 2:
                return {"success": False, "error": "2つ以上の項目が必要です"}
            
            # 項目間相関行列
            corr_matrix = item_data.corr()
            
            # 項目分散
            item_variances = item_data.var()
            
            # 総分散
            total_variance = item_data.sum(axis=1).var()
            
            # Cronbach's α計算
            n_items = len(items)
            sum_item_variance = item_variances.sum()
            
            alpha = (n_items / (n_items - 1)) * (1 - sum_item_variance / total_variance)
            
            # 項目削除時のα
            alpha_if_deleted = {}
            for item in items:
                remaining_items = [i for i in items if i != item]
                if len(remaining_items) >= 2:
                    remaining_data = item_data[remaining_items]
                    remaining_var = remaining_data.var().sum()
                    remaining_total_var = remaining_data.sum(axis=1).var()
                    alpha_if_deleted[item] = (len(remaining_items) / (len(remaining_items) - 1)) * \
                                           (1 - remaining_var / remaining_total_var)
            
            return {
                "success": True,
                "cronbach_alpha": alpha,
                "n_items": n_items,
                "item_variances": item_variances.to_dict(),
                "correlation_matrix": corr_matrix.to_dict(),
                "alpha_if_deleted": alpha_if_deleted,
                "interpretation": self._interpret_cronbach_alpha(alpha)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _interpret_cronbach_alpha(self, alpha: float) -> str:
        """Cronbach's α解釈"""
        if alpha < 0.6:
            return "信頼性が低い (Poor reliability)"
        elif alpha < 0.7:
            return "信頼性がやや低い (Questionable reliability)"
        elif alpha < 0.8:
            return "信頼性が良好 (Good reliability)"
        elif alpha < 0.9:
            return "信頼性が優秀 (Excellent reliability)"
        else:
            return "信頼性が非常に優秀 (Very excellent reliability)"
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Cohen's d計算"""
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Cohen's d解釈"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "小さい効果量 (Small effect)"
        elif abs_d < 0.5:
            return "中程度の効果量 (Medium effect)"
        elif abs_d < 0.8:
            return "大きい効果量 (Large effect)"
        else:
            return "非常に大きい効果量 (Very large effect)"
    
    def _interpret_eta_squared(self, eta_squared: float) -> str:
        """eta squared解釈"""
        if eta_squared < 0.01:
            return "小さい効果量 (Small effect)"
        elif eta_squared < 0.06:
            return "中程度の効果量 (Medium effect)"
        elif eta_squared < 0.14:
            return "大きい効果量 (Large effect)"
        else:
            return "非常に大きい効果量 (Very large effect)" 