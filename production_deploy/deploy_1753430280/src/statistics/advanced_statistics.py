#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Statistics Module
高度な統計分析モジュール

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
License: MIT
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import f_oneway, levene, shapiro, bartlett
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class AdvancedStatsAnalyzer:
    """高度な統計分析クラス"""
    
    def __init__(self):
        """初期化"""
        self.analysis_results = {}
        
    def comprehensive_t_test(self, data: pd.DataFrame, group_col: str, value_col: str, 
                           test_type: str = 'independent', equal_var: bool = True) -> Dict[str, Any]:
        """
        包括的t検定（対応あり・なし、等分散性検定、効果量計算含む）
        
        Args:
            data: データフレーム
            group_col: グループ変数名
            value_col: 測定値変数名
            test_type: 'independent' または 'paired'
            equal_var: 等分散性仮定（対応なしt検定のみ）
        """
        try:
            if test_type == 'independent':
                return self._independent_t_test(data, group_col, value_col, equal_var)
            elif test_type == 'paired':
                return self._paired_t_test(data, group_col, value_col)
            else:
                return {"success": False, "error": "無効な検定タイプ"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _independent_t_test(self, data: pd.DataFrame, group_col: str, value_col: str, 
                          equal_var: bool = True) -> Dict[str, Any]:
        """対応なしt検定"""
        # データ準備
        groups = data.groupby(group_col)[value_col].apply(lambda x: x.dropna()).to_dict()
        group_names = list(groups.keys())
        
        if len(group_names) != 2:
            return {"success": False, "error": "2つのグループが必要です"}
        
        group1 = np.array(groups[group_names[0]])
        group2 = np.array(groups[group_names[1]])
        
        # 記述統計
        desc_stats = {
            group_names[0]: {
                'n': len(group1), 'mean': np.mean(group1), 'std': np.std(group1, ddof=1),
                'var': np.var(group1, ddof=1), 'median': np.median(group1)
            },
            group_names[1]: {
                'n': len(group2), 'mean': np.mean(group2), 'std': np.std(group2, ddof=1),
                'var': np.var(group2, ddof=1), 'median': np.median(group2)
            }
        }
        
        # 等分散性検定（Levene検定）
        levene_stat, levene_p = levene(group1, group2)
        
        # t検定実行
        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
        
        # 効果量計算（Cohen's d）
        cohens_d = self._calculate_cohens_d(group1, group2)
        
        # 信頼区間
        df = len(group1) + len(group2) - 2
        mean_diff = np.mean(group1) - np.mean(group2)
        pooled_se = np.sqrt((np.var(group1, ddof=1)/len(group1) + np.var(group2, ddof=1)/len(group2)))
        ci_lower, ci_upper = stats.t.interval(0.95, df, loc=mean_diff, scale=pooled_se)
        
        return {
            "success": True,
            "test_type": "Independent t-test",
            "descriptive_stats": desc_stats,
            "levene_test": {"statistic": levene_stat, "p_value": levene_p},
            "t_test": {"statistic": t_stat, "p_value": p_value, "df": df},
            "effect_size": {"cohens_d": cohens_d, "interpretation": self._interpret_cohens_d(cohens_d)},
            "confidence_interval": {"lower": ci_lower, "upper": ci_upper, "mean_diff": mean_diff},
            "equal_variance_assumption": equal_var
        }
    
    def _paired_t_test(self, data: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
        """対応ありt検定"""
        # データ準備
        groups = data.groupby(group_col)[value_col].apply(lambda x: x.dropna()).to_dict()
        group_names = list(groups.keys())
        
        if len(group_names) != 2:
            return {"success": False, "error": "2つのグループが必要です"}
        
        group1 = np.array(groups[group_names[0]])
        group2 = np.array(groups[group_names[1]])
        
        if len(group1) != len(group2):
            return {"success": False, "error": "対応データの数が一致しません"}
        
        # 差分計算
        diff = group1 - group2
        
        # 記述統計
        desc_stats = {
            group_names[0]: {
                'n': len(group1), 'mean': np.mean(group1), 'std': np.std(group1, ddof=1)
            },
            group_names[1]: {
                'n': len(group2), 'mean': np.mean(group2), 'std': np.std(group2, ddof=1)
            },
            'difference': {
                'mean': np.mean(diff), 'std': np.std(diff, ddof=1)
            }
        }
        
        # 正規性検定（差分について）
        shapiro_stat, shapiro_p = shapiro(diff)
        
        # 対応ありt検定
        t_stat, p_value = stats.ttest_rel(group1, group2)
        
        # 効果量計算（対応ありの場合）
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        
        # 信頼区間
        df = len(diff) - 1
        ci_lower, ci_upper = stats.t.interval(0.95, df, loc=np.mean(diff), scale=stats.sem(diff))
        
        return {
            "success": True,
            "test_type": "Paired t-test",
            "descriptive_stats": desc_stats,
            "normality_test": {"statistic": shapiro_stat, "p_value": shapiro_p},
            "t_test": {"statistic": t_stat, "p_value": p_value, "df": df},
            "effect_size": {"cohens_d": cohens_d, "interpretation": self._interpret_cohens_d(cohens_d)},
            "confidence_interval": {"lower": ci_lower, "upper": ci_upper, "mean_diff": np.mean(diff)}
        }
    
    def comprehensive_anova(self, data: pd.DataFrame, group_col: str, value_col: str, 
                          post_hoc: bool = True) -> Dict[str, Any]:
        """
        包括的ANOVA（一元配置分散分析）
        
        Args:
            data: データフレーム
            group_col: グループ変数名
            value_col: 測定値変数名
            post_hoc: 事後検定の実行
        """
        try:
            # データ準備
            clean_data = data[[group_col, value_col]].dropna()
            groups = clean_data.groupby(group_col)[value_col].apply(list).to_dict()
            group_names = list(groups.keys())
            
            if len(group_names) < 2:
                return {"success": False, "error": "2つ以上のグループが必要です"}
            
            # 記述統計
            desc_stats = {}
            for name, group_data in groups.items():
                desc_stats[name] = {
                    'n': len(group_data),
                    'mean': np.mean(group_data),
                    'std': np.std(group_data, ddof=1),
                    'var': np.var(group_data, ddof=1),
                    'median': np.median(group_data)
                }
            
            # 等分散性検定（Levene検定）
            levene_stat, levene_p = levene(*groups.values())
            
            # ANOVA実行
            f_stat, p_value = f_oneway(*groups.values())
            
            # 効果量計算（eta squared）
            ss_between = sum([len(groups[name]) * (desc_stats[name]['mean'] - np.mean(clean_data[value_col]))**2 
                             for name in group_names])
            ss_total = sum([(x - np.mean(clean_data[value_col]))**2 for x in clean_data[value_col]])
            eta_squared = ss_between / ss_total
            
            # 自由度
            df_between = len(group_names) - 1
            df_within = len(clean_data) - len(group_names)
            
            result = {
                "success": True,
                "test_type": "One-way ANOVA",
                "descriptive_stats": desc_stats,
                "levene_test": {"statistic": levene_stat, "p_value": levene_p},
                "anova": {
                    "f_statistic": f_stat,
                    "p_value": p_value,
                    "df_between": df_between,
                    "df_within": df_within
                },
                "effect_size": {
                    "eta_squared": eta_squared,
                    "interpretation": self._interpret_eta_squared(eta_squared)
                }
            }
            
            # 事後検定
            if post_hoc and p_value < 0.05 and len(group_names) > 2:
                post_hoc_result = self._run_post_hoc_tests(clean_data, group_col, value_col)
                result["post_hoc"] = post_hoc_result
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _run_post_hoc_tests(self, data: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
        """事後検定の実行"""
        try:
            # Tukey HSD検定
            tukey_result = pairwise_tukeyhsd(data[value_col], data[group_col], alpha=0.05)
            
            # 結果をDataFrameに変換
            tukey_df = pd.DataFrame({
                'group1': tukey_result.groupsunique[tukey_result._multicomp.pairindices[0]],
                'group2': tukey_result.groupsunique[tukey_result._multicomp.pairindices[1]],
                'meandiff': tukey_result.meandiffs,
                'p_adj': tukey_result.pvalues,
                'lower': tukey_result.confint[:, 0],
                'upper': tukey_result.confint[:, 1],
                'reject': tukey_result.reject
            })
            
            return {
                "method": "Tukey HSD",
                "results": tukey_df.to_dict('records'),
                "summary": str(tukey_result)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def multiple_comparison_tests(self, data: pd.DataFrame, group_col: str, value_col: str, 
                                method: str = 'tukey') -> Dict[str, Any]:
        """多重比較検定"""
        try:
            clean_data = data[[group_col, value_col]].dropna()
            
            if method.lower() == 'tukey':
                # Tukey HSD検定
                tukey_results = pairwise_tukeyhsd(
                    clean_data[value_col], 
                    clean_data[group_col], 
                    alpha=0.05
                )
                
                # 結果をDataFrameに変換
                results_df = pd.DataFrame({
                    'group1': tukey_results.groupsunique[tukey_results._multicomp.pairindices[0]],
                    'group2': tukey_results.groupsunique[tukey_results._multicomp.pairindices[1]],
                    'meandiff': tukey_results.meandiffs,
                    'p_adj': tukey_results.pvalues,
                    'lower': tukey_results.confint[:, 0],
                    'upper': tukey_results.confint[:, 1],
                    'reject': tukey_results.reject
                })
                
                return {
                    "success": True,
                    "method": "Tukey HSD",
                    "results": results_df.to_dict('records'),
                    "summary": str(tukey_results)
                }
            
            elif method.lower() == 'bonferroni':
                # Bonferroni補正
                groups = clean_data.groupby(group_col)[value_col]
                group_names = list(groups.groups.keys())
                
                comparisons = []
                for i in range(len(group_names)):
                    for j in range(i+1, len(group_names)):
                        group1_data = groups.get_group(group_names[i])
                        group2_data = groups.get_group(group_names[j])
                        
                        # t検定
                        t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
                        
                        comparisons.append({
                            'group1': group_names[i],
                            'group2': group_names[j],
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'mean_diff': group1_data.mean() - group2_data.mean()
                        })
                
                # Bonferroni補正
                n_comparisons = len(comparisons)
                for comp in comparisons:
                    comp['p_adj_bonferroni'] = min(comp['p_value'] * n_comparisons, 1.0)
                    comp['significant_bonferroni'] = comp['p_adj_bonferroni'] < 0.05
                
                return {
                    "success": True,
                    "method": "Bonferroni",
                    "results": comparisons,
                    "n_comparisons": n_comparisons
                }
            
            else:
                return {"success": False, "error": "無効な多重比較手法"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def sphericity_test(self, data: pd.DataFrame, subject_col: str, time_col: str, value_col: str) -> Dict[str, Any]:
        """
        球面性検定（Mauchly's test）
        
        Args:
            data: データフレーム
            subject_col: 被験者ID列
            time_col: 時間/条件列
            value_col: 測定値列
        """
        try:
            # データをピボット形式に変換
            pivot_data = data.pivot(index=subject_col, columns=time_col, values=value_col)
            pivot_data = pivot_data.dropna()
            
            if pivot_data.shape[1] < 3:
                return {"success": False, "error": "球面性検定には3つ以上の条件が必要です"}
            
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
                "success": True,
                "test_type": "Mauchly's Sphericity Test",
                "statistic": w,
                "chi_square": chi_square,
                "df": df,
                "p_value": p_value,
                "sphericity_assumption": p_value > 0.05
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def effect_size_analysis(self, data: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
        """効果量分析"""
        try:
            if group_col not in data.columns or value_col not in data.columns:
                return {"success": False, "error": "指定された列が見つかりません"}
            
            # データ準備
            clean_data = data[[group_col, value_col]].dropna()
            groups = clean_data.groupby(group_col)[value_col]
            
            results = {}
            
            # 記述統計
            descriptive = groups.agg(['count', 'mean', 'std', 'median']).round(4)
            results['descriptive'] = descriptive
            
            # 2群比較の場合
            if len(groups) == 2:
                group_names = list(groups.groups.keys())
                group1_data = groups.get_group(group_names[0])
                group2_data = groups.get_group(group_names[1])
                
                # Cohen's d
                cohens_d = self._calculate_cohens_d(group1_data, group2_data)
                
                # Glass's delta
                glass_delta = (group1_data.mean() - group2_data.mean()) / group1_data.std()
                
                # Hedge's g
                pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                                     (len(group2_data) - 1) * group2_data.var()) / 
                                    (len(group1_data) + len(group2_data) - 2))
                hedges_g = (group1_data.mean() - group2_data.mean()) / pooled_std
                
                results['effect_sizes'] = {
                    "cohens_d": cohens_d,
                    "glass_delta": glass_delta,
                    "hedges_g": hedges_g,
                    "interpretation_cohens_d": self._interpret_cohens_d(cohens_d)
                }
            
            # 多群比較の場合
            elif len(groups) > 2:
                # eta squared計算
                grand_mean = clean_data[value_col].mean()
                ss_between = sum([len(group_data) * (group_data.mean() - grand_mean)**2 
                                 for group_data in groups])
                ss_total = sum([(x - grand_mean)**2 for x in clean_data[value_col]])
                eta_squared = ss_between / ss_total
                
                results['effect_sizes'] = {
                    "eta_squared": eta_squared,
                    "interpretation": self._interpret_eta_squared(eta_squared)
                }
            
            return {"success": True, "results": results}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
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
    
    def normality_tests(self, data: pd.Series) -> Dict[str, Any]:
        """正規性検定"""
        try:
            clean_data = data.dropna()
            
            # Shapiro-Wilk検定
            shapiro_stat, shapiro_p = shapiro(clean_data)
            
            # Kolmogorov-Smirnov検定
            ks_stat, ks_p = stats.kstest(clean_data, 'norm', 
                                        args=(clean_data.mean(), clean_data.std()))
            
            # Anderson-Darling検定
            ad_stat, ad_critical, ad_significance = stats.anderson(clean_data)
            
            return {
                "success": True,
                "shapiro_wilk": {"statistic": shapiro_stat, "p_value": shapiro_p},
                "kolmogorov_smirnov": {"statistic": ks_stat, "p_value": ks_p},
                "anderson_darling": {"statistic": ad_stat, "critical_values": ad_critical, "significance": ad_significance},
                "normality_assumption": shapiro_p > 0.05
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def homogeneity_of_variance_tests(self, data: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Any]:
        """等分散性検定"""
        try:
            clean_data = data[[group_col, value_col]].dropna()
            groups = clean_data.groupby(group_col)[value_col]
            
            # Levene検定
            levene_stat, levene_p = levene(*groups.apply(list).values)
            
            # Bartlett検定
            bartlett_stat, bartlett_p = bartlett(*groups.apply(list).values)
            
            # Brown-Forsythe検定（中央値ベース）
            bf_stat, bf_p = stats.levene(*groups.apply(list).values, center='median')
            
            return {
                "success": True,
                "levene": {"statistic": levene_stat, "p_value": levene_p},
                "bartlett": {"statistic": bartlett_stat, "p_value": bartlett_p},
                "brown_forsythe": {"statistic": bf_stat, "p_value": bf_p},
                "homogeneity_assumption": levene_p > 0.05
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}