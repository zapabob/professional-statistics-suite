#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Professional Statistics Suite - Advanced GUI
高度な統計分析システムGUI

Author: Professional Statistics Suite Team
Email: r.minegishi1987@gmail.com
License: MIT
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import json
import os
import threading
import queue
import uuid
from datetime import datetime
import sys
import traceback
from typing import Dict, List, Any, Optional, Callable
import warnings
import logging
warnings.filterwarnings('ignore')

# ボタンデバッグログ設定
button_logger = logging.getLogger('button_debug')
button_logger.setLevel(logging.DEBUG)

# ログディレクトリの自動作成
os.makedirs('logs', exist_ok=True)

button_handler = logging.FileHandler('logs/button_debug.log', encoding='utf-8')
button_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
button_logger.addHandler(button_handler)

# 高度なモジュールをインポート
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'statistics'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'visualization'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'security'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gui'))

from ai_integration import AIOrchestrator, QueryProcessor, ContextManager, AnalysisContext
from statistical_method_advisor import StatisticalMethodAdvisor
from assumption_validator import AssumptionValidator
from professional_reports import ReportGenerator
from data_preprocessing import DataPreprocessor
from statistical_power_analysis import PowerAnalysisEngine
from bayesian_analysis import DeepBayesianAnalyzer as BayesianAnalyzer
from survival_analysis import CompleteSurvivalAnalyzer as SurvivalAnalyzer
from advanced_statistics import AdvancedStatsAnalyzer as AdvancedStatistics
from advanced_visualization import AdvancedVisualizer
from audit_compliance_system import AuditTrailManager, ComplianceChecker
from contextual_retriever import ContextualRetriever
from gguf_model_selector import GGUFModelSelector, create_gguf_selector_dialog
from gui_responsiveness_optimizer import GUIResponsivenessOptimizer, ResponsivenessTestSuite

class ProfessionalStatisticsGUI:
    """Professional Statistics Suite - 高度な統計分析GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Professional Statistics Suite - Advanced Analytics")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#2c3e50')
        
        # セッション管理
        self.session_id = f"pss_session_{int(datetime.now().timestamp())}"
        self.backup_dir = "pss_backups"
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # データ管理
        self.data = pd.DataFrame()
        self.analysis_results = {}
        self.results_queue = queue.Queue()
        self.current_analysis = None
        
        # 応答性最適化システム初期化
        self.responsiveness_optimizer = GUIResponsivenessOptimizer(root)
        self.responsiveness_test_suite = ResponsivenessTestSuite(root)
        
        # 高度なコンポーネント初期化
        self.initialize_advanced_components()
        
        # GUI初期化
        self.create_advanced_widgets()
        self.setup_auto_save()
        
        # ボタン統計の読み込み
        self.load_button_statistics()
        
        # 実装ログの読み込み
        self.load_implementation_logs()

    def initialize_advanced_components(self):
        """高度なコンポーネントの初期化"""
        try:
            print("🔧 AI統合コンポーネントを初期化中...")
            # AI統合コンポーネント
            self.ai_orchestrator = AIOrchestrator()
            self.query_processor = QueryProcessor()
            self.context_manager = ContextManager()
            print("✅ AI統合コンポーネント初期化完了")
            
            print("🔧 統計分析コンポーネントを初期化中...")
            # 統計分析コンポーネント
            self.statistical_advisor = StatisticalMethodAdvisor()
            self.assumption_validator = AssumptionValidator()
            self.data_preprocessor = DataPreprocessor()
            self.power_analyzer = PowerAnalysisEngine()
            print("✅ 統計分析コンポーネント初期化完了")
            
            print("🔧 高度な分析コンポーネントを初期化中...")
            # 高度な分析コンポーネント
            self.bayesian_analyzer = BayesianAnalyzer()
            self.survival_analyzer = SurvivalAnalyzer()
            self.advanced_stats = AdvancedStatistics()
            self.advanced_viz = AdvancedVisualizer()
            print("✅ 高度な分析コンポーネント初期化完了")
            
            print("🔧 レポート・監査コンポーネントを初期化中...")
            # レポート・監査コンポーネント
            self.report_generator = ReportGenerator()
            self.audit_manager = AuditTrailManager()
            self.compliance_checker = ComplianceChecker(self.audit_manager)
            print("✅ レポート・監査コンポーネント初期化完了")
            
            print("🔧 コンテキスト検索を初期化中...")
            # コンテキスト検索
            self.contextual_retriever = ContextualRetriever()
            print("✅ コンテキスト検索初期化完了")
            
            print("✅ 高度なコンポーネントの初期化が完了しました")
            
            # 機能チェック
            self.check_component_functionality()
            
        except Exception as e:
            print(f"❌ 高度なコンポーネントの初期化エラー: {e}")
            import traceback
            traceback.print_exc()
            # 基本的な機能のみで動作を継続
            self.ai_orchestrator = None
            self.query_processor = None
            self.context_manager = None
            self.statistical_advisor = None
            self.assumption_validator = None
            self.data_preprocessor = None
            self.power_analyzer = None
            self.bayesian_analyzer = None
            self.survival_analyzer = None
            self.advanced_stats = None
            self.advanced_viz = None
            self.report_generator = None
            self.audit_manager = None
            self.compliance_checker = None
            self.contextual_retriever = None
            print("⚠️ 基本的な機能のみで動作を継続します")

    def check_component_functionality(self):
        """コンポーネントの機能チェック"""
        try:
            print("🔍 コンポーネント機能チェック中...")
            
            # AI統合コンポーネントのチェック
            if self.ai_orchestrator:
                print("✅ AIOrchestrator: 利用可能")
            else:
                print("❌ AIOrchestrator: 利用不可")
                
            if self.query_processor:
                print("✅ QueryProcessor: 利用可能")
            else:
                print("❌ QueryProcessor: 利用不可")
                
            if self.context_manager:
                print("✅ ContextManager: 利用可能")
            else:
                print("❌ ContextManager: 利用不可")
            
            # 統計分析コンポーネントのチェック
            if self.statistical_advisor:
                print("✅ StatisticalMethodAdvisor: 利用可能")
            else:
                print("❌ StatisticalMethodAdvisor: 利用不可")
                
            if self.assumption_validator:
                print("✅ AssumptionValidator: 利用可能")
            else:
                print("❌ AssumptionValidator: 利用不可")
                
            if self.data_preprocessor:
                print("✅ DataPreprocessor: 利用可能")
            else:
                print("❌ DataPreprocessor: 利用不可")
                
            if self.power_analyzer:
                print("✅ PowerAnalysisEngine: 利用可能")
            else:
                print("❌ PowerAnalysisEngine: 利用不可")
            
            # 高度な分析コンポーネントのチェック
            if self.bayesian_analyzer:
                print("✅ BayesianAnalyzer: 利用可能")
            else:
                print("❌ BayesianAnalyzer: 利用不可")
                
            if self.survival_analyzer:
                print("✅ SurvivalAnalyzer: 利用可能")
            else:
                print("❌ SurvivalAnalyzer: 利用不可")
                
            if self.advanced_stats:
                print("✅ AdvancedStatistics: 利用可能")
            else:
                print("❌ AdvancedStatistics: 利用不可")
                
            if self.advanced_viz:
                print("✅ AdvancedVisualizer: 利用可能")
            else:
                print("❌ AdvancedVisualizer: 利用不可")
            
            # レポート・監査コンポーネントのチェック
            if self.report_generator:
                print("✅ ReportGenerator: 利用可能")
            else:
                print("❌ ReportGenerator: 利用不可")
                
            if self.audit_manager:
                print("✅ AuditTrailManager: 利用可能")
            else:
                print("❌ AuditTrailManager: 利用不可")
                
            if self.compliance_checker:
                print("✅ ComplianceChecker: 利用可能")
            else:
                print("❌ ComplianceChecker: 利用不可")
                
            if self.contextual_retriever:
                print("✅ ContextualRetriever: 利用可能")
            else:
                print("❌ ContextualRetriever: 利用不可")
            
            print("🔍 コンポーネント機能チェック完了")
            
        except Exception as e:
            print(f"❌ コンポーネント機能チェックエラー: {e}")

    def create_advanced_widgets(self):
        """高度なGUIウィジェットの作成"""
        # メインフレーム
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ノートブック（タブ）の作成
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 各タブの作成
        self.create_data_management_tab()
        self.create_ai_analysis_tab()
        self.create_advanced_statistics_tab()
        self.create_bayesian_analysis_tab()
        self.create_survival_analysis_tab()
        self.create_power_analysis_tab()
        self.create_advanced_visualization_tab()
        self.create_assumption_validation_tab()
        self.create_machine_learning_tab()
        self.create_reports_tab()
        self.create_audit_compliance_tab()
        self.create_logs_tab()

    def create_data_management_tab(self):
        """データ管理タブ"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="📊 データ管理")
        
        # データ操作フレーム
        data_ops_frame = ttk.LabelFrame(data_frame, text="データ操作", padding=10)
        data_ops_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # ボタン（デバッグログ機能付き）
        load_button = ttk.Button(data_ops_frame, text="CSV読み込み", 
                                command=self.create_debug_button_wrapper("CSV読み込み", self.load_csv_data))
        load_button.pack(side=tk.LEFT, padx=5)
        
        save_button = ttk.Button(data_ops_frame, text="データ保存", 
                                command=self.create_debug_button_wrapper("データ保存", self.save_data))
        save_button.pack(side=tk.LEFT, padx=5)
        
        clear_button = ttk.Button(data_ops_frame, text="データクリア", 
                                 command=self.create_debug_button_wrapper("データクリア", self.clear_data))
        clear_button.pack(side=tk.LEFT, padx=5)
        
        preprocess_button = ttk.Button(data_ops_frame, text="前処理実行", 
                                      command=self.create_debug_button_wrapper("前処理実行", self.run_data_preprocessing))
        preprocess_button.pack(side=tk.LEFT, padx=5)
        
        # データ表示
        data_display_frame = ttk.LabelFrame(data_frame, text="データ表示", padding=10)
        data_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.data_text = scrolledtext.ScrolledText(data_display_frame, height=20)
        self.data_text.pack(fill=tk.BOTH, expand=True)

    def create_ai_analysis_tab(self):
        """AI分析タブ（LLM切り替え機能強化版）"""
        ai_frame = ttk.Frame(self.notebook)
        self.notebook.add(ai_frame, text="🤖 AI分析")
        
        # LLMプロバイダー選択
        provider_frame = ttk.LabelFrame(ai_frame, text="LLMプロバイダー選択", padding=10)
        provider_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # プロバイダー選択コンボボックス
        self.provider_var = tk.StringVar(value="auto")
        provider_combo = ttk.Combobox(provider_frame, textvariable=self.provider_var, 
                                     values=["auto", "google", "ollama", "lmstudio", "koboldcpp"], 
                                     state="readonly", width=15)
        provider_combo.pack(side=tk.LEFT, padx=5)
        ttk.Label(provider_frame, text="プロバイダー:").pack(side=tk.LEFT, padx=5)
        
        # プロバイダー状態表示
        self.provider_status_label = ttk.Label(provider_frame, text="状態: 確認中...")
        self.provider_status_label.pack(side=tk.RIGHT, padx=5)
        
        # プロバイダー状態更新ボタン
        update_status_button = ttk.Button(provider_frame, text="状態更新", 
                                         command=self.create_debug_button_wrapper("状態更新", self.update_provider_status))
        update_status_button.pack(side=tk.RIGHT, padx=5)
        
        # GGUFモデル選択ボタン
        gguf_button = ttk.Button(provider_frame, text="GGUFモデル選択", 
                                 command=self.create_debug_button_wrapper("GGUFモデル選択", self.select_gguf_model))
        gguf_button.pack(side=tk.RIGHT, padx=5)
        
        # クエリ入力
        query_frame = ttk.LabelFrame(ai_frame, text="自然言語クエリ", padding=10)
        query_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.query_entry = ttk.Entry(query_frame, width=80)
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        analyze_button = ttk.Button(query_frame, text="分析実行", 
                                   command=self.create_debug_button_wrapper("分析実行", self.execute_ai_analysis))
        analyze_button.pack(side=tk.RIGHT, padx=5)
        
        # 結果表示
        result_frame = ttk.LabelFrame(ai_frame, text="AI分析結果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.ai_result_text = scrolledtext.ScrolledText(result_frame, height=25)
        self.ai_result_text.pack(fill=tk.BOTH, expand=True)
        
        # 初期状態更新
        self.update_provider_status()

    def create_advanced_statistics_tab(self):
        """高度統計分析タブ"""
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="📈 高度統計")
        
        # 分析タイプ選択
        analysis_frame = ttk.LabelFrame(stats_frame, text="分析タイプ", padding=10)
        analysis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        analysis_types = [
            "記述統計", "相関分析", "回帰分析", "分散分析", 
            "クラスター分析", "因子分析", "時系列分析", "多変量分析"
        ]
        
        def create_analysis_button(analysis_type):
            return lambda: self.run_advanced_analysis(analysis_type)
        
        for i, analysis_type in enumerate(analysis_types):
            row = i // 4
            col = i % 4
            button = ttk.Button(analysis_frame, text=analysis_type, 
                               command=self.create_debug_button_wrapper(analysis_type, create_analysis_button(analysis_type)))
            button.grid(row=row, column=col, padx=5, pady=5)
        
        # 結果表示
        result_frame = ttk.LabelFrame(stats_frame, text="分析結果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.stats_result_text = scrolledtext.ScrolledText(result_frame, height=20)
        self.stats_result_text.pack(fill=tk.BOTH, expand=True)

    def create_bayesian_analysis_tab(self):
        """ベイズ分析タブ"""
        bayes_frame = ttk.Frame(self.notebook)
        self.notebook.add(bayes_frame, text="🔮 ベイズ分析")
        
        # ベイズ分析オプション
        options_frame = ttk.LabelFrame(bayes_frame, text="ベイズ分析オプション", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        bayes_regression_button = ttk.Button(options_frame, text="ベイズ回帰", 
                                            command=self.create_debug_button_wrapper("ベイズ回帰", lambda: self.run_bayesian_analysis("regression")))
        bayes_regression_button.pack(side=tk.LEFT, padx=5)
        
        bayes_classification_button = ttk.Button(options_frame, text="ベイズ分類", 
                                                command=self.create_debug_button_wrapper("ベイズ分類", lambda: self.run_bayesian_analysis("classification")))
        bayes_classification_button.pack(side=tk.LEFT, padx=5)
        
        bayes_test_button = ttk.Button(options_frame, text="ベイズ検定", 
                                       command=self.create_debug_button_wrapper("ベイズ検定", lambda: self.run_bayesian_analysis("test")))
        bayes_test_button.pack(side=tk.LEFT, padx=5)
        
        bayes_estimation_button = ttk.Button(options_frame, text="ベイズ推定", 
                                             command=self.create_debug_button_wrapper("ベイズ推定", lambda: self.run_bayesian_analysis("estimation")))
        bayes_estimation_button.pack(side=tk.LEFT, padx=5)
        
        # 結果表示
        result_frame = ttk.LabelFrame(bayes_frame, text="ベイズ分析結果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.bayes_result_text = scrolledtext.ScrolledText(result_frame, height=25)
        self.bayes_result_text.pack(fill=tk.BOTH, expand=True)

    def create_survival_analysis_tab(self):
        """生存時間分析タブ"""
        survival_frame = ttk.Frame(self.notebook)
        self.notebook.add(survival_frame, text="⏰ 生存時間分析")
        
        # 生存時間分析オプション
        options_frame = ttk.LabelFrame(survival_frame, text="生存時間分析オプション", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        kaplan_meier_button = ttk.Button(options_frame, text="Kaplan-Meier推定", 
                                         command=self.create_debug_button_wrapper("Kaplan-Meier推定", lambda: self.run_survival_analysis("kaplan_meier")))
        kaplan_meier_button.pack(side=tk.LEFT, padx=5)
        
        cox_button = ttk.Button(options_frame, text="Cox比例ハザード", 
                                command=self.create_debug_button_wrapper("Cox比例ハザード", lambda: self.run_survival_analysis("cox")))
        cox_button.pack(side=tk.LEFT, padx=5)
        
        survival_function_button = ttk.Button(options_frame, text="生存関数推定", 
                                             command=self.create_debug_button_wrapper("生存関数推定", lambda: self.run_survival_analysis("survival_function")))
        survival_function_button.pack(side=tk.LEFT, padx=5)
        
        hazard_function_button = ttk.Button(options_frame, text="ハザード関数推定", 
                                           command=self.create_debug_button_wrapper("ハザード関数推定", lambda: self.run_survival_analysis("hazard_function")))
        hazard_function_button.pack(side=tk.LEFT, padx=5)
        
        # 結果表示
        result_frame = ttk.LabelFrame(survival_frame, text="生存時間分析結果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.survival_result_text = scrolledtext.ScrolledText(result_frame, height=25)
        self.survival_result_text.pack(fill=tk.BOTH, expand=True)

    def create_power_analysis_tab(self):
        """統計的検出力分析タブ"""
        power_frame = ttk.Frame(self.notebook)
        self.notebook.add(power_frame, text="⚡ 検出力分析")
        
        # 検出力分析オプション
        options_frame = ttk.LabelFrame(power_frame, text="検出力分析オプション", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        sample_size_button = ttk.Button(options_frame, text="サンプルサイズ計算", 
                                       command=self.create_debug_button_wrapper("サンプルサイズ計算", self.calculate_sample_size))
        sample_size_button.pack(side=tk.LEFT, padx=5)
        
        power_button = ttk.Button(options_frame, text="検出力計算", 
                                 command=self.create_debug_button_wrapper("検出力計算", self.calculate_power))
        power_button.pack(side=tk.LEFT, padx=5)
        
        effect_size_button = ttk.Button(options_frame, text="効果量計算", 
                                       command=self.create_debug_button_wrapper("効果量計算", self.calculate_effect_size))
        effect_size_button.pack(side=tk.LEFT, padx=5)
        
        power_curve_button = ttk.Button(options_frame, text="検出力曲線", 
                                       command=self.create_debug_button_wrapper("検出力曲線", self.plot_power_curve))
        power_curve_button.pack(side=tk.LEFT, padx=5)
        
        # 結果表示
        result_frame = ttk.LabelFrame(power_frame, text="検出力分析結果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.power_result_text = scrolledtext.ScrolledText(result_frame, height=25)
        self.power_result_text.pack(fill=tk.BOTH, expand=True)

    def create_advanced_visualization_tab(self):
        """高度可視化タブ"""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="📊 高度可視化")
        
        # 可視化オプション
        viz_options_frame = ttk.LabelFrame(viz_frame, text="可視化オプション", padding=10)
        viz_options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        viz_types = [
            "ヒストグラム", "散布図", "箱ひげ図", "相関行列", 
            "時系列プロット", "密度プロット", "QQプロット", "残差プロット"
        ]
        
        def create_viz_button(viz_type):
            return lambda: self.create_advanced_visualization(viz_type)
        
        for i, viz_type in enumerate(viz_types):
            row = i // 4
            col = i % 4
            button = ttk.Button(viz_options_frame, text=viz_type, 
                               command=self.create_debug_button_wrapper(viz_type, create_viz_button(viz_type)))
            button.grid(row=row, column=col, padx=5, pady=5)
        
        # グラフ表示エリア
        self.viz_canvas_frame = ttk.Frame(viz_frame)
        self.viz_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_assumption_validation_tab(self):
        """仮定検証タブ"""
        validation_frame = ttk.Frame(self.notebook)
        self.notebook.add(validation_frame, text="🔍 仮定検証")
        
        # 仮定検証オプション
        validation_options_frame = ttk.LabelFrame(validation_frame, text="仮定検証オプション", padding=10)
        validation_options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        normality_button = ttk.Button(validation_options_frame, text="正規性検定", 
                                     command=self.create_debug_button_wrapper("正規性検定", lambda: self.validate_assumptions("normality")))
        normality_button.pack(side=tk.LEFT, padx=5)
        
        homogeneity_button = ttk.Button(validation_options_frame, text="等分散性検定", 
                                       command=self.create_debug_button_wrapper("等分散性検定", lambda: self.validate_assumptions("homogeneity")))
        homogeneity_button.pack(side=tk.LEFT, padx=5)
        
        independence_button = ttk.Button(validation_options_frame, text="独立性検定", 
                                        command=self.create_debug_button_wrapper("独立性検定", lambda: self.validate_assumptions("independence")))
        independence_button.pack(side=tk.LEFT, padx=5)
        
        linearity_button = ttk.Button(validation_options_frame, text="線形性検定", 
                                     command=self.create_debug_button_wrapper("線形性検定", lambda: self.validate_assumptions("linearity")))
        linearity_button.pack(side=tk.LEFT, padx=5)
        
        all_assumptions_button = ttk.Button(validation_options_frame, text="全仮定検証", 
                                           command=self.create_debug_button_wrapper("全仮定検証", lambda: self.validate_assumptions("all")))
        all_assumptions_button.pack(side=tk.LEFT, padx=5)
        
        # 結果表示
        result_frame = ttk.LabelFrame(validation_frame, text="仮定検証結果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.validation_result_text = scrolledtext.ScrolledText(result_frame, height=25)
        self.validation_result_text.pack(fill=tk.BOTH, expand=True)

    def create_machine_learning_tab(self):
        """機械学習タブ"""
        ml_frame = ttk.Frame(self.notebook)
        self.notebook.add(ml_frame, text="🤖 機械学習")
        
        # 機械学習オプション
        options_frame = ttk.LabelFrame(ml_frame, text="機械学習オプション", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        classification_button = ttk.Button(options_frame, text="分類", 
                                          command=self.create_debug_button_wrapper("分類", lambda: self.run_machine_learning("classification")))
        classification_button.pack(side=tk.LEFT, padx=5)
        
        regression_button = ttk.Button(options_frame, text="回帰", 
                                      command=self.create_debug_button_wrapper("回帰", lambda: self.run_machine_learning("regression")))
        regression_button.pack(side=tk.LEFT, padx=5)
        
        clustering_button = ttk.Button(options_frame, text="クラスタリング", 
                                      command=self.create_debug_button_wrapper("クラスタリング", lambda: self.run_machine_learning("clustering")))
        clustering_button.pack(side=tk.LEFT, padx=5)
        
        dimensionality_button = ttk.Button(options_frame, text="次元削減", 
                                          command=self.create_debug_button_wrapper("次元削減", lambda: self.run_machine_learning("dimensionality_reduction")))
        dimensionality_button.pack(side=tk.LEFT, padx=5)
        
        # 結果表示
        result_frame = ttk.LabelFrame(ml_frame, text="機械学習結果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.ml_result_text = scrolledtext.ScrolledText(result_frame, height=25)
        self.ml_result_text.pack(fill=tk.BOTH, expand=True)

    def create_reports_tab(self):
        """レポートタブ"""
        reports_frame = ttk.Frame(self.notebook)
        self.notebook.add(reports_frame, text="📋 レポート")
        
        # レポート生成オプション
        report_options_frame = ttk.LabelFrame(reports_frame, text="レポート生成オプション", padding=10)
        report_options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        comprehensive_report_button = ttk.Button(report_options_frame, text="包括的レポート", 
                                               command=self.create_debug_button_wrapper("包括的レポート", self.generate_comprehensive_report))
        comprehensive_report_button.pack(side=tk.LEFT, padx=5)
        
        ai_report_button = ttk.Button(report_options_frame, text="AI分析レポート", 
                                     command=self.create_debug_button_wrapper("AI分析レポート", self.generate_ai_report))
        ai_report_button.pack(side=tk.LEFT, padx=5)
        
        statistical_report_button = ttk.Button(report_options_frame, text="統計手法レポート", 
                                             command=self.create_debug_button_wrapper("統計手法レポート", self.generate_statistical_report))
        statistical_report_button.pack(side=tk.LEFT, padx=5)
        
        bayesian_report_button = ttk.Button(report_options_frame, text="ベイズ分析レポート", 
                                           command=self.create_debug_button_wrapper("ベイズ分析レポート", self.generate_bayesian_report))
        bayesian_report_button.pack(side=tk.LEFT, padx=5)
        
        survival_report_button = ttk.Button(report_options_frame, text="生存時間分析レポート", 
                                           command=self.create_debug_button_wrapper("生存時間分析レポート", self.generate_survival_report))
        survival_report_button.pack(side=tk.LEFT, padx=5)
        
        # 結果表示
        result_frame = ttk.LabelFrame(reports_frame, text="レポート結果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.report_result_text = scrolledtext.ScrolledText(result_frame, height=25)
        self.report_result_text.pack(fill=tk.BOTH, expand=True)

    def create_audit_compliance_tab(self):
        """監査・コンプライアンスタブ"""
        audit_frame = ttk.Frame(self.notebook)
        self.notebook.add(audit_frame, text="🛡️ 監査・コンプライアンス")
        
        # 監査・コンプライアンスオプション
        audit_options_frame = ttk.LabelFrame(audit_frame, text="監査・コンプライアンスオプション", padding=10)
        audit_options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        audit_logs_button = ttk.Button(audit_options_frame, text="監査ログ表示", 
                                      command=self.create_debug_button_wrapper("監査ログ表示", self.show_audit_logs))
        audit_logs_button.pack(side=tk.LEFT, padx=5)
        
        compliance_check_button = ttk.Button(audit_options_frame, text="コンプライアンスチェック", 
                                            command=self.create_debug_button_wrapper("コンプライアンスチェック", self.run_compliance_check))
        compliance_check_button.pack(side=tk.LEFT, padx=5)
        
        privacy_audit_button = ttk.Button(audit_options_frame, text="データプライバシー監査", 
                                          command=self.create_debug_button_wrapper("データプライバシー監査", self.run_privacy_audit))
        privacy_audit_button.pack(side=tk.LEFT, padx=5)
        
        security_audit_button = ttk.Button(audit_options_frame, text="セキュリティ監査", 
                                          command=self.create_debug_button_wrapper("セキュリティ監査", self.run_security_audit))
        security_audit_button.pack(side=tk.LEFT, padx=5)
        
        # 結果表示
        result_frame = ttk.LabelFrame(audit_frame, text="監査・コンプライアンス結果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.audit_result_text = scrolledtext.ScrolledText(result_frame, height=25)
        self.audit_result_text.pack(fill=tk.BOTH, expand=True)

    def create_logs_tab(self):
        """ログタブ"""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="📝 ログ")
        
        # ログ表示
        logs_display_frame = ttk.LabelFrame(logs_frame, text="システムログ", padding=10)
        logs_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.logs_text = scrolledtext.ScrolledText(logs_display_frame, height=30)
        self.logs_text.pack(fill=tk.BOTH, expand=True)
        
        # ログ更新ボタン
        update_logs_button = ttk.Button(logs_frame, text="ログ更新", 
                                       command=self.create_debug_button_wrapper("ログ更新", self.update_logs))
        update_logs_button.pack(pady=5)
        
        # ボタン統計表示ボタン
        button_stats_button = ttk.Button(logs_frame, text="ボタン統計表示", 
                                        command=self.show_button_statistics)
        button_stats_button.pack(pady=5)

    # データ管理メソッド
    def load_csv_data(self):
        """CSVデータの読み込み"""
        try:
            file_path = filedialog.askopenfilename(
                title="CSVファイルを選択",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if file_path:
                self.data = pd.read_csv(file_path)
                self.update_data_display()
                self.log_message(f"CSVファイルを読み込みました: {file_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"CSVファイルの読み込みに失敗しました: {e}")

    def update_data_display(self):
        """データ表示の更新"""
        if not self.data.empty:
            info_text = f"データ形状: {self.data.shape}\n"
            info_text += f"列名: {list(self.data.columns)}\n"
            info_text += f"データ型:\n{self.data.dtypes}\n"
            info_text += f"基本統計:\n{self.data.describe()}\n"
            info_text += f"欠損値:\n{self.data.isnull().sum()}\n"
            
            self.data_text.delete(1.0, tk.END)
            self.data_text.insert(tk.END, info_text)

    def clear_data(self):
        """データのクリア"""
        self.data = pd.DataFrame()
        self.data_text.delete(1.0, tk.END)
        self.log_message("データをクリアしました")

    def save_data(self):
        """データの保存"""
        try:
            file_path = filedialog.asksaveasfilename(
                title="データを保存",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if file_path:
                self.data.to_csv(file_path, index=False)
                self.log_message(f"データを保存しました: {file_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"データの保存に失敗しました: {e}")

    def run_data_preprocessing(self):
        """データ前処理の実行"""
        if self.data.empty:
            messagebox.showwarning("警告", "データが読み込まれていません")
            return
        
        try:
            # 前処理の実行
            processed_data, preprocessing_info = self.data_preprocessor.handle_missing_values(self.data)
            processed_data, outlier_info = self.data_preprocessor.detect_outliers(processed_data)
            
            # 結果表示
            result_text = "データ前処理結果:\n\n"
            result_text += f"前処理情報: {preprocessing_info}\n\n"
            result_text += f"外れ値検出結果: {outlier_info}\n"
            
            self.data_text.delete(1.0, tk.END)
            self.data_text.insert(tk.END, result_text)
            
            self.log_message("データ前処理を実行しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"データ前処理に失敗しました: {e}")

    # AI分析メソッド
    def update_provider_status(self):
        """プロバイダー状態の更新"""
        try:
            if self.ai_orchestrator is None:
                self.provider_status_label.config(text="状態: AI統合機能なし")
                return
            
            # 利用可能なプロバイダーを確認
            available_providers = []
            if hasattr(self.ai_orchestrator, 'providers') and self.ai_orchestrator.providers:
                for name, provider in self.ai_orchestrator.providers.items():
                    if hasattr(provider, 'is_available') and provider.is_available():
                        available_providers.append(name)
            
            if available_providers:
                status_text = f"状態: 利用可能 ({', '.join(available_providers)})"
                self.provider_status_label.config(text=status_text)
            else:
                self.provider_status_label.config(text="状態: 利用可能なプロバイダーなし")
                
        except Exception as e:
            self.provider_status_label.config(text=f"状態: エラー ({str(e)})")
            self.log_message(f"プロバイダー状態更新エラー: {e}")
    
    def select_gguf_model(self):
        """GGUFモデル選択ダイアログを表示"""
        try:
            selected_path = create_gguf_selector_dialog(self.root, "GGUFモデル選択")
            
            if selected_path:
                # 選択されたモデルをLMStudioプロバイダーに追加
                if hasattr(self.ai_orchestrator, 'providers') and 'lmstudio' in self.ai_orchestrator.providers:
                    lmstudio_provider = self.ai_orchestrator.providers['lmstudio']
                    if hasattr(lmstudio_provider, 'scan_custom_directory'):
                        # モデルファイルのディレクトリをスキャン
                        model_dir = os.path.dirname(selected_path)
                        custom_models = lmstudio_provider.scan_custom_directory(model_dir)
                        
                        if custom_models:
                            messagebox.showinfo("成功", f"GGUFモデルを追加しました: {os.path.basename(selected_path)}")
                            # プロバイダー状態を更新
                            self.update_provider_status()
                        else:
                            messagebox.showwarning("警告", "GGUFモデルの追加に失敗しました")
                    else:
                        messagebox.showwarning("警告", "LMStudioプロバイダーが利用できません")
                else:
                    messagebox.showwarning("警告", "LMStudioプロバイダーが設定されていません")
                    
        except Exception as e:
            messagebox.showerror("エラー", f"GGUFモデル選択中にエラーが発生しました: {str(e)}")
            print(f"GGUFモデル選択エラー: {e}")
    
    def execute_ai_analysis(self):
        """AI分析の実行（LLM切り替え機能強化版）"""
        if self.ai_orchestrator is None:
            messagebox.showwarning("警告", "AI分析機能が利用できません")
            return
            
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showwarning("警告", "クエリを入力してください")
            return
        
        if self.data.empty:
            messagebox.showwarning("警告", "データが読み込まれていません")
            return
        
        # 選択されたプロバイダーを取得
        selected_provider = self.provider_var.get()
        if selected_provider == "auto":
            selected_provider = None  # 自動選択
        
        # 非同期でAI分析を実行
        threading.Thread(target=self._execute_ai_analysis_async, args=(query, selected_provider), daemon=True).start()

    def _execute_ai_analysis_async(self, query, selected_provider=None):
        """非同期AI分析（LLM切り替え機能強化版）"""
        try:
            # 分析コンテキストを作成
            context = AnalysisContext(
                user_id="gui_user",
                session_id=str(uuid.uuid4()),
                data_fingerprint=hash(str(self.data.shape)),
                analysis_history=[]
            )
            
            # プロバイダー設定をコンテキストに追加
            if selected_provider and selected_provider != "auto":
                context.privacy_settings = {
                    "use_local_llm": selected_provider in ["ollama", "lmstudio", "koboldcpp"],
                    "preferred_provider": selected_provider
                }
            
            # AI分析の実行（非同期）
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # エラーハンドリング強化
            try:
                result = loop.run_until_complete(
                    self.ai_orchestrator.process_user_query(query, context, data=self.data)
                )
                
                # 結果をメインスレッドで表示
                self.root.after(0, self._display_ai_result, result)
                
            except AttributeError as e:
                # analyze_queryメソッドが見つからない場合のフォールバック
                if "analyze_query" in str(e):
                    self.logger.warning("analyze_queryメソッドが見つかりません。process_user_queryを使用します。")
                    result = loop.run_until_complete(
                        self.ai_orchestrator.process_user_query(query, context, data=self.data)
                    )
                    self.root.after(0, self._display_ai_result, result)
                else:
                    raise
                    
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"AI分析エラー: {error_msg}")
            
            # エラーメッセージの改善
            if "API key not valid" in error_msg:
                error_msg = "Google API keyが無効です。ローカルLLMの使用を推奨します。"
            elif "analyze_query" in error_msg:
                error_msg = "AI統合エラーが発生しました。システムを再起動してください。"
            elif "timeout" in error_msg.lower():
                error_msg = "処理がタイムアウトしました。データサイズを確認してください。"
            elif "provider" in error_msg.lower():
                error_msg = f"プロバイダー '{selected_provider}' が利用できません。別のプロバイダーを選択してください。"
            
            self.root.after(0, self._display_ai_error, error_msg)

    def _display_ai_result(self, result):
        """AI分析結果の表示"""
        self.ai_result_text.delete(1.0, tk.END)
        self.ai_result_text.insert(tk.END, str(result))
        self.log_message("AI分析を実行しました")

    def _display_ai_error(self, error_msg):
        """AI分析エラーの表示"""
        messagebox.showerror("AI分析エラー", error_msg)
        self.log_message(f"AI分析エラー: {error_msg}")

    # 高度統計分析メソッド
    def run_advanced_analysis(self, analysis_type):
        """高度統計分析の実行（エラーハンドリング強化版）"""
        if self.advanced_stats is None:
            messagebox.showwarning("警告", "高度統計分析機能が利用できません")
            return
            
        if self.data.empty:
            messagebox.showwarning("警告", "データが読み込まれていません")
            return
        
        try:
            # 分析タイプに応じたメソッドを呼び出し
            if analysis_type == "記述統計":
                result = self.advanced_stats.descriptive_statistics(self.data)
            elif analysis_type == "相関分析":
                result = self.advanced_stats.correlation_analysis(self.data)
            elif analysis_type == "回帰分析":
                result = self.advanced_stats.regression_analysis(self.data)
            elif analysis_type == "分散分析":
                result = self.advanced_stats.anova_analysis(self.data)
            elif analysis_type == "クラスター分析":
                result = self.advanced_stats.clustering_analysis(self.data)
            elif analysis_type == "因子分析":
                result = self.advanced_stats.factor_analysis(self.data)
            elif analysis_type == "時系列分析":
                # 日付列と値列を自動検出
                date_cols = self.data.select_dtypes(include=['datetime64']).columns
                if len(date_cols) > 0:
                    date_col = date_cols[0]
                    numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        value_col = numeric_cols[0]
                        result = self.advanced_stats.time_series_analysis(self.data, date_col, value_col)
                    else:
                        result = {"error": "数値列が見つかりません"}
                else:
                    result = {"error": "日付列が見つかりません"}
            elif analysis_type == "多変量分析":
                result = self.advanced_stats.multivariate_analysis(self.data)
            elif analysis_type == "包括的EDA":
                result = self.advanced_stats.comprehensive_eda(self.data)
            else:
                # その他の分析タイプは包括的EDAで対応
                result = self.advanced_stats.comprehensive_eda(self.data)
            
            # 結果の表示
            self.stats_result_text.delete(1.0, tk.END)
            if isinstance(result, dict):
                # 辞書形式の場合は整形して表示
                formatted_result = json.dumps(result, ensure_ascii=False, indent=2)
                self.stats_result_text.insert(tk.END, formatted_result)
            else:
                self.stats_result_text.insert(tk.END, str(result))
            
            self.log_message(f"高度統計分析を実行しました: {analysis_type}")
            
        except AttributeError as e:
            error_msg = f"分析メソッド '{analysis_type}' が見つかりません: {e}"
            messagebox.showerror("エラー", error_msg)
            self.log_message(f"高度統計分析エラー: {error_msg}")
        except Exception as e:
            error_msg = f"高度統計分析に失敗しました: {e}"
            messagebox.showerror("エラー", error_msg)
            self.log_message(f"高度統計分析エラー: {error_msg}")

    # ベイズ分析メソッド
    def run_bayesian_analysis(self, analysis_type):
        """ベイズ分析の実行"""
        if self.bayesian_analyzer is None:
            messagebox.showwarning("警告", "ベイズ分析機能が利用できません")
            return
            
        if self.data.empty:
            messagebox.showwarning("警告", "データが読み込まれていません")
            return
        
        try:
            # 数値列とターゲット列を自動検出
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                result = {"error": "分析に必要な数値列が不足しています（最低2列必要）"}
            else:
                target_col = numeric_cols[0]
                predictor_cols = numeric_cols[1:min(5, len(numeric_cols))]  # 最大4つの予測変数
                
                # 分析タイプに応じたメソッドを呼び出し
                if analysis_type == "regression":
                    result = self.bayesian_analyzer.bayesian_linear_regression(
                        self.data, target_col, predictor_cols
                    )
                elif analysis_type == "classification":
                    result = self.bayesian_analyzer.bayesian_logistic_regression(
                        self.data, target_col, predictor_cols
                    )
                elif analysis_type == "test":
                    # ベイズ検定（2群比較）
                    if len(numeric_cols) >= 2:
                        result = self.bayesian_analyzer.bayesian_linear_regression(
                            self.data, numeric_cols[0], [numeric_cols[1]]
                        )
                    else:
                        result = {"error": "ベイズ検定には最低2つの数値列が必要です"}
                elif analysis_type == "estimation":
                    result = self.bayesian_analyzer.bayesian_linear_regression(
                        self.data, target_col, predictor_cols
                    )
                else:
                    result = {"error": f"未対応の分析タイプ: {analysis_type}"}
            
            self.bayes_result_text.delete(1.0, tk.END)
            self.bayes_result_text.insert(tk.END, str(result))
            
            self.log_message(f"ベイズ分析を実行しました: {analysis_type}")
            
        except Exception as e:
            messagebox.showerror("エラー", f"ベイズ分析に失敗しました: {e}")
            self.log_message(f"ベイズ分析エラー: {e}")

    # 生存時間分析メソッド
    def run_survival_analysis(self, analysis_type):
        """生存時間分析の実行"""
        if self.survival_analyzer is None:
            messagebox.showwarning("警告", "生存時間分析機能が利用できません")
            return
            
        if self.data.empty:
            messagebox.showwarning("警告", "データが読み込まれていません")
            return
        
        try:
            # 数値列を自動検出
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                result = {"error": "生存時間分析には最低2つの数値列が必要です（時間とイベント）"}
            else:
                duration_col = numeric_cols[0]  # 最初の数値列を時間として使用
                event_col = numeric_cols[1]     # 2番目の数値列をイベントとして使用
                
                # 分析タイプに応じたメソッドを呼び出し
                if analysis_type == "kaplan_meier":
                    result = self.survival_analyzer.kaplan_meier_analysis(
                        self.data, duration_col, event_col
                    )
                elif analysis_type == "cox":
                    # Cox比例ハザード（共変量がある場合）
                    if len(numeric_cols) > 2:
                        covariate_cols = numeric_cols[2:min(5, len(numeric_cols))]
                        result = self.survival_analyzer.cox_regression_analysis(
                            self.data, duration_col, event_col, covariate_cols
                        )
                    else:
                        result = {"error": "Cox比例ハザードには共変量が必要です"}
                elif analysis_type == "survival_function":
                    result = self.survival_analyzer.kaplan_meier_analysis(
                        self.data, duration_col, event_col
                    )
                elif analysis_type == "hazard_function":
                    result = self.survival_analyzer.parametric_survival_analysis(
                        self.data, duration_col, event_col
                    )
                else:
                    result = {"error": f"未対応の分析タイプ: {analysis_type}"}
            
            self.survival_result_text.delete(1.0, tk.END)
            self.survival_result_text.insert(tk.END, str(result))
            
            self.log_message(f"生存時間分析を実行しました: {analysis_type}")
            
        except Exception as e:
            messagebox.showerror("エラー", f"生存時間分析に失敗しました: {e}")
            self.log_message(f"生存時間分析エラー: {e}")

    # 機械学習メソッド
    def run_machine_learning(self, ml_type):
        """機械学習の実行"""
        if self.data.empty:
            messagebox.showwarning("警告", "データが読み込まれていません")
            return
        
        try:
            # 数値列を自動検出
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                result = {"error": "機械学習には最低2つの数値列が必要です"}
            else:
                # 基本的な機械学習分析
                if ml_type == "classification":
                    result = self._run_classification_analysis()
                elif ml_type == "regression":
                    result = self._run_regression_analysis()
                elif ml_type == "clustering":
                    result = self._run_clustering_analysis()
                elif ml_type == "dimensionality_reduction":
                    result = self._run_dimensionality_reduction()
                else:
                    result = {"error": f"未対応の機械学習タイプ: {ml_type}"}
            
            self.ml_result_text.delete(1.0, tk.END)
            self.ml_result_text.insert(tk.END, str(result))
            
            self.log_message(f"機械学習を実行しました: {ml_type}")
            
        except Exception as e:
            messagebox.showerror("エラー", f"機械学習に失敗しました: {e}")
            self.log_message(f"機械学習エラー: {e}")

    def _run_classification_analysis(self):
        """分類分析の実行"""
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, classification_report
            
            numeric_data = self.data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) < 2:
                return {"error": "分類には最低2つの数値列が必要です"}
            
            # 最初の列をターゲット、残りを特徴量として使用
            X = numeric_data.iloc[:, 1:].values
            y = numeric_data.iloc[:, 0].values
            
            # データ分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # ランダムフォレスト分類
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            
            # 予測
            y_pred = clf.predict(X_test)
            
            # 評価
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            return {
                "method": "Random Forest Classification",
                "accuracy": accuracy,
                "classification_report": report,
                "feature_importance": dict(zip(numeric_data.columns[1:], clf.feature_importances_))
            }
        except Exception as e:
            return {"error": f"分類分析エラー: {e}"}

    def _run_regression_analysis(self):
        """回帰分析の実行"""
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import r2_score, mean_squared_error
            
            numeric_data = self.data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) < 2:
                return {"error": "回帰には最低2つの数値列が必要です"}
            
            # 最初の列をターゲット、残りを特徴量として使用
            X = numeric_data.iloc[:, 1:].values
            y = numeric_data.iloc[:, 0].values
            
            # データ分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # ランダムフォレスト回帰
            reg = RandomForestRegressor(n_estimators=100, random_state=42)
            reg.fit(X_train, y_train)
            
            # 予測
            y_pred = reg.predict(X_test)
            
            # 評価
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            return {
                "method": "Random Forest Regression",
                "r2_score": r2,
                "mean_squared_error": mse,
                "feature_importance": dict(zip(numeric_data.columns[1:], reg.feature_importances_))
            }
        except Exception as e:
            return {"error": f"回帰分析エラー: {e}"}

    def _run_clustering_analysis(self):
        """クラスタリング分析の実行"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            numeric_data = self.data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) < 2:
                return {"error": "クラスタリングには最低2つの数値列が必要です"}
            
            # データ標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(numeric_data.values)
            
            # K-meansクラスタリング
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # クラスタリング結果
            self.data['cluster'] = clusters
            
            return {
                "method": "K-means Clustering",
                "n_clusters": 3,
                "cluster_sizes": [int((clusters == i).sum()) for i in range(3)],
                "inertia": kmeans.inertia_,
                "cluster_centers": kmeans.cluster_centers_.tolist()
            }
        except Exception as e:
            return {"error": f"クラスタリング分析エラー: {e}"}

    def _run_dimensionality_reduction(self):
        """次元削減分析の実行"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            numeric_data = self.data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) < 2:
                return {"error": "次元削減には最低2つの数値列が必要です"}
            
            # データ標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(numeric_data.values)
            
            # PCA
            pca = PCA()
            pca_result = pca.fit_transform(X_scaled)
            
            return {
                "method": "Principal Component Analysis",
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
                "n_components": len(pca.explained_variance_ratio_),
                "components": pca.components_.tolist()
            }
        except Exception as e:
            return {"error": f"次元削減分析エラー: {e}"}

    # 検出力分析メソッド
    def calculate_sample_size(self):
        """サンプルサイズ計算"""
        if self.power_analyzer is None:
            messagebox.showwarning("警告", "検出力分析機能が利用できません")
            return
            
        try:
            # サンプルサイズ計算の実装
            result = self.power_analyzer.calculate_sample_size()
            
            self.power_result_text.delete(1.0, tk.END)
            self.power_result_text.insert(tk.END, str(result))
            
            self.log_message("サンプルサイズ計算を実行しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"サンプルサイズ計算に失敗しました: {e}")

    def calculate_power(self):
        """検出力計算"""
        try:
            result = self.power_analyzer.calculate_power()
            
            self.power_result_text.delete(1.0, tk.END)
            self.power_result_text.insert(tk.END, str(result))
            
            self.log_message("検出力計算を実行しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"検出力計算に失敗しました: {e}")

    def calculate_effect_size(self):
        """効果量計算"""
        try:
            result = self.power_analyzer.calculate_effect_size()
            
            self.power_result_text.delete(1.0, tk.END)
            self.power_result_text.insert(tk.END, str(result))
            
            self.log_message("効果量計算を実行しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"効果量計算に失敗しました: {e}")

    def plot_power_curve(self):
        """検出力曲線のプロット"""
        try:
            fig = self.power_analyzer.plot_power_curve()
            
            # グラフ表示
            canvas = FigureCanvasTkAgg(fig, self.viz_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.log_message("検出力曲線をプロットしました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"検出力曲線のプロットに失敗しました: {e}")

    # 高度可視化メソッド
    def create_advanced_visualization(self, viz_type):
        """高度可視化の作成"""
        if self.data.empty:
            messagebox.showwarning("警告", "データが読み込まれていません")
            return
        
        try:
            fig = self.advanced_viz.create_visualization(self.data, viz_type)
            
            # 既存のキャンバスをクリア
            for widget in self.viz_canvas_frame.winfo_children():
                widget.destroy()
            
            # 新しいグラフを表示
            canvas = FigureCanvasTkAgg(fig, self.viz_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.log_message(f"高度可視化を作成しました: {viz_type}")
            
        except Exception as e:
            messagebox.showerror("エラー", f"高度可視化の作成に失敗しました: {e}")

    # 仮定検証メソッド
    def validate_assumptions(self, assumption_type):
        """仮定検証の実行"""
        if self.data.empty:
            messagebox.showwarning("警告", "データが読み込まれていません")
            return
        
        try:
            result = self.assumption_validator.validate_assumptions(self.data, assumption_type)
            
            self.validation_result_text.delete(1.0, tk.END)
            self.validation_result_text.insert(tk.END, str(result))
            
            self.log_message(f"仮定検証を実行しました: {assumption_type}")
            
        except Exception as e:
            messagebox.showerror("エラー", f"仮定検証に失敗しました: {e}")

    # レポート生成メソッド
    def generate_comprehensive_report(self):
        """包括的レポートの生成"""
        try:
            report = self.report_generator.generate_comprehensive_report(self.data, self.analysis_results)
            
            self.report_result_text.delete(1.0, tk.END)
            self.report_result_text.insert(tk.END, report)
            
            self.log_message("包括的レポートを生成しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"包括的レポートの生成に失敗しました: {e}")

    def generate_ai_report(self):
        """AI分析レポートの生成"""
        try:
            report = self.report_generator.generate_ai_report(self.data, self.analysis_results)
            
            self.report_result_text.delete(1.0, tk.END)
            self.report_result_text.insert(tk.END, report)
            
            self.log_message("AI分析レポートを生成しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"AI分析レポートの生成に失敗しました: {e}")

    def generate_statistical_report(self):
        """統計手法レポートの生成"""
        try:
            report = self.report_generator.generate_statistical_report(self.data, self.analysis_results)
            
            self.report_result_text.delete(1.0, tk.END)
            self.report_result_text.insert(tk.END, report)
            
            self.log_message("統計手法レポートを生成しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"統計手法レポートの生成に失敗しました: {e}")

    def generate_bayesian_report(self):
        """ベイズ分析レポートの生成"""
        try:
            report = self.report_generator.generate_bayesian_report(self.data, self.analysis_results)
            
            self.report_result_text.delete(1.0, tk.END)
            self.report_result_text.insert(tk.END, report)
            
            self.log_message("ベイズ分析レポートを生成しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"ベイズ分析レポートの生成に失敗しました: {e}")

    def generate_survival_report(self):
        """生存時間分析レポートの生成"""
        try:
            report = self.report_generator.generate_survival_report(self.data, self.analysis_results)
            
            self.report_result_text.delete(1.0, tk.END)
            self.report_result_text.insert(tk.END, report)
            
            self.log_message("生存時間分析レポートを生成しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"生存時間分析レポートの生成に失敗しました: {e}")

    # 監査・コンプライアンスメソッド
    def show_audit_logs(self):
        """監査ログの表示"""
        try:
            logs = self.audit_manager.get_audit_logs()
            
            self.audit_result_text.delete(1.0, tk.END)
            self.audit_result_text.insert(tk.END, logs)
            
            self.log_message("監査ログを表示しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"監査ログの表示に失敗しました: {e}")

    def run_compliance_check(self):
        """コンプライアンスチェックの実行"""
        try:
            result = self.compliance_checker.check_compliance()
            
            self.audit_result_text.delete(1.0, tk.END)
            self.audit_result_text.insert(tk.END, str(result))
            
            self.log_message("コンプライアンスチェックを実行しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"コンプライアンスチェックに失敗しました: {e}")

    def run_privacy_audit(self):
        """データプライバシー監査の実行"""
        try:
            result = self.audit_manager.run_privacy_audit(self.data)
            
            self.audit_result_text.delete(1.0, tk.END)
            self.audit_result_text.insert(tk.END, str(result))
            
            self.log_message("データプライバシー監査を実行しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"データプライバシー監査に失敗しました: {e}")

    def run_security_audit(self):
        """セキュリティ監査の実行"""
        try:
            result = self.audit_manager.run_security_audit()
            
            self.audit_result_text.delete(1.0, tk.END)
            self.audit_result_text.insert(tk.END, str(result))
            
            self.log_message("セキュリティ監査を実行しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"セキュリティ監査に失敗しました: {e}")

    # ログ管理メソッド
    def update_logs(self):
        """ログの更新"""
        try:
            log_content = self.load_implementation_logs()
            if log_content:
                self.logs_text.delete(1.0, tk.END)
                self.logs_text.insert(tk.END, log_content)
            
            self.log_message("ログを更新しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"ログの更新に失敗しました: {e}")
    
    def show_button_statistics(self):
        """ボタン統計を表示（強化版）"""
        try:
            if not hasattr(self, 'button_stats') or not self.button_stats:
                messagebox.showinfo("情報", "まだボタンがクリックされていません")
                return
            
            # 統計情報を収集
            total_clicks = sum(stats['clicks'] for stats in self.button_stats.values())
            total_success = sum(stats['success'] for stats in self.button_stats.values())
            total_failed = sum(stats['failed'] for stats in self.button_stats.values())
            overall_success_rate = (total_success / total_clicks * 100) if total_clicks > 0 else 0
            
            # 統計テキストを生成
            stats_text = "=== ボタン統計レポート ===\n"
            stats_text += f"📊 総クリック回数: {total_clicks}\n"
            stats_text += f"✅ 総成功回数: {total_success}\n"
            stats_text += f"❌ 総失敗回数: {total_failed}\n"
            stats_text += f"📈 全体成功率: {overall_success_rate:.1f}%\n\n"
            stats_text += "=== 個別ボタン統計 ===\n\n"
            
            # 成功率でソート
            sorted_stats = sorted(
                self.button_stats.items(),
                key=lambda x: (x[1]['success'] / x[1]['clicks'] if x[1]['clicks'] > 0 else 0),
                reverse=True
            )
            
            for button_name, stats in sorted_stats:
                success_rate = (stats['success'] / stats['clicks'] * 100) if stats['clicks'] > 0 else 0
                status_icon = "🟢" if success_rate >= 80 else "🟡" if success_rate >= 50 else "🔴"
                
                stats_text += f"{status_icon} {button_name}:\n"
                stats_text += f"   クリック回数: {stats['clicks']}\n"
                stats_text += f"   成功回数: {stats['success']}\n"
                stats_text += f"   失敗回数: {stats['failed']}\n"
                stats_text += f"   成功率: {success_rate:.1f}%\n\n"
            
            # 統計をログテキストに表示
            self.logs_text.delete(1.0, tk.END)
            self.logs_text.insert(tk.END, stats_text)
            
            self.log_message("ボタン統計レポートを表示しました")
            
        except Exception as e:
            error_msg = f"ボタン統計の表示に失敗しました: {e}"
            messagebox.showerror("エラー", error_msg)
            self.log_message(error_msg)

    def log_message(self, message):
        """メッセージのログ記録"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        print(log_entry.strip())
    
    def log_button_click(self, button_name: str, function_name: str, success: bool = True, error_msg: str = None):
        """ボタンクリックログを記録（強化版）"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        status = "✅ SUCCESS" if success else "❌ FAILED"
        log_entry = f"[{timestamp}] 🖱️ BUTTON_CLICK: {button_name} -> {function_name} - {status}"
        
        if error_msg:
            log_entry += f" | ERROR: {error_msg}"
        
        # コンソールに出力
        print(log_entry)
        
        # ボタンデバッグログファイルに記録
        button_logger.info(log_entry)
        
        # GUIログにも記録
        if hasattr(self, 'logs_text'):
            self.logs_text.insert(tk.END, log_entry + "\n")
            self.logs_text.see(tk.END)
        
        # ボタン統計を更新
        if not hasattr(self, 'button_stats'):
            self.button_stats = {}
        
        if button_name not in self.button_stats:
            self.button_stats[button_name] = {'clicks': 0, 'success': 0, 'failed': 0, 'last_click': None}
        
        self.button_stats[button_name]['clicks'] += 1
        self.button_stats[button_name]['last_click'] = timestamp
        
        if success:
            self.button_stats[button_name]['success'] += 1
        else:
            self.button_stats[button_name]['failed'] += 1
        
        # 統計の自動保存
        self.save_button_statistics()
    
    def save_button_statistics(self):
        """ボタン統計を自動保存"""
        try:
            if hasattr(self, 'button_stats'):
                stats_file = os.path.join('logs', 'button_statistics.json')
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(self.button_stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"ボタン統計保存エラー: {e}")
    
    def load_button_statistics(self):
        """ボタン統計を読み込み"""
        try:
            stats_file = os.path.join('logs', 'button_statistics.json')
            if os.path.exists(stats_file):
                with open(stats_file, 'r', encoding='utf-8') as f:
                    self.button_stats = json.load(f)
            else:
                self.button_stats = {}
        except Exception as e:
            print(f"ボタン統計読み込みエラー: {e}")
            self.button_stats = {}
    
    def create_unified_button_wrapper(self, button_name: str, original_function: Callable):
        """統一されたボタンラッパー（デバッグログ + 応答性最適化）"""
        def unified_wrapper(*args, **kwargs):
            try:
                # ボタンクリックログを記録
                self.log_button_click(button_name, original_function.__name__, True)
                
                # 元の関数を実行
                result = original_function(*args, **kwargs)
                
                # 成功ログを記録
                self.log_message(f"ボタン '{button_name}' が正常に実行されました")
                
                return result
                
            except Exception as e:
                error_msg = str(e)
                # エラーログを記録
                self.log_button_click(button_name, original_function.__name__, False, error_msg)
                
                # ユーザーフレンドリーなエラーメッセージを表示
                error_display_msg = f"ボタン '{button_name}' の実行中にエラーが発生しました:\n{error_msg}"
                messagebox.showerror("エラー", error_display_msg)
                
                # コンソールにも詳細を出力
                print(f"詳細エラー情報: {traceback.format_exc()}")
                
                # エラーを再発生させない（GUIがフリーズしないように）
                return None
        
        return unified_wrapper

    def create_debug_button_wrapper(self, button_name: str, original_function: Callable):
        """デバッグ用ボタンラッパー（後方互換性のため）"""
        return self.create_unified_button_wrapper(button_name, original_function)

    def optimize_button_responsiveness(self, button: tk.Widget, original_command: Callable):
        """ボタン応答性最適化（後方互換性のため）"""
        return self.create_unified_button_wrapper("応答性最適化", original_command)
    
    def get_responsiveness_report(self) -> Dict[str, Any]:
        """応答性レポート取得"""
        return self.responsiveness_optimizer.get_responsiveness_report()
    
    def run_responsiveness_test(self) -> Dict[str, Any]:
        """応答性テスト実行"""
        return self.responsiveness_test_suite.run_comprehensive_test()

    def load_implementation_logs(self):
        """実装ログの読み込み"""
        try:
            log_file = "_docs/implementation_log_2025-01-27.md"
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                return log_content
            else:
                return "実装ログファイルが見つかりません"
        except Exception as e:
            return f"実装ログ読み込みエラー: {e}"

    def setup_auto_save(self):
        """自動保存の設定"""
        def auto_save():
            try:
                if not self.data.empty:
                    backup_file = os.path.join(self.backup_dir, f"backup_{self.session_id}.csv")
                    self.data.to_csv(backup_file, index=False)
                    print(f"自動保存完了: {backup_file}")
            except Exception as e:
                print(f"自動保存エラー: {e}")
            
            # 5分後に再実行
            self.root.after(300000, auto_save)
        
        # 初回実行
        self.root.after(300000, auto_save)

def main():
    """メイン関数"""
    root = tk.Tk()
    app = ProfessionalStatisticsGUI(root)
    
    def on_closing():
        """アプリケーション終了時の処理"""
        try:
            # セッション終了処理
            print("Professional Statistics Suiteを終了します")
        except Exception as e:
            print(f"終了処理エラー: {e}")
        finally:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main() 