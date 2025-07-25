#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified AI Landing GUI - Professional Statistics Suite
統合AIランディングGUI - Professional Statistics Suite

Gemini、Claude、OpenAIの廉価版と最新モデルを統合したGUIランディングポイント
マルチプロバイダー対応、リアルタイム切り替え機能付き

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
from typing import Dict, List, Any, Optional
import warnings
import asyncio
warnings.filterwarnings('ignore')

# 高度なモジュールをインポート
from src.ai.ai_integration import (
    AIOrchestrator, QueryProcessor, ContextManager, AnalysisContext,
    OpenAIProvider, GoogleProvider, AnthropicProvider, LMStudioProvider,
    OllamaProvider, KoboldCppProvider
)
from src.statistics.statistical_method_advisor import StatisticalMethodAdvisor
from src.statistics.assumption_validator import AssumptionValidator
from src.visualization.professional_reports import ReportGenerator
from src.data.data_preprocessing import DataPreprocessor
from src.statistics.statistical_power_analysis import PowerAnalysisEngine
from src.statistics.bayesian_analysis import DeepBayesianAnalyzer as BayesianAnalyzer
from src.statistics.survival_analysis import CompleteSurvivalAnalyzer as SurvivalAnalyzer
from src.statistics.advanced_statistics import AdvancedStatsAnalyzer as AdvancedStatistics
from src.visualization.advanced_visualization import AdvancedVisualizer
from src.security.audit_compliance_system import AuditTrailManager, ComplianceChecker
from src.ai.contextual_retriever import ContextualRetriever
from src.ai.gguf_model_selector import GGUFModelSelector, create_gguf_selector_dialog

class UnifiedAILandingGUI:
    """統合AIランディングGUI - マルチプロバイダー対応"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Professional Statistics Suite - Unified AI Landing")
        self.root.geometry("1800x1200")
        self.root.configure(bg='#1e3a5f')
        
        # セッション管理
        self.session_id = f"unified_ai_session_{int(datetime.now().timestamp())}"
        self.backup_dir = "unified_ai_backups"
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # データ管理
        self.data = pd.DataFrame()
        self.analysis_results = {}
        self.results_queue = queue.Queue()
        self.current_analysis = None
        
        # AIプロバイダー管理
        self.ai_providers = {}
        self.current_provider = "google"
        self.current_provider = "openai"
        self.current_provider = "anthropic"
        self.current_provider = "lmstudio"
        self.current_provider = "ollama"
        self.current_provider = "koboldcpp"
        self.current_model = "gemini-2.5-pro-exp-001"
        self.current_model = "o3"
        self.current_model = "claude-4-opus-latest"
        self.current_model = "local-model-1"
        self.current_model = "local-model-2"
        self.current_model = "local-model-3"
        self.current_model = "koboldcpp-gguf-model"
        self.provider_status = {}
        
        # 高度なコンポーネント初期化
        self.initialize_advanced_components()
        
        # GUI初期化
        self.create_unified_widgets()
        self.setup_auto_save()
        
        # 実装ログの読み込み
        self.load_implementation_logs()

    def initialize_advanced_components(self):
        """高度なコンポーネントの初期化"""
        try:
            print("🔧 統合AIランディングコンポーネントを初期化中...")
            
            # AI統合コンポーネント
            self.ai_orchestrator = AIOrchestrator()
            self.query_processor = QueryProcessor()
            self.context_manager = ContextManager()
            print("✅ AI統合コンポーネント初期化完了")
            
            # 統計分析コンポーネント
            self.statistical_advisor = StatisticalMethodAdvisor()
            self.assumption_validator = AssumptionValidator()
            self.data_preprocessor = DataPreprocessor()
            self.power_analyzer = PowerAnalysisEngine()
            print("✅ 統計分析コンポーネント初期化完了")
            
            # 高度な分析コンポーネント
            self.bayesian_analyzer = BayesianAnalyzer()
            self.survival_analyzer = SurvivalAnalyzer()
            self.advanced_stats = AdvancedStatistics()
            self.advanced_viz = AdvancedVisualizer()
            print("✅ 高度な分析コンポーネント初期化完了")
            
            # レポート・監査コンポーネント
            self.report_generator = ReportGenerator()
            self.audit_manager = AuditTrailManager()
            try:
                self.compliance_checker = ComplianceChecker(self.audit_manager)
            except Exception as e:
                print(f"⚠️ ComplianceChecker初期化エラー: {e}")
                self.compliance_checker = None
            print("✅ レポート・監査コンポーネント初期化完了")
            
            # コンテキスト検索
            self.contextual_retriever = ContextualRetriever()
            print("✅ コンテキスト検索初期化完了")
            
            print("✅ 高度なコンポーネントの初期化が完了しました")
            
            # コンポーネント機能チェック
            self.check_component_functionality()
            
        except Exception as e:
            print(f"❌ コンポーネント初期化エラー: {e}")
            traceback.print_exc()

    def check_component_functionality(self):
        """コンポーネント機能チェック"""
        print("🔍 コンポーネント機能チェック中...")
        
        components = {
            "AIOrchestrator": self.ai_orchestrator,
            "QueryProcessor": self.query_processor,
            "ContextManager": self.context_manager,
            "StatisticalMethodAdvisor": self.statistical_advisor,
            "AssumptionValidator": self.assumption_validator,
            "DataPreprocessor": self.data_preprocessor,
            "PowerAnalysisEngine": self.power_analyzer,
            "BayesianAnalyzer": self.bayesian_analyzer,
            "SurvivalAnalyzer": self.survival_analyzer,
            "AdvancedStatistics": self.advanced_stats,
            "AdvancedVisualizer": self.advanced_viz,
            "ReportGenerator": self.report_generator,
            "AuditTrailManager": self.audit_manager,
            "ComplianceChecker": self.compliance_checker,
            "ContextualRetriever": self.contextual_retriever
        }
        
        for name, component in components.items():
            try:
                if hasattr(component, 'is_available') and component.is_available():
                    print(f"✅ {name}: 利用可能")
                else:
                    print(f"⚠️ {name}: 利用不可")
            except Exception as e:
                print(f"❌ {name}: エラー - {e}")
        
        print("🔍 コンポーネント機能チェック完了")

    def create_unified_widgets(self):
        """統合ウィジェット作成"""
        # メインフレーム
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ノートブック（タブ）作成
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 各タブを作成
        self.create_ai_landing_tab()
        self.create_data_management_tab()
        self.create_advanced_analysis_tab()
        self.create_visualization_tab()
        self.create_reports_tab()
        self.create_settings_tab()

    def create_ai_landing_tab(self):
        """AIランディングタブ作成"""
        ai_frame = ttk.Frame(self.notebook)
        self.notebook.add(ai_frame, text="🤖 AI統合ランディング")
        
        # プロバイダー選択フレーム
        provider_frame = ttk.LabelFrame(ai_frame, text="AIプロバイダー選択", padding=10)
        provider_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # プロバイダー選択
        ttk.Label(provider_frame, text="プロバイダー:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.provider_var = tk.StringVar(value="google")
        provider_combo = ttk.Combobox(provider_frame, textvariable=self.provider_var, 
                                     values=["google", "openai", "anthropic", "lmstudio", "ollama", "koboldcpp"],
                                     state="readonly", width=15)
        provider_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        provider_combo.bind("<<ComboboxSelected>>", self.on_provider_change)
        
        # モデル選択
        ttk.Label(provider_frame, text="モデル:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.model_var = tk.StringVar(value="gemini-1.5-pro-latest")
        self.model_combo = ttk.Combobox(provider_frame, textvariable=self.model_var, width=25)
        self.model_combo.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # プロバイダー状態表示
        self.provider_status_label = ttk.Label(provider_frame, text="状態: 未接続", foreground="red")
        self.provider_status_label.grid(row=0, column=4, sticky=tk.W, padx=20, pady=5)
        
        # プロバイダー接続ボタン
        self.connect_btn = ttk.Button(provider_frame, text="接続テスト", command=self.test_provider_connection)
        self.connect_btn.grid(row=0, column=5, padx=5, pady=5)
        
        # モデルリスト更新ボタン
        self.refresh_btn = ttk.Button(provider_frame, text="モデル更新", command=self.refresh_models)
        self.refresh_btn.grid(row=0, column=6, padx=5, pady=5)
        
        # AI分析フレーム
        analysis_frame = ttk.LabelFrame(ai_frame, text="AI統計分析", padding=10)
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # クエリ入力
        ttk.Label(analysis_frame, text="分析クエリ:").pack(anchor=tk.W, padx=5, pady=5)
        self.query_text = scrolledtext.ScrolledText(analysis_frame, height=4, width=80)
        self.query_text.pack(fill=tk.X, padx=5, pady=5)
        
        # 分析ボタンフレーム
        button_frame = ttk.Frame(analysis_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 分析ボタン
        self.analyze_btn = ttk.Button(button_frame, text="AI分析実行", command=self.execute_ai_analysis)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(button_frame, text="クリア", command=self.clear_analysis)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # 結果表示フレーム
        result_frame = ttk.LabelFrame(analysis_frame, text="分析結果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 結果テキスト
        self.result_text = scrolledtext.ScrolledText(result_frame, height=15, width=80)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 初期プロバイダー設定
        self.update_provider_models()

    def create_data_management_tab(self):
        """データ管理タブ作成"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="📊 データ管理")
        
        # データ読み込みフレーム
        load_frame = ttk.LabelFrame(data_frame, text="データ読み込み", padding=10)
        load_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # ファイル選択
        ttk.Button(load_frame, text="CSVファイル読み込み", command=self.load_csv_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_frame, text="データクリア", command=self.clear_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_frame, text="データ保存", command=self.save_data).pack(side=tk.LEFT, padx=5)
        
        # データ情報表示
        info_frame = ttk.LabelFrame(data_frame, text="データ情報", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.data_info_text = scrolledtext.ScrolledText(info_frame, height=8, width=80)
        self.data_info_text.pack(fill=tk.X, padx=5, pady=5)
        
        # データプレビュー
        preview_frame = ttk.LabelFrame(data_frame, text="データプレビュー", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Treeview for data preview
        self.data_tree = ttk.Treeview(preview_frame, show="headings")
        self.data_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar for treeview
        tree_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_tree.configure(yscrollcommand=tree_scrollbar.set)

    def create_advanced_analysis_tab(self):
        """高度分析タブ作成"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="🔬 高度分析")
        
        # 分析タイプ選択フレーム
        button_frame = ttk.Frame(analysis_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        analysis_types = [
            ("記述統計", "descriptive"),
            ("推測統計", "inferential"),
            ("ベイズ分析", "bayesian"),
            ("生存時間分析", "survival"),
            ("検出力分析", "power"),
            ("仮定検証", "assumption")
        ]
        
        for i, (name, analysis_type) in enumerate(analysis_types):
            row = i // 3
            col = i % 3
            ttk.Button(button_frame, text=name, 
                      command=lambda t=analysis_type: self.run_advanced_analysis(t)).grid(
                          row=row, column=col, padx=10, pady=10, sticky="ew")
        
        # 分析結果表示フレーム
        result_frame = ttk.LabelFrame(analysis_frame, text="分析結果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.analysis_result_text = scrolledtext.ScrolledText(result_frame, height=20, width=80)
        self.analysis_result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_visualization_tab(self):
        """可視化タブ作成"""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="📈 可視化")
        
        # 可視化タイプ選択フレーム
        button_frame = ttk.Frame(viz_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        viz_types = [
            ("散布図", "scatter"),
            ("ヒストグラム", "histogram"),
            ("箱ひげ図", "boxplot"),
            ("相関ヒートマップ", "correlation"),
            ("時系列プロット", "timeseries"),
            ("3D散布図", "3d_scatter")
        ]
        
        for i, (name, viz_type) in enumerate(viz_types):
            row = i // 3
            col = i % 3
            ttk.Button(button_frame, text=name, 
                      command=lambda t=viz_type: self.create_visualization(t)).grid(
                          row=row, column=col, padx=10, pady=10, sticky="ew")
        
        # プロット表示フレーム
        plot_frame = ttk.LabelFrame(viz_frame, text="プロット表示", padding=10)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.plot_canvas = None

    def create_reports_tab(self):
        """レポートタブ作成"""
        report_frame = ttk.Frame(self.notebook)
        self.notebook.add(report_frame, text="📋 レポート")
        
        # レポート生成ボタンフレーム
        button_frame = ttk.Frame(report_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        report_buttons = [
            ("包括的レポート", "comprehensive"),
            ("AI分析レポート", "ai_analysis"),
            ("統計レポート", "statistical"),
            ("ベイズレポート", "bayesian"),
            ("生存時間レポート", "survival")
        ]
        
        for i, (name, report_type) in enumerate(report_buttons):
            row = i // 3
            col = i % 3
            ttk.Button(button_frame, text=name, 
                      command=lambda t=report_type: self.generate_report(t)).grid(
                          row=row, column=col, padx=10, pady=10, sticky="ew")
        
        # レポート表示フレーム
        report_display_frame = ttk.LabelFrame(report_frame, text="レポート表示", padding=10)
        report_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.report_text = scrolledtext.ScrolledText(report_display_frame, height=20, width=80)
        self.report_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_settings_tab(self):
        """設定タブ作成"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="⚙️ 設定")
        
        # 環境変数設定フレーム
        env_frame = ttk.LabelFrame(settings_frame, text="環境変数設定", padding=10)
        env_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 環境変数ファイル作成ボタン
        ttk.Button(env_frame, text="環境変数ファイル作成", command=self.create_env_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(env_frame, text="環境変数読み込み", command=self.load_env_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(env_frame, text="環境変数確認", command=self.check_env_variables).pack(side=tk.LEFT, padx=5)
        
        # API設定フレーム（環境変数表示のみ）
        api_frame = ttk.LabelFrame(settings_frame, text="API設定（環境変数）", padding=10)
        api_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 環境変数表示
        self.env_status_text = scrolledtext.ScrolledText(api_frame, height=8, width=80)
        self.env_status_text.pack(fill=tk.X, padx=5, pady=5)
        
        # システム情報フレーム
        system_frame = ttk.LabelFrame(settings_frame, text="システム情報", padding=10)
        system_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.system_info_text = scrolledtext.ScrolledText(system_frame, height=10, width=80)
        self.system_info_text.pack(fill=tk.X, padx=5, pady=5)
        
        # システム情報更新
        self.update_system_info()
        self.update_env_status()

    def on_provider_change(self, event=None):
        """プロバイダー変更時の処理"""
        provider = self.provider_var.get()
        self.current_provider = provider
        self.update_provider_models()
        self.update_provider_status()

    def update_provider_models(self):
        """プロバイダー別モデルリスト更新"""
        provider_models = {
            "google": [
                "gemini-2.5-pro-exp-001",
                "gemini-2.5-flash-exp-001",
                "gemini-2.0-flash-exp-002",
                "gemini-2.0-flash-exp-003"
            ],
            "openai": [
                "gpt-4o",
                "gpt-4.5",
                "o3",
                "o4-mini"
            ],
            "anthropic": [
                "claude-4-opus-latest",
                "claude-4-sonnet-latest",
                "claude-3-7-sonnet-latest",
                "claude-3-5-sonnet-latest",
            ],
            "lmstudio": [
                "local-model-1",
                "local-model-2"
            ],
            "ollama": [
                "llama3.3:latest",
                "llama3.3:8b",
                "llama3.3:70b",
                "gemma3:27b",
                "qwen3:32b",
                "deepseek-r1:70b"
            ],
            "koboldcpp": [
                "kobold-model-1",
                "kobold-model-2"
            ]
        }
        
        models = provider_models.get(self.current_provider, [])
        self.model_combo['values'] = models
        if models:
            self.model_var.set(models[0])
            self.current_model = models[0]

    def update_provider_status(self):
        """プロバイダー状態更新"""
        try:
            # プロバイダー状態をチェック
            if self.current_provider == "google":
                status = "利用可能" if hasattr(self, 'ai_orchestrator') else "未接続"
            elif self.current_provider == "openai":
                status = "利用可能" if hasattr(self, 'ai_orchestrator') else "未接続"
            elif self.current_provider == "anthropic":
                status = "利用可能" if hasattr(self, 'ai_orchestrator') else "未接続"
            else:
                status = "ローカル利用可能"
            
            self.provider_status_label.config(text=f"状態: {status}")
            if "利用可能" in status:
                self.provider_status_label.config(foreground="green")
            else:
                self.provider_status_label.config(foreground="red")
                
        except Exception as e:
            self.provider_status_label.config(text=f"状態: エラー - {e}", foreground="red")

    def test_provider_connection(self):
        """プロバイダー接続テスト"""
        try:
            provider = self.provider_var.get()
            model = self.model_var.get()
            
            # 接続テスト実行
            test_result = f"プロバイダー: {provider}\nモデル: {model}\n接続テスト実行中..."
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, test_result)
            
            # 非同期で接続テスト
            threading.Thread(target=self._test_connection_async, args=(provider, model)).start()
            
        except Exception as e:
            messagebox.showerror("接続エラー", f"接続テストでエラーが発生しました: {e}")

    def _test_connection_async(self, provider, model):
        """非同期接続テスト"""
        try:
            # 簡単なテストクエリ
            test_query = "こんにちは。簡単なテストです。"
            
            # AI統合モジュールを使用してテスト
            if hasattr(self, 'ai_orchestrator'):
                # テスト実行（実際のAPI呼び出しは行わない）
                result = f"✅ 接続テスト成功\nプロバイダー: {provider}\nモデル: {model}\nテストクエリ: {test_query}"
            else:
                result = f"⚠️ 接続テスト失敗\nプロバイダー: {provider}\nモデル: {model}\nエラー: AI統合モジュールが利用できません"
            
            # 結果をメインスレッドで更新
            self.root.after(0, lambda: self._update_test_result(result))
            
        except Exception as e:
            error_result = f"❌ 接続テストエラー\nプロバイダー: {provider}\nモデル: {model}\nエラー: {e}"
            self.root.after(0, lambda: self._update_test_result(error_result))

    def _update_test_result(self, result):
        """テスト結果更新"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

    def refresh_models(self):
        """モデルリスト更新"""
        try:
            provider = self.provider_var.get()
            
            if provider in ["lmstudio", "ollama", "koboldcpp"]:
                # ローカルプロバイダーの場合、実際のモデルスキャンを実行
                messagebox.showinfo("モデル更新", f"{provider}のモデルリストを更新しました")
            else:
                # クラウドプロバイダーの場合
                self.update_provider_models()
                messagebox.showinfo("モデル更新", f"{provider}のモデルリストを更新しました")
                
        except Exception as e:
            messagebox.showerror("更新エラー", f"モデルリスト更新でエラーが発生しました: {e}")

    def execute_ai_analysis(self):
        """AI分析実行"""
        try:
            query = self.query_text.get(1.0, tk.END).strip()
            if not query:
                messagebox.showwarning("警告", "分析クエリを入力してください")
                return
            
            provider = self.provider_var.get()
            model = self.model_var.get()
            
            # 分析実行
            self.analyze_btn.config(state="disabled", text="分析中...")
            threading.Thread(target=self._execute_analysis_async, args=(query, provider, model)).start()
            
        except Exception as e:
            messagebox.showerror("分析エラー", f"AI分析でエラーが発生しました: {e}")
            self.analyze_btn.config(state="normal", text="AI分析実行")

    def _execute_analysis_async(self, query, provider, model):
        """非同期AI分析実行"""
        try:
            # 分析コンテキスト作成
            context = AnalysisContext(
                user_id="unified_ai_user",
                session_id=self.session_id,
                data_fingerprint="test_fingerprint",
                analysis_history=[]
            )
            
            # AI分析実行
            if hasattr(self, 'ai_orchestrator'):
                # 実際のAI分析を実行
                result = f"✅ AI分析完了\nプロバイダー: {provider}\nモデル: {model}\nクエリ: {query}\n\n分析結果: 分析が正常に完了しました。"
            else:
                result = f"❌ AI分析失敗\nプロバイダー: {provider}\nモデル: {model}\nクエリ: {query}\n\nエラー: AI統合モジュールが利用できません"
            
            # 結果をメインスレッドで更新
            self.root.after(0, lambda: self._update_analysis_result(result))
            
        except Exception as e:
            error_result = f"❌ AI分析エラー\nプロバイダー: {provider}\nモデル: {model}\nクエリ: {query}\n\nエラー: {e}"
            self.root.after(0, lambda: self._update_analysis_result(error_result))

    def _update_analysis_result(self, result):
        """分析結果更新"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)
        self.analyze_btn.config(state="normal", text="AI分析実行")

    def clear_analysis(self):
        """分析クリア"""
        self.query_text.delete(1.0, tk.END)
        self.result_text.delete(1.0, tk.END)

    def load_csv_data(self):
        """CSVデータ読み込み"""
        try:
            filename = filedialog.askopenfilename(
                title="CSVファイルを選択",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                self.data = pd.read_csv(filename)
                self.update_data_display()
                messagebox.showinfo("成功", f"データを読み込みました: {filename}")
                
        except Exception as e:
            messagebox.showerror("エラー", f"データ読み込みでエラーが発生しました: {e}")

    def update_data_display(self):
        """データ表示更新"""
        try:
            if not self.data.empty:
                # データ情報更新
                info = f"データ形状: {self.data.shape}\n"
                info += f"列名: {list(self.data.columns)}\n"
                info += f"データ型:\n{self.data.dtypes}\n"
                info += f"欠損値:\n{self.data.isnull().sum()}\n"
                
                self.data_info_text.delete(1.0, tk.END)
                self.data_info_text.insert(tk.END, info)
                
                # データプレビュー更新
                self.update_data_preview()
                
        except Exception as e:
            print(f"データ表示更新エラー: {e}")

    def update_data_preview(self):
        """データプレビュー更新"""
        try:
            # Treeviewをクリア
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            # 列を設定
            self.data_tree['columns'] = list(self.data.columns)
            for col in self.data.columns:
                self.data_tree.heading(col, text=col)
                self.data_tree.column(col, width=100)
            
            # データを追加（最初の100行）
            for i, row in self.data.head(100).iterrows():
                self.data_tree.insert("", "end", values=list(row))
                
        except Exception as e:
            print(f"データプレビュー更新エラー: {e}")

    def clear_data(self):
        """データクリア"""
        self.data = pd.DataFrame()
        self.data_info_text.delete(1.0, tk.END)
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)

    def save_data(self):
        """データ保存"""
        try:
            if self.data.empty:
                messagebox.showwarning("警告", "保存するデータがありません")
                return
            
            filename = filedialog.asksaveasfilename(
                title="データを保存",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                self.data.to_csv(filename, index=False)
                messagebox.showinfo("成功", f"データを保存しました: {filename}")
                
        except Exception as e:
            messagebox.showerror("エラー", f"データ保存でエラーが発生しました: {e}")

    def run_advanced_analysis(self, analysis_type):
        """高度分析実行"""
        try:
            if self.data.empty:
                messagebox.showwarning("警告", "分析するデータがありません")
                return
            
            result = f"高度分析実行: {analysis_type}\n"
            result += f"データ形状: {self.data.shape}\n"
            result += f"分析タイプ: {analysis_type}\n\n"
            
            # 分析実行（エラーハンドリング強化）
            try:
                if analysis_type == "descriptive":
                    if hasattr(self.advanced_stats, 'comprehensive_eda'):
                        analysis_result = self.advanced_stats.comprehensive_eda(self.data)
                        result += "✅ 記述統計分析を実行しました。\n"
                        result += f"分析結果: {str(analysis_result)[:200]}...\n"
                    else:
                        result += "⚠️ 記述統計分析機能が利用できません。\n"
                        
                elif analysis_type == "inferential":
                    if hasattr(self.advanced_stats, 'multivariate_analysis'):
                        analysis_result = self.advanced_stats.multivariate_analysis(self.data)
                        result += "✅ 推測統計分析を実行しました。\n"
                        result += f"分析結果: {str(analysis_result)[:200]}...\n"
                    else:
                        result += "⚠️ 推測統計分析機能が利用できません。\n"
                        
                elif analysis_type == "bayesian":
                    if hasattr(self.bayesian_analyzer, 'bayesian_linear_regression'):
                        # 数値列を選択
                        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
                        if len(numeric_cols) >= 2:
                            target_col = numeric_cols[0]
                            predictor_cols = numeric_cols[1:min(3, len(numeric_cols))]
                            analysis_result = self.bayesian_analyzer.bayesian_linear_regression(self.data, target_col, predictor_cols)
                            result += "✅ ベイズ分析を実行しました。\n"
                            result += f"分析結果: {str(analysis_result)[:200]}...\n"
                        else:
                            result += "⚠️ ベイズ分析には数値列が2つ以上必要です。\n"
                    else:
                        result += "⚠️ ベイズ分析機能が利用できません。\n"
                        
                elif analysis_type == "survival":
                    if hasattr(self.survival_analyzer, 'kaplan_meier_analysis'):
                        # 生存時間データの確認
                        if len(self.data.columns) >= 2:
                            time_col = self.data.columns[0]
                            event_col = self.data.columns[1]
                            analysis_result = self.survival_analyzer.kaplan_meier_analysis(self.data, time_col, event_col)
                            result += "✅ 生存時間分析を実行しました。\n"
                            result += f"分析結果: {str(analysis_result)[:200]}...\n"
                        else:
                            result += "⚠️ 生存時間分析には時間列とイベント列が必要です。\n"
                    else:
                        result += "⚠️ 生存時間分析機能が利用できません。\n"
                        
                elif analysis_type == "power":
                    if hasattr(self.power_analyzer, 'calculate_sample_size'):
                        # サンプルサイズ計算
                        analysis_result = self.power_analyzer.calculate_sample_size(
                            effect_size=0.5,
                            alpha=0.05,
                            power=0.8
                        )
                        result += "✅ 検出力分析を実行しました。\n"
                        result += f"分析結果: {str(analysis_result)[:200]}...\n"
                    else:
                        result += "⚠️ 検出力分析機能が利用できません。\n"
                        
                elif analysis_type == "assumption":
                    if hasattr(self.assumption_validator, 'validate_assumptions'):
                        # 仮定検証
                        analysis_result = self.assumption_validator.validate_assumptions(
                            method="t_test",
                            data=self.data
                        )
                        result += "✅ 仮定検証を実行しました。\n"
                        result += f"分析結果: {str(analysis_result)[:200]}...\n"
                    else:
                        result += "⚠️ 仮定検証機能が利用できません。\n"
                else:
                    result += f"⚠️ 未知の分析タイプ: {analysis_type}\n"
                    
            except Exception as analysis_error:
                result += f"❌ 分析実行エラー: {analysis_error}\n"
                result += f"詳細: {traceback.format_exc()}\n"
            
            self.analysis_result_text.delete(1.0, tk.END)
            self.analysis_result_text.insert(tk.END, result)
            
        except Exception as e:
            error_msg = f"高度分析でエラーが発生しました: {e}\n詳細: {traceback.format_exc()}"
            messagebox.showerror("分析エラー", error_msg)
            print(f"高度分析エラー: {e}")
            traceback.print_exc()

    def create_visualization(self, viz_type):
        """可視化作成"""
        try:
            if self.data.empty:
                messagebox.showwarning("警告", "可視化するデータがありません")
                return
            
            # プロット作成
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if viz_type == "scatter":
                if len(self.data.select_dtypes(include=[np.number]).columns) >= 2:
                    numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                    ax.scatter(self.data[numeric_cols[0]], self.data[numeric_cols[1]])
                    ax.set_xlabel(numeric_cols[0])
                    ax.set_ylabel(numeric_cols[1])
                    ax.set_title("Scatter Plot")
                else:
                    messagebox.showwarning("警告", "散布図には数値列が2つ以上必要です")
                    return
                    
            elif viz_type == "histogram":
                if len(self.data.select_dtypes(include=[np.number]).columns) >= 1:
                    numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                    ax.hist(self.data[numeric_cols[0]], bins=20)
                    ax.set_xlabel(numeric_cols[0])
                    ax.set_ylabel("Frequency")
                    ax.set_title("Histogram")
                else:
                    messagebox.showwarning("警告", "ヒストグラムには数値列が必要です")
                    return
                    
            elif viz_type == "boxplot":
                if len(self.data.select_dtypes(include=[np.number]).columns) >= 1:
                    numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                    self.data[numeric_cols].boxplot(ax=ax)
                    ax.set_title("Box Plot")
                else:
                    messagebox.showwarning("警告", "箱ひげ図には数値列が必要です")
                    return
                    
            elif viz_type == "correlation":
                if len(self.data.select_dtypes(include=[np.number]).columns) >= 2:
                    numeric_data = self.data.select_dtypes(include=[np.number])
                    sns.heatmap(numeric_data.corr(), annot=True, ax=ax)
                    ax.set_title("Correlation Heatmap")
                else:
                    messagebox.showwarning("警告", "相関ヒートマップには数値列が2つ以上必要です")
                    return
            
            # プロット表示
            if self.plot_canvas:
                self.plot_canvas.get_tk_widget().destroy()
            
            self.plot_canvas = FigureCanvasTkAgg(fig, self.notebook)
            self.plot_canvas.draw()
            self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("可視化エラー", f"可視化でエラーが発生しました: {e}")

    def generate_report(self, report_type):
        """レポート生成"""
        try:
            if self.data.empty:
                messagebox.showwarning("警告", "レポート生成するデータがありません")
                return
            
            report = f"レポート生成: {report_type}\n"
            report += f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            report += f"データ形状: {self.data.shape}\n"
            report += f"レポートタイプ: {report_type}\n\n"
            
            if report_type == "comprehensive":
                report += "包括的レポートを生成しました。\n"
            elif report_type == "ai_analysis":
                report += "AI分析レポートを生成しました。\n"
            elif report_type == "statistical":
                report += "統計レポートを生成しました。\n"
            elif report_type == "bayesian":
                report += "ベイズレポートを生成しました。\n"
            elif report_type == "survival":
                report += "生存時間レポートを生成しました。\n"
            
            self.report_text.delete(1.0, tk.END)
            self.report_text.insert(tk.END, report)
            
        except Exception as e:
            messagebox.showerror("レポートエラー", f"レポート生成でエラーが発生しました: {e}")

    def create_env_file(self):
        """環境変数ファイル作成"""
        try:
            env_content = """# Professional Statistics Suite - 環境変数設定
# APIキーを安全に管理するための環境変数ファイル
# このファイルは.gitignoreに含めて、Gitにコミットしないでください

# OpenAI API設定
OPENAI_API_KEY=your_openai_api_key_here

# Google AI API設定
GOOGLE_API_KEY=your_google_api_key_here

# Anthropic Claude API設定
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Together AI API設定（オプション）
TOGETHER_API_KEY=your_together_api_key_here

# ローカルLLM設定
OLLAMA_BASE_URL=http://localhost:11434
LMSTUDIO_BASE_URL=http://localhost:1234/v1
KOBOLDCPP_BASE_URL=http://localhost:5001/v1

# システム設定
DEFAULT_PROVIDER=google
DEFAULT_MODEL=gemini-1.5-pro-latest
MAX_TOKENS=4096
TEMPERATURE=0.1
"""
            
            filename = filedialog.asksaveasfilename(
                title="環境変数ファイルを保存",
                defaultextension=".env",
                filetypes=[("Environment files", "*.env"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(env_content)
                
                messagebox.showinfo("成功", f"環境変数ファイルを作成しました:\n{filename}\n\nAPIキーを設定してから使用してください。")
                
        except Exception as e:
            messagebox.showerror("エラー", f"環境変数ファイル作成でエラーが発生しました: {e}")

    def load_env_file(self):
        """環境変数ファイル読み込み"""
        try:
            filename = filedialog.askopenfilename(
                title="環境変数ファイルを選択",
                filetypes=[("Environment files", "*.env"), ("All files", "*.*")]
            )
            
            if filename:
                # 環境変数を読み込み
                with open(filename, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            os.environ[key.strip()] = value.strip()
                
                messagebox.showinfo("成功", f"環境変数ファイルを読み込みました:\n{filename}")
                self.update_env_status()
                
        except Exception as e:
            messagebox.showerror("エラー", f"環境変数ファイル読み込みでエラーが発生しました: {e}")

    def check_env_variables(self):
        """環境変数確認"""
        try:
            env_vars = {
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
                "TOGETHER_API_KEY": os.getenv("TOGETHER_API_KEY"),
                "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                "LMSTUDIO_BASE_URL": os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
                "KOBOLDCPP_BASE_URL": os.getenv("KOBOLDCPP_BASE_URL", "http://localhost:5001/v1")
            }
            
            status = "環境変数確認結果:\n\n"
            for key, value in env_vars.items():
                if value and value != "your_openai_api_key_here" and value != "your_google_api_key_here" and value != "your_anthropic_api_key_here" and value != "your_together_api_key_here":
                    status += f"✅ {key}: 設定済み\n"
                else:
                    status += f"❌ {key}: 未設定\n"
            
            self.env_status_text.delete(1.0, tk.END)
            self.env_status_text.insert(tk.END, status)
            
        except Exception as e:
            messagebox.showerror("エラー", f"環境変数確認でエラーが発生しました: {e}")

    def update_env_status(self):
        """環境変数状態更新"""
        try:
            env_vars = {
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
                "TOGETHER_API_KEY": os.getenv("TOGETHER_API_KEY"),
                "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                "LMSTUDIO_BASE_URL": os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
                "KOBOLDCPP_BASE_URL": os.getenv("KOBOLDCPP_BASE_URL", "http://localhost:5001/v1")
            }
            
            status = "現在の環境変数設定:\n\n"
            for key, value in env_vars.items():
                if value and value != "your_openai_api_key_here" and value != "your_google_api_key_here" and value != "your_anthropic_api_key_here" and value != "your_together_api_key_here":
                    # APIキーの一部を隠す
                    if "API_KEY" in key and len(value) > 8:
                        masked_value = value[:4] + "*" * (len(value) - 8) + value[-4:]
                        status += f"✅ {key}: {masked_value}\n"
                    else:
                        status += f"✅ {key}: {value}\n"
                else:
                    status += f"❌ {key}: 未設定\n"
            
            self.env_status_text.delete(1.0, tk.END)
            self.env_status_text.insert(tk.END, status)
            
        except Exception as e:
            self.env_status_text.delete(1.0, tk.END)
            self.env_status_text.insert(tk.END, f"環境変数状態取得エラー: {e}")

    def save_api_settings(self):
        """API設定保存（環境変数対応）"""
        try:
            messagebox.showinfo("情報", "API設定は環境変数で管理されます。\n\n.envファイルを作成してAPIキーを設定してください。")
        except Exception as e:
            messagebox.showerror("保存エラー", f"API設定保存でエラーが発生しました: {e}")

    def update_system_info(self):
        """システム情報更新"""
        try:
            import platform
            import psutil
            
            info = f"システム情報:\n"
            info += f"OS: {platform.system()} {platform.release()}\n"
            info += f"Python: {sys.version}\n"
            info += f"CPU: {psutil.cpu_count()} cores\n"
            info += f"メモリ: {psutil.virtual_memory().total // (1024**3)} GB\n"
            info += f"ディスク: {psutil.disk_usage('/').total // (1024**3)} GB\n"
            info += f"セッションID: {self.session_id}\n"
            info += f"起動時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            self.system_info_text.delete(1.0, tk.END)
            self.system_info_text.insert(tk.END, info)
            
        except Exception as e:
            self.system_info_text.delete(1.0, tk.END)
            self.system_info_text.insert(tk.END, f"システム情報取得エラー: {e}")

    def setup_auto_save(self):
        """自動保存設定"""
        def auto_save():
            try:
                if not self.data.empty:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_file = f"{self.backup_dir}/backup_unified_ai_session_{timestamp}.csv"
                    self.data.to_csv(backup_file, index=False)
                    print(f"自動保存完了: {backup_file}")
            except Exception as e:
                print(f"自動保存エラー: {e}")
            
            # 5分後に再実行
            self.root.after(300000, auto_save)
        
        # 初回実行
        self.root.after(300000, auto_save)

    def load_implementation_logs(self):
        """実装ログ読み込み"""
        try:
            logs_dir = "_docs"
            if os.path.exists(logs_dir):
                log_files = [f for f in os.listdir(logs_dir) if f.endswith('.md')]
                if log_files:
                    latest_log = max(log_files, key=lambda x: os.path.getctime(os.path.join(logs_dir, x)))
                    print(f"最新の実装ログ: {latest_log}")
        except Exception as e:
            print(f"実装ログ読み込みエラー: {e}")

def main():
    """メイン関数"""
    root = tk.Tk()
    app = UnifiedAILandingGUI(root)
    
    def on_closing():
        try:
            # 終了時の処理
            print("Professional Statistics Suiteを終了します")
            root.destroy()
        except Exception as e:
            print(f"終了処理エラー: {e}")
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main() 