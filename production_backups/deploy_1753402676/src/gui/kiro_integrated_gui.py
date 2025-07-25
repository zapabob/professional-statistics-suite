# -*- coding: utf-8 -*-
"""
Kiro統合GUIアプリケーション
Kiro Integrated GUI Application

Author: Kiro AI Assistant
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

# Kiro統合モジュールをインポート
from src.ai.ai_integration import AIOrchestrator, QueryProcessor, ContextManager, AnalysisContext
from src.statistics.statistical_method_advisor import StatisticalMethodAdvisor
from src.statistics.assumption_validator import AssumptionValidator
from src.visualization.professional_reports import ReportGenerator
from src.data.data_preprocessing import DataPreprocessor

class KiroIntegratedGUI:
    """Kiro統合GUIアプリケーション"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Kiro統合統計分析システム")
        self.root.geometry("1400x900")
        
        # セッション管理
        self.session_id = f"kiro_session_{int(datetime.now().timestamp())}"
        self.backup_dir = "kiro_backups"
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # データ管理
        self.data = pd.DataFrame()
        self.analysis_results = {}
        self.results_queue = queue.Queue()
        
        # Kiro統合コンポーネント初期化
        self.initialize_kiro_components()
        
        # GUI初期化
        self.create_widgets()
        self.setup_auto_save()
        
        # 実装ログの読み込み
        self.load_implementation_logs()

    def initialize_kiro_components(self):
        """Kiro統合コンポーネントの初期化"""
        try:
            # AI統合コンポーネント
            self.ai_orchestrator = AIOrchestrator()
            self.query_processor = QueryProcessor()
            self.context_manager = ContextManager()
            
            # 統計分析コンポーネント
            self.statistical_advisor = StatisticalMethodAdvisor()
            self.assumption_validator = AssumptionValidator()
            self.data_preprocessor = DataPreprocessor()
            
            # レポート生成
            self.report_generator = ReportGenerator()
            
            print("✅ Kiro統合コンポーネントの初期化が完了しました")
            
        except Exception as e:
            print(f"❌ Kiro統合コンポーネントの初期化エラー: {e}")
            messagebox.showerror("初期化エラー", f"Kiro統合コンポーネントの初期化に失敗しました: {e}")

    def load_implementation_logs(self):
        """実装ログの読み込み"""
        try:
            log_file = "_docs/implementation_log_2025-01-27.md"
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                print("✅ 実装ログを読み込みました")
                return log_content
            else:
                print("⚠️ 実装ログファイルが見つかりません")
                return None
        except Exception as e:
            print(f"❌ 実装ログ読み込みエラー: {e}")
            return None

    def create_widgets(self):
        """GUIウィジェットの作成"""
        # メインフレーム
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ノートブック（タブ）の作成
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # タブの作成
        self.create_data_tab()
        self.create_ai_analysis_tab()
        self.create_statistical_advisor_tab()
        self.create_assumption_validation_tab()
        self.create_visualization_tab()
        self.create_reports_tab()
        self.create_logs_tab()

    def create_data_tab(self):
        """データ管理タブ"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="📊 データ管理")
        
        # データ読み込みセクション
        load_frame = ttk.LabelFrame(data_frame, text="データ読み込み", padding=10)
        load_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(load_frame, text="CSVファイル読み込み", 
                  command=self.load_csv_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_frame, text="データクリア", 
                  command=self.clear_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_frame, text="データ保存", 
                  command=self.save_data).pack(side=tk.LEFT, padx=5)
        
        # データ表示セクション
        display_frame = ttk.LabelFrame(data_frame, text="データ表示", padding=10)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # データ情報表示
        self.data_info_text = scrolledtext.ScrolledText(display_frame, height=8)
        self.data_info_text.pack(fill=tk.X, pady=5)
        
        # データテーブル表示
        self.data_tree = ttk.Treeview(display_frame, show="headings")
        self.data_tree.pack(fill=tk.BOTH, expand=True)

    def create_ai_analysis_tab(self):
        """AI分析タブ"""
        ai_frame = ttk.Frame(self.notebook)
        self.notebook.add(ai_frame, text="🤖 AI分析")
        
        # 自然言語クエリセクション
        query_frame = ttk.LabelFrame(ai_frame, text="自然言語分析クエリ", padding=10)
        query_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(query_frame, text="分析したい内容を自然言語で入力してください:").pack(anchor=tk.W)
        self.query_entry = scrolledtext.ScrolledText(query_frame, height=4)
        self.query_entry.pack(fill=tk.X, pady=5)
        
        ttk.Button(query_frame, text="AI分析実行", 
                  command=self.execute_ai_analysis).pack(pady=5)
        
        # AI分析結果セクション
        result_frame = ttk.LabelFrame(ai_frame, text="AI分析結果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.ai_result_text = scrolledtext.ScrolledText(result_frame)
        self.ai_result_text.pack(fill=tk.BOTH, expand=True)

    def create_statistical_advisor_tab(self):
        """統計手法アドバイザータブ"""
        advisor_frame = ttk.Frame(self.notebook)
        self.notebook.add(advisor_frame, text="📈 統計手法アドバイザー")
        
        # 手法推奨セクション
        recommend_frame = ttk.LabelFrame(advisor_frame, text="統計手法推奨", padding=10)
        recommend_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(recommend_frame, text="データ特性分析", 
                  command=self.analyze_data_characteristics).pack(side=tk.LEFT, padx=5)
        ttk.Button(recommend_frame, text="統計手法推奨", 
                  command=self.recommend_statistical_methods).pack(side=tk.LEFT, padx=5)
        
        # 推奨結果セクション
        self.advisor_result_text = scrolledtext.ScrolledText(advisor_frame)
        self.advisor_result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def create_assumption_validation_tab(self):
        """仮説検証タブ"""
        validation_frame = ttk.Frame(self.notebook)
        self.notebook.add(validation_frame, text="🔍 仮説検証")
        
        # 仮説検証セクション
        validate_frame = ttk.LabelFrame(validation_frame, text="統計的仮説の検証", padding=10)
        validate_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(validate_frame, text="正規性検定", 
                  command=lambda: self.validate_assumptions("normality")).pack(side=tk.LEFT, padx=5)
        ttk.Button(validate_frame, text="等分散性検定", 
                  command=lambda: self.validate_assumptions("homoscedasticity")).pack(side=tk.LEFT, padx=5)
        ttk.Button(validate_frame, text="独立性検定", 
                  command=lambda: self.validate_assumptions("independence")).pack(side=tk.LEFT, padx=5)
        
        # 検証結果セクション
        self.validation_result_text = scrolledtext.ScrolledText(validation_frame)
        self.validation_result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def create_visualization_tab(self):
        """可視化タブ"""
        viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(viz_frame, text="📊 可視化")
        
        # 可視化オプション
        viz_options_frame = ttk.LabelFrame(viz_frame, text="可視化オプション", padding=10)
        viz_options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(viz_options_frame, text="ヒストグラム", 
                  command=lambda: self.create_visualization("histogram")).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_options_frame, text="散布図", 
                  command=lambda: self.create_visualization("scatter")).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_options_frame, text="箱ひげ図", 
                  command=lambda: self.create_visualization("boxplot")).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_options_frame, text="相関行列", 
                  command=lambda: self.create_visualization("correlation")).pack(side=tk.LEFT, padx=5)
        
        # 可視化表示エリア
        self.viz_frame = ttk.Frame(viz_frame)
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def create_reports_tab(self):
        """レポート生成タブ"""
        report_frame = ttk.Frame(self.notebook)
        self.notebook.add(report_frame, text="📋 レポート生成")
        
        # レポート生成オプション
        report_options_frame = ttk.LabelFrame(report_frame, text="レポート生成オプション", padding=10)
        report_options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(report_options_frame, text="包括的レポート生成", 
                  command=self.generate_comprehensive_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(report_options_frame, text="AI分析レポート", 
                  command=self.generate_ai_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(report_options_frame, text="統計手法レポート", 
                  command=self.generate_statistical_report).pack(side=tk.LEFT, padx=5)
        
        # レポート表示エリア
        self.report_text = scrolledtext.ScrolledText(report_frame)
        self.report_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def create_logs_tab(self):
        """ログ表示タブ"""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="📝 ログ")
        
        # ログ表示
        self.logs_text = scrolledtext.ScrolledText(logs_frame)
        self.logs_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # ログ更新ボタン
        ttk.Button(logs_frame, text="ログ更新", 
                  command=self.update_logs).pack(pady=5)

    def load_csv_data(self):
        """CSVデータの読み込み"""
        try:
            filename = filedialog.askopenfilename(
                title="CSVファイルを選択",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                self.data = pd.read_csv(filename)
                self.update_data_display()
                self.log_message(f"CSVファイルを読み込みました: {filename}")
                messagebox.showinfo("成功", f"データを読み込みました\n行数: {len(self.data)}\n列数: {len(self.data.columns)}")
                
        except Exception as e:
            messagebox.showerror("エラー", f"ファイル読み込みエラー: {e}")
            self.log_message(f"ファイル読み込みエラー: {e}")

    def update_data_display(self):
        """データ表示の更新"""
        if not self.data.empty:
            # データ情報の表示
            info_text = f"""
データ情報:
- 行数: {len(self.data)}
- 列数: {len(self.data.columns)}
- 列名: {list(self.data.columns)}
- データ型: {dict(self.data.dtypes)}
- 欠損値: {dict(self.data.isnull().sum())}
"""
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(1.0, info_text)
            
            # データテーブルの更新
            for widget in self.data_tree.winfo_children():
                widget.destroy()
            
            # 列ヘッダーの設定
            self.data_tree["columns"] = list(self.data.columns)
            for col in self.data.columns:
                self.data_tree.heading(col, text=col)
                self.data_tree.column(col, width=100)
            
            # データの挿入（最初の100行のみ）
            for i, row in self.data.head(100).iterrows():
                self.data_tree.insert("", tk.END, values=list(row))

    def clear_data(self):
        """データのクリア"""
        self.data = pd.DataFrame()
        self.data_info_text.delete(1.0, tk.END)
        for widget in self.data_tree.winfo_children():
            widget.destroy()
        self.log_message("データをクリアしました")

    def save_data(self):
        """データの保存"""
        try:
            filename = filedialog.asksaveasfilename(
                title="データを保存",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                self.data.to_csv(filename, index=False)
                self.log_message(f"データを保存しました: {filename}")
                messagebox.showinfo("成功", "データを保存しました")
                
        except Exception as e:
            messagebox.showerror("エラー", f"データ保存エラー: {e}")
            self.log_message(f"データ保存エラー: {e}")

    def execute_ai_analysis(self):
        """AI分析の実行"""
        try:
            query = self.query_entry.get(1.0, tk.END).strip()
            if not query:
                messagebox.showwarning("警告", "分析クエリを入力してください")
                return
            
            if self.data.empty:
                messagebox.showwarning("警告", "データを読み込んでください")
                return
            
            # AI分析の実行（非同期）
            threading.Thread(target=self._execute_ai_analysis_async, 
                           args=(query,), daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("エラー", f"AI分析エラー: {e}")
            self.log_message(f"AI分析エラー: {e}")

    def _execute_ai_analysis_async(self, query):
        """AI分析の非同期実行"""
        try:
            self.log_message(f"AI分析を開始: {query}")
            
            # 分析コンテキストを作成
            context = AnalysisContext(
                user_id="kiro_user",
                session_id=str(uuid.uuid4()),
                data_fingerprint=hash(str(self.data.shape)),
                analysis_history=[]
            )
            
            # AI分析の実行（非同期）
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                self.ai_orchestrator.process_user_query(query, context, data=self.data)
            )
            
            # 結果の表示
            self.root.after(0, lambda: self._display_ai_result(result))
            
        except Exception as exc:
            self.root.after(0, lambda: self._display_ai_error(str(exc)))

    def _display_ai_result(self, result):
        """AI分析結果の表示"""
        try:
            result_text = f"""
AI分析結果:
{result.get('content', '分析結果がありません')}

使用プロバイダー: {result.get('provider_used', 'N/A')}
信頼度: {result.get('confidence', 'N/A')}
処理時間: {result.get('processing_time', 'N/A')}秒
"""
            self.ai_result_text.delete(1.0, tk.END)
            self.ai_result_text.insert(1.0, result_text)
            
            self.log_message("AI分析が完了しました")
            
        except Exception as e:
            self._display_ai_error(str(e))

    def _display_ai_error(self, error_msg):
        """AI分析エラーの表示"""
        self.ai_result_text.delete(1.0, tk.END)
        self.ai_result_text.insert(1.0, f"AI分析エラー: {error_msg}")
        self.log_message(f"AI分析エラー: {error_msg}")

    def analyze_data_characteristics(self):
        """データ特性の分析"""
        try:
            if self.data.empty:
                messagebox.showwarning("警告", "データを読み込んでください")
                return
            
            # データ特性の分析
            characteristics = self.statistical_advisor.analyze_data_characteristics(self.data)
            
            result_text = f"""
データ特性分析結果:
{characteristics}
"""
            self.advisor_result_text.delete(1.0, tk.END)
            self.advisor_result_text.insert(1.0, result_text)
            
            self.log_message("データ特性分析を実行しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"データ特性分析エラー: {e}")
            self.log_message(f"データ特性分析エラー: {e}")

    def recommend_statistical_methods(self):
        """統計手法の推奨"""
        try:
            if self.data.empty:
                messagebox.showwarning("警告", "データを読み込んでください")
                return
            
            # 統計手法の推奨
            suggestions = self.statistical_advisor.suggest_methods(self.data)
            
            result_text = "推奨統計手法:\n\n"
            for i, suggestion in enumerate(suggestions, 1):
                result_text += f"{i}. {suggestion.get('method_name', 'N/A')}\n"
                result_text += f"   信頼度: {suggestion.get('confidence_score', 'N/A')}\n"
                result_text += f"   理由: {suggestion.get('rationale', 'N/A')}\n\n"
            
            self.advisor_result_text.delete(1.0, tk.END)
            self.advisor_result_text.insert(1.0, result_text)
            
            self.log_message("統計手法推奨を実行しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"統計手法推奨エラー: {e}")
            self.log_message(f"統計手法推奨エラー: {e}")

    def validate_assumptions(self, assumption_type):
        """仮説の検証"""
        try:
            if self.data.empty:
                messagebox.showwarning("警告", "データを読み込んでください")
                return
            
            # 仮説の検証
            validation_result = self.assumption_validator.validate_assumptions(
                assumption_type, self.data
            )
            
            result_text = f"""
{assumption_type}仮説検証結果:
{validation_result}
"""
            self.validation_result_text.delete(1.0, tk.END)
            self.validation_result_text.insert(1.0, result_text)
            
            self.log_message(f"{assumption_type}仮説検証を実行しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"仮説検証エラー: {e}")
            self.log_message(f"仮説検証エラー: {e}")

    def create_visualization(self, viz_type):
        """可視化の作成"""
        try:
            if self.data.empty:
                messagebox.showwarning("警告", "データを読み込んでください")
                return
            
            # 既存の可視化をクリア
            for widget in self.viz_frame.winfo_children():
                widget.destroy()
            
            # 新しい可視化の作成
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if viz_type == "histogram":
                # ヒストグラム
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    ax.hist(self.data[col].dropna(), bins=20, alpha=0.7)
                    ax.set_title(f"{col}のヒストグラム")
                    ax.set_xlabel(col)
                    ax.set_ylabel("頻度")
                else:
                    ax.text(0.5, 0.5, "数値データがありません", ha='center', va='center')
                    
            elif viz_type == "scatter":
                # 散布図
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    ax.scatter(self.data[numeric_cols[0]], self.data[numeric_cols[1]])
                    ax.set_title(f"{numeric_cols[0]} vs {numeric_cols[1]}")
                    ax.set_xlabel(numeric_cols[0])
                    ax.set_ylabel(numeric_cols[1])
                else:
                    ax.text(0.5, 0.5, "数値データが2列以上必要です", ha='center', va='center')
                    
            elif viz_type == "boxplot":
                # 箱ひげ図
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    self.data[numeric_cols].boxplot(ax=ax)
                    ax.set_title("箱ひげ図")
                else:
                    ax.text(0.5, 0.5, "数値データがありません", ha='center', va='center')
                    
            elif viz_type == "correlation":
                # 相関行列
                numeric_data = self.data.select_dtypes(include=[np.number])
                if len(numeric_data.columns) > 1:
                    corr_matrix = numeric_data.corr()
                    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
                    ax.set_xticks(range(len(corr_matrix.columns)))
                    ax.set_yticks(range(len(corr_matrix.columns)))
                    ax.set_xticklabels(corr_matrix.columns, rotation=45)
                    ax.set_yticklabels(corr_matrix.columns)
                    ax.set_title("相関行列")
                    plt.colorbar(im, ax=ax)
                else:
                    ax.text(0.5, 0.5, "数値データが2列以上必要です", ha='center', va='center')
            
            # 可視化の表示
            canvas = FigureCanvasTkAgg(fig, self.viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.log_message(f"{viz_type}可視化を作成しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"可視化エラー: {e}")
            self.log_message(f"可視化エラー: {e}")

    def generate_comprehensive_report(self):
        """包括的レポートの生成"""
        try:
            if self.data.empty:
                messagebox.showwarning("警告", "データを読み込んでください")
                return
            
            # 包括的レポートの生成
            report = self.report_generator.generate_comprehensive_report(
                self.data, self.analysis_results
            )
            
            self.report_text.delete(1.0, tk.END)
            self.report_text.insert(1.0, report)
            
            self.log_message("包括的レポートを生成しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"レポート生成エラー: {e}")
            self.log_message(f"レポート生成エラー: {e}")

    def generate_ai_report(self):
        """AI分析レポートの生成"""
        try:
            # AI分析レポートの生成
            report = "AI分析レポート\n" + "=" * 50 + "\n\n"
            report += "このレポートはAI分析の結果を含みます。\n"
            
            self.report_text.delete(1.0, tk.END)
            self.report_text.insert(1.0, report)
            
            self.log_message("AI分析レポートを生成しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"AI分析レポート生成エラー: {e}")
            self.log_message(f"AI分析レポート生成エラー: {e}")

    def generate_statistical_report(self):
        """統計手法レポートの生成"""
        try:
            # 統計手法レポートの生成
            report = "統計手法レポート\n" + "=" * 50 + "\n\n"
            report += "このレポートは統計手法の分析結果を含みます。\n"
            
            self.report_text.delete(1.0, tk.END)
            self.report_text.insert(1.0, report)
            
            self.log_message("統計手法レポートを生成しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"統計手法レポート生成エラー: {e}")
            self.log_message(f"統計手法レポート生成エラー: {e}")

    def update_logs(self):
        """ログの更新"""
        try:
            # 実装ログの読み込み
            log_content = self.load_implementation_logs()
            
            if log_content:
                self.logs_text.delete(1.0, tk.END)
                self.logs_text.insert(1.0, log_content)
            else:
                self.logs_text.delete(1.0, tk.END)
                self.logs_text.insert(1.0, "ログファイルが見つかりません")
                
        except Exception as e:
            self.logs_text.delete(1.0, tk.END)
            self.logs_text.insert(1.0, f"ログ更新エラー: {e}")

    def log_message(self, message):
        """ログメッセージの記録"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # ログファイルに記録
        log_file = os.path.join(self.backup_dir, f"kiro_gui_{datetime.now().strftime('%Y%m%d')}.log")
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"ログファイル書き込みエラー: {e}")

    def setup_auto_save(self):
        """自動保存の設定"""
        def auto_save():
            try:
                if not self.data.empty:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_file = os.path.join(self.backup_dir, f"auto_backup_{timestamp}.csv")
                    self.data.to_csv(backup_file, index=False)
                    self.log_message(f"自動保存: {backup_file}")
            except Exception as e:
                self.log_message(f"自動保存エラー: {e}")
            
            # 5分後に再実行
            self.root.after(300000, auto_save)
        
        # 自動保存開始
        self.root.after(300000, auto_save)

def main():
    """メイン関数"""
    root = tk.Tk()
    app = KiroIntegratedGUI(root)
    
    # ウィンドウクローズ時の処理
    def on_closing():
        try:
            app.log_message("アプリケーションを終了します")
            root.destroy()
        except Exception as e:
            print(f"終了処理エラー: {e}")
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # アプリケーション開始
    root.mainloop()

if __name__ == "__main__":
    main() 