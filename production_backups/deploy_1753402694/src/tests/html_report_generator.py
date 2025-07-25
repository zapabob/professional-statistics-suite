#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HTML Report Generator
HTML形式の詳細レポート生成システム

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
License: MIT
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from dataclasses import dataclass, field
import base64
import hashlib
from jinja2 import Template
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO

@dataclass
class TestResult:
    """テスト結果データ"""
    test_name: str
    status: str  # success, failure, error, skipped
    execution_time: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    coverage_percentage: Optional[float] = None

@dataclass
class CoverageData:
    """カバレッジデータ"""
    file_path: str
    total_lines: int
    covered_lines: int
    coverage_percentage: float
    uncovered_lines: List[int]
    branch_coverage: Optional[float] = None

@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    test_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gc_collections: int
    gc_time: float

class HTMLReportGenerator:
    """HTML形式の詳細レポート生成システム"""
    
    def __init__(self, output_dir: str = "test_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # レポートデータ
        self.test_results: List[TestResult] = []
        self.coverage_data: List[CoverageData] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        self.summary_stats: Dict[str, Any] = {}
        
        # HTMLテンプレート
        self.html_template = self._load_html_template()
        
    def _load_html_template(self) -> Template:
        """HTMLテンプレートを読み込み"""
        template_content = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Professional Statistics Suite - テストレポート</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 0;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .summary-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }
        
        .summary-card:hover {
            transform: translateY(-5px);
        }
        
        .summary-card.success {
            border-left: 5px solid #28a745;
        }
        
        .summary-card.warning {
            border-left: 5px solid #ffc107;
        }
        
        .summary-card.danger {
            border-left: 5px solid #dc3545;
        }
        
        .summary-card.info {
            border-left: 5px solid #17a2b8;
        }
        
        .summary-card h3 {
            font-size: 2em;
            margin-bottom: 10px;
            color: #333;
        }
        
        .summary-card p {
            color: #666;
            font-size: 1.1em;
        }
        
        .chart-container {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }
        
        .chart-container h3 {
            margin-bottom: 20px;
            color: #333;
            font-size: 1.5em;
        }
        
        .test-results {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }
        
        .test-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .test-table th,
        .test-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .test-table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #333;
        }
        
        .test-table tr:hover {
            background-color: #f5f5f5;
        }
        
        .status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 600;
        }
        
        .status-success {
            background-color: #d4edda;
            color: #155724;
        }
        
        .status-failure {
            background-color: #f8d7da;
            color: #721c24;
        }
        
        .status-error {
            background-color: #f8d7da;
            color: #721c24;
        }
        
        .status-skipped {
            background-color: #fff3cd;
            color: #856404;
        }
        
        .coverage-section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }
        
        .coverage-bar {
            background-color: #e9ecef;
            border-radius: 10px;
            height: 20px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .coverage-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        
        .coverage-high {
            background: linear-gradient(90deg, #28a745, #20c997);
        }
        
        .coverage-medium {
            background: linear-gradient(90deg, #ffc107, #fd7e14);
        }
        
        .coverage-low {
            background: linear-gradient(90deg, #dc3545, #e83e8c);
        }
        
        .performance-section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }
        
        .tabs {
            display: flex;
            border-bottom: 1px solid #ddd;
            margin-bottom: 20px;
        }
        
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 1em;
            color: #666;
            transition: color 0.3s ease;
        }
        
        .tab.active {
            color: #667eea;
            border-bottom: 2px solid #667eea;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .summary-cards {
                grid-template-columns: 1fr;
            }
            
            .test-table {
                font-size: 0.9em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Professional Statistics Suite</h1>
            <p>包括的テストレポート - {{ report_date }}</p>
        </div>
        
        <div class="summary-cards">
            <div class="summary-card {{ 'success' if summary.total_tests > 0 and summary.success_rate >= 90 else 'warning' if summary.success_rate >= 70 else 'danger' }}">
                <h3>{{ summary.total_tests }}</h3>
                <p>総テスト数</p>
            </div>
            <div class="summary-card {{ 'success' if summary.success_rate >= 90 else 'warning' if summary.success_rate >= 70 else 'danger' }}">
                <h3>{{ "%.1f"|format(summary.success_rate) }}%</h3>
                <p>成功率</p>
            </div>
            <div class="summary-card {{ 'success' if summary.coverage_percentage >= 80 else 'warning' if summary.coverage_percentage >= 60 else 'danger' }}">
                <h3>{{ "%.1f"|format(summary.coverage_percentage) }}%</h3>
                <p>カバレッジ率</p>
            </div>
            <div class="summary-card info">
                <h3>{{ "%.2f"|format(summary.total_execution_time) }}s</h3>
                <p>総実行時間</p>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>📊 テスト結果分布</h3>
            <div style="text-align: center;">
                <img src="data:image/png;base64,{{ coverage_chart }}" alt="Coverage Chart" style="max-width: 100%; height: auto;">
            </div>
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('test-results')">テスト結果</button>
            <button class="tab" onclick="showTab('coverage')">カバレッジ詳細</button>
            <button class="tab" onclick="showTab('performance')">パフォーマンス</button>
        </div>
        
        <div id="test-results" class="tab-content active">
            <div class="test-results">
                <h3>📋 テスト結果詳細</h3>
                <table class="test-table">
                    <thead>
                        <tr>
                            <th>テスト名</th>
                            <th>ステータス</th>
                            <th>実行時間</th>
                            <th>メモリ使用量</th>
                            <th>CPU使用率</th>
                            <th>タイムスタンプ</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for result in test_results %}
                        <tr>
                            <td>{{ result.test_name }}</td>
                            <td><span class="status-badge status-{{ result.status }}">{{ result.status.upper() }}</span></td>
                            <td>{{ "%.3f"|format(result.execution_time) }}s</td>
                            <td>{{ "%.1f"|format(result.memory_usage) }}MB</td>
                            <td>{{ "%.1f"|format(result.cpu_usage) }}%</td>
                            <td>{{ result.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="coverage" class="tab-content">
            <div class="coverage-section">
                <h3>📈 カバレッジ詳細</h3>
                {% for coverage in coverage_data %}
                <div style="margin-bottom: 20px;">
                    <h4>{{ coverage.file_path }}</h4>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span>{{ "%.1f"|format(coverage.coverage_percentage) }}%</span>
                        <span>{{ coverage.covered_lines }}/{{ coverage.total_lines }} 行</span>
                    </div>
                    <div class="coverage-bar">
                        <div class="coverage-fill {{ 'coverage-high' if coverage.coverage_percentage >= 80 else 'coverage-medium' if coverage.coverage_percentage >= 60 else 'coverage-low' }}" 
                             style="width: {{ coverage.coverage_percentage }}%"></div>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <div id="performance" class="tab-content">
            <div class="performance-section">
                <h3>⚡ パフォーマンスメトリクス</h3>
                <div style="text-align: center;">
                    <img src="data:image/png;base64,{{ performance_chart }}" alt="Performance Chart" style="max-width: 100%; height: auto;">
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>レポート生成日時: {{ report_date }}</p>
            <p>Professional Statistics Suite v1.0.0</p>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {
            // すべてのタブコンテンツを非表示
            var tabContents = document.getElementsByClassName('tab-content');
            for (var i = 0; i < tabContents.length; i++) {
                tabContents[i].classList.remove('active');
            }
            
            // すべてのタブボタンを非アクティブ
            var tabs = document.getElementsByClassName('tab');
            for (var i = 0; i < tabs.length; i++) {
                tabs[i].classList.remove('active');
            }
            
            // 選択されたタブを表示
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
        """
        return Template(template_content)
    
    def add_test_result(self, result: TestResult):
        """テスト結果を追加"""
        self.test_results.append(result)
    
    def add_coverage_data(self, coverage: CoverageData):
        """カバレッジデータを追加"""
        self.coverage_data.append(coverage)
    
    def add_performance_metrics(self, metrics: PerformanceMetrics):
        """パフォーマンスメトリクスを追加"""
        self.performance_metrics.append(metrics)
    
    def _generate_summary_stats(self) -> Dict[str, Any]:
        """サマリー統計を生成"""
        if not self.test_results:
            return {}
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.status == 'success')
        failed_tests = sum(1 for r in self.test_results if r.status == 'failure')
        error_tests = sum(1 for r in self.test_results if r.status == 'error')
        skipped_tests = sum(1 for r in self.test_results if r.status == 'skipped')
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        total_execution_time = sum(r.execution_time for r in self.test_results)
        average_execution_time = total_execution_time / total_tests if total_tests > 0 else 0
        
        # カバレッジ率の計算
        coverage_percentage = 0
        if self.coverage_data:
            total_lines = sum(c.total_lines for c in self.coverage_data)
            covered_lines = sum(c.covered_lines for c in self.coverage_data)
            coverage_percentage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "error_tests": error_tests,
            "skipped_tests": skipped_tests,
            "success_rate": success_rate,
            "total_execution_time": total_execution_time,
            "average_execution_time": average_execution_time,
            "coverage_percentage": coverage_percentage
        }
    
    def _generate_coverage_chart(self) -> str:
        """カバレッジチャートを生成"""
        try:
            plt.figure(figsize=(10, 6))
            
            # テスト結果の分布
            status_counts = {}
            for result in self.test_results:
                status_counts[result.status] = status_counts.get(result.status, 0) + 1
            
            if status_counts:
                colors = {
                    'success': '#28a745',
                    'failure': '#dc3545',
                    'error': '#dc3545',
                    'skipped': '#ffc107'
                }
                
                plt.pie(status_counts.values(), labels=status_counts.keys(), 
                       autopct='%1.1f%%', colors=[colors.get(k, '#6c757d') for k in status_counts.keys()])
                plt.title('テスト結果分布')
            
            # チャートをBase64エンコード
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
            buffer.seek(0)
            chart_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_data
            
        except Exception as e:
            self.logger.warning(f"チャート生成エラー: {e}")
            return ""
    
    def _generate_performance_chart(self) -> str:
        """パフォーマンスチャートを生成"""
        try:
            if not self.performance_metrics:
                return ""
            
            plt.figure(figsize=(12, 8))
            
            # 実行時間の分布
            execution_times = [m.execution_time for m in self.performance_metrics]
            memory_usage = [m.memory_usage_mb for m in self.performance_metrics]
            test_names = [m.test_name for m in self.performance_metrics]
            
            # サブプロット作成
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 実行時間チャート
            ax1.bar(range(len(test_names)), execution_times, color='skyblue', alpha=0.7)
            ax1.set_title('テスト実行時間')
            ax1.set_ylabel('実行時間 (秒)')
            ax1.set_xticks(range(len(test_names)))
            ax1.set_xticklabels(test_names, rotation=45, ha='right')
            
            # メモリ使用量チャート
            ax2.bar(range(len(test_names)), memory_usage, color='lightcoral', alpha=0.7)
            ax2.set_title('メモリ使用量')
            ax2.set_ylabel('メモリ使用量 (MB)')
            ax2.set_xticks(range(len(test_names)))
            ax2.set_xticklabels(test_names, rotation=45, ha='right')
            
            plt.tight_layout()
            
            # チャートをBase64エンコード
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
            buffer.seek(0)
            chart_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return chart_data
            
        except Exception as e:
            self.logger.warning(f"パフォーマンスチャート生成エラー: {e}")
            return ""
    
    def generate_report(self, output_filename: str = "test_report.html") -> str:
        """HTMLレポートを生成"""
        try:
            # サマリー統計を生成
            self.summary_stats = self._generate_summary_stats()
            
            # チャートを生成
            coverage_chart = self._generate_coverage_chart()
            performance_chart = self._generate_performance_chart()
            
            # HTMLレポートを生成
            html_content = self.html_template.render(
                report_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                summary=self.summary_stats,
                test_results=self.test_results,
                coverage_data=self.coverage_data,
                performance_metrics=self.performance_metrics,
                coverage_chart=coverage_chart,
                performance_chart=performance_chart
            )
            
            # ファイルに保存
            output_path = self.output_dir / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"✅ HTMLレポート生成完了: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ HTMLレポート生成エラー: {e}")
            return ""

class ReportManager:
    """レポート管理システム"""
    
    def __init__(self, output_dir: str = "test_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.generator = HTMLReportGenerator(output_dir)
    
    def load_test_results_from_file(self, file_path: str):
        """ファイルからテスト結果を読み込み"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # テスト結果を読み込み
            if 'test_results' in data:
                for result_data in data['test_results']:
                    result = TestResult(
                        test_name=result_data['test_name'],
                        status=result_data['status'],
                        execution_time=result_data['execution_time'],
                        memory_usage=result_data.get('memory_usage', 0),
                        cpu_usage=result_data.get('cpu_usage', 0),
                        timestamp=datetime.fromisoformat(result_data['timestamp']),
                        error_message=result_data.get('error_message'),
                        stack_trace=result_data.get('stack_trace'),
                        coverage_percentage=result_data.get('coverage_percentage')
                    )
                    self.generator.add_test_result(result)
            
            # カバレッジデータを読み込み
            if 'coverage_data' in data:
                for coverage_data in data['coverage_data']:
                    coverage = CoverageData(
                        file_path=coverage_data['file_path'],
                        total_lines=coverage_data['total_lines'],
                        covered_lines=coverage_data['covered_lines'],
                        coverage_percentage=coverage_data['coverage_percentage'],
                        uncovered_lines=coverage_data.get('uncovered_lines', []),
                        branch_coverage=coverage_data.get('branch_coverage')
                    )
                    self.generator.add_coverage_data(coverage)
            
            # パフォーマンスメトリクスを読み込み
            if 'performance_metrics' in data:
                for metrics_data in data['performance_metrics']:
                    metrics = PerformanceMetrics(
                        test_name=metrics_data['test_name'],
                        execution_time=metrics_data['execution_time'],
                        memory_usage_mb=metrics_data['memory_usage_mb'],
                        cpu_usage_percent=metrics_data['cpu_usage_percent'],
                        gc_collections=metrics_data.get('gc_collections', 0),
                        gc_time=metrics_data.get('gc_time', 0)
                    )
                    self.generator.add_performance_metrics(metrics)
            
            self.logger.info(f"✅ テスト結果を読み込みました: {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ テスト結果読み込みエラー: {e}")
    
    def generate_comprehensive_report(self, output_filename: str = "comprehensive_test_report.html") -> str:
        """包括的レポートを生成"""
        return self.generator.generate_report(output_filename)
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """サマリーレポートを生成"""
        return self.generator._generate_summary_stats()

def main():
    """メイン実行関数"""
    print("🚀 HTMLレポート生成システム起動")
    
    # レポートマネージャー初期化
    report_manager = ReportManager()
    
    # サンプルデータを追加
    sample_results = [
        TestResult(
            test_name="test_data_processing",
            status="success",
            execution_time=2.5,
            memory_usage=150.5,
            cpu_usage=25.3,
            timestamp=datetime.now()
        ),
        TestResult(
            test_name="test_ai_integration",
            status="success",
            execution_time=1.8,
            memory_usage=200.2,
            cpu_usage=30.1,
            timestamp=datetime.now()
        ),
        TestResult(
            test_name="test_gui_buttons",
            status="failure",
            execution_time=0.5,
            memory_usage=50.0,
            cpu_usage=10.0,
            timestamp=datetime.now(),
            error_message="Button not found"
        )
    ]
    
    # サンプルデータをレポートに追加
    for result in sample_results:
        report_manager.generator.add_test_result(result)
    
    # サンプルカバレッジデータ
    sample_coverage = [
        CoverageData(
            file_path="src/core/data_processor.py",
            total_lines=100,
            covered_lines=85,
            coverage_percentage=85.0,
            uncovered_lines=[15, 23, 45, 67, 89]
        ),
        CoverageData(
            file_path="src/ai/ai_integration.py",
            total_lines=200,
            covered_lines=180,
            coverage_percentage=90.0,
            uncovered_lines=[25, 50, 75]
        )
    ]
    
    for coverage in sample_coverage:
        report_manager.generator.add_coverage_data(coverage)
    
    # サンプルパフォーマンスメトリクス
    sample_metrics = [
        PerformanceMetrics(
            test_name="test_data_processing",
            execution_time=2.5,
            memory_usage_mb=150.5,
            cpu_usage_percent=25.3,
            gc_collections=5,
            gc_time=0.1
        ),
        PerformanceMetrics(
            test_name="test_ai_integration",
            execution_time=1.8,
            memory_usage_mb=200.2,
            cpu_usage_percent=30.1,
            gc_collections=3,
            gc_time=0.05
        )
    ]
    
    for metrics in sample_metrics:
        report_manager.generator.add_performance_metrics(metrics)
    
    # レポート生成
    report_path = report_manager.generate_comprehensive_report()
    
    if report_path:
        print(f"✅ レポート生成完了: {report_path}")
        
        # サマリー表示
        summary = report_manager.generate_summary_report()
        print("\n📊 レポートサマリー:")
        print(f"  総テスト数: {summary['total_tests']}")
        print(f"  成功率: {summary['success_rate']:.1f}%")
        print(f"  カバレッジ率: {summary['coverage_percentage']:.1f}%")
        print(f"  総実行時間: {summary['total_execution_time']:.2f}秒")
    else:
        print("❌ レポート生成に失敗しました")

if __name__ == "__main__":
    main() 