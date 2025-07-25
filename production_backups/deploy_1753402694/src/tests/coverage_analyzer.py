#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Coverage Analyzer
テストカバレッジ分析システム

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
License: MIT
"""

import coverage
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import json
import logging
from datetime import datetime
import subprocess
from dataclasses import dataclass, field
import ast
import re

@dataclass
class CoverageMetrics:
    """カバレッジメトリクス"""
    file_path: str
    total_lines: int
    covered_lines: int
    missing_lines: int
    coverage_percentage: float
    uncovered_lines: List[int] = field(default_factory=list)
    branch_coverage: Optional[float] = None
    function_coverage: Optional[float] = None
    class_coverage: Optional[float] = None

@dataclass
class ModuleCoverage:
    """モジュールカバレッジ"""
    module_name: str
    total_files: int
    covered_files: int
    total_lines: int
    covered_lines: int
    coverage_percentage: float
    files: List[CoverageMetrics] = field(default_factory=list)

class CoverageAnalyzer:
    """カバレッジ分析システム"""
    
    def __init__(self, source_path: str = "src", test_path: str = "src/tests"):
        self.source_path = Path(source_path)
        self.test_path = Path(test_path)
        self.logger = logging.getLogger(__name__)
        self.cov = coverage.Coverage()
        self.coverage_data: Dict[str, CoverageMetrics] = {}
        
    def start_coverage_measurement(self):
        """カバレッジ測定開始"""
        self.logger.info("📊 カバレッジ測定開始")
        self.cov.start()
    
    def stop_coverage_measurement(self):
        """カバレッジ測定停止"""
        self.logger.info("📊 カバレッジ測定停止")
        self.cov.stop()
        self.cov.save()
    
    def analyze_coverage(self) -> Dict:
        """カバレッジ分析"""
        self.logger.info("🔍 カバレッジ分析実行中...")
        
        # カバレッジデータを取得
        self.cov.load()
        
        # ソースファイルを分析
        source_files = self._discover_source_files()
        
        total_coverage = {
            "total_files": 0,
            "covered_files": 0,
            "total_lines": 0,
            "covered_lines": 0,
            "coverage_percentage": 0.0,
            "modules": {},
            "files": {}
        }
        
        for file_path in source_files:
            try:
                metrics = self._analyze_file_coverage(file_path)
                if metrics:
                    self.coverage_data[file_path] = metrics
                    
                    # 全体統計を更新
                    total_coverage["total_files"] += 1
                    total_coverage["total_lines"] += metrics.total_lines
                    total_coverage["covered_lines"] += metrics.covered_lines
                    
                    if metrics.coverage_percentage > 0:
                        total_coverage["covered_files"] += 1
                    
                    # モジュール別統計を更新
                    module_name = self._get_module_name(file_path)
                    if module_name not in total_coverage["modules"]:
                        total_coverage["modules"][module_name] = ModuleCoverage(
                            module_name=module_name,
                            total_files=0,
                            covered_files=0,
                            total_lines=0,
                            covered_lines=0,
                            coverage_percentage=0.0
                        )
                    
                    module = total_coverage["modules"][module_name]
                    module.total_files += 1
                    module.total_lines += metrics.total_lines
                    module.covered_lines += metrics.covered_lines
                    module.files.append(metrics)
                    
                    if metrics.coverage_percentage > 0:
                        module.covered_files += 1
                    
            except Exception as e:
                self.logger.warning(f"ファイル '{file_path}' の分析エラー: {e}")
        
        # カバレッジ率を計算
        if total_coverage["total_lines"] > 0:
            total_coverage["coverage_percentage"] = (
                total_coverage["covered_lines"] / total_coverage["total_lines"] * 100
            )
        
        # モジュール別カバレッジ率を計算
        for module in total_coverage["modules"].values():
            if module.total_lines > 0:
                module.coverage_percentage = (
                    module.covered_lines / module.total_lines * 100
                )
        
        # ファイル別データを追加
        total_coverage["files"] = {
            path: metrics.__dict__ for path, metrics in self.coverage_data.items()
        }
        
        return total_coverage
    
    def _discover_source_files(self) -> List[str]:
        """ソースファイルを発見"""
        source_files = []
        
        if self.source_path.exists():
            for file_path in self.source_path.rglob("*.py"):
                # テストファイルと__pycache__を除外
                if not file_path.name.startswith("test_") and "__pycache__" not in str(file_path):
                    source_files.append(str(file_path))
        
        self.logger.info(f"📁 発見されたソースファイル: {len(source_files)}件")
        return source_files
    
    def _analyze_file_coverage(self, file_path: str) -> Optional[CoverageMetrics]:
        """ファイルのカバレッジを分析"""
        try:
            # ファイルの行数を取得
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            
            # カバレッジデータを取得
            file_coverage = self.cov.analysis2(file_path)
            if not file_coverage:
                return None
            
            missing_lines, executable_lines, missing_branches, executable_branches = file_coverage
            
            covered_lines = executable_lines - len(missing_lines)
            
            # カバレッジ率を計算
            coverage_percentage = (covered_lines / executable_lines * 100) if executable_lines > 0 else 0
            
            # ブランチカバレッジを計算
            branch_coverage = None
            if executable_branches > 0:
                covered_branches = executable_branches - len(missing_branches)
                branch_coverage = (covered_branches / executable_branches * 100)
            
            # 関数とクラスのカバレッジを分析
            function_coverage, class_coverage = self._analyze_code_structure(file_path, lines)
            
            return CoverageMetrics(
                file_path=file_path,
                total_lines=total_lines,
                covered_lines=covered_lines,
                missing_lines=len(missing_lines),
                coverage_percentage=coverage_percentage,
                uncovered_lines=list(missing_lines),
                branch_coverage=branch_coverage,
                function_coverage=function_coverage,
                class_coverage=class_coverage
            )
            
        except Exception as e:
            self.logger.warning(f"ファイル '{file_path}' の分析エラー: {e}")
            return None
    
    def _analyze_code_structure(self, file_path: str, lines: List[str]) -> tuple[Optional[float], Optional[float]]:
        """コード構造を分析（関数とクラスのカバレッジ）"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            functions = []
            classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node)
            
            # 関数とクラスのカバレッジを計算（簡易版）
            # 実際の実装では、より詳細な分析が必要
            function_coverage = None
            class_coverage = None
            
            return function_coverage, class_coverage
            
        except Exception as e:
            self.logger.debug(f"コード構造分析エラー: {e}")
            return None, None
    
    def _get_module_name(self, file_path: str) -> str:
        """モジュール名を取得"""
        try:
            relative_path = Path(file_path).relative_to(self.source_path)
            return str(relative_path.parent)
        except ValueError:
            return "unknown"
    
    def generate_coverage_report(self, output_file: str = "coverage_report.json") -> Dict:
        """カバレッジレポート生成"""
        coverage_data = self.analyze_coverage()
        
        # レポートに追加情報を含める
        report = {
            "timestamp": datetime.now().isoformat(),
            "source_path": str(self.source_path),
            "test_path": str(self.test_path),
            "coverage_data": coverage_data,
            "recommendations": self._generate_recommendations(coverage_data)
        }
        
        # ファイルに保存
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"✅ カバレッジレポート保存: {output_file}")
            
        except Exception as e:
            self.logger.error(f"❌ レポート保存エラー: {e}")
        
        return report
    
    def _generate_recommendations(self, coverage_data: Dict) -> List[str]:
        """カバレッジ改善の推奨事項を生成"""
        recommendations = []
        
        overall_coverage = coverage_data["coverage_percentage"]
        
        if overall_coverage < 50:
            recommendations.append("⚠️ 全体的なカバレッジが50%未満です。基本的なテストケースの追加が必要です")
        elif overall_coverage < 80:
            recommendations.append("📈 カバレッジを80%以上に向上させることを推奨します")
        
        # カバレッジが低いモジュールを特定
        low_coverage_modules = []
        for module_name, module_data in coverage_data["modules"].items():
            if module_data.coverage_percentage < 70:
                low_coverage_modules.append((module_name, module_data.coverage_percentage))
        
        if low_coverage_modules:
            recommendations.append(f"🔍 カバレッジが低いモジュール: {', '.join([f'{name}({cov:.1f}%)' for name, cov in low_coverage_modules])}")
        
        # カバレッジが0%のファイルを特定
        uncovered_files = []
        for file_path, metrics in coverage_data["files"].items():
            if metrics["coverage_percentage"] == 0:
                uncovered_files.append(file_path)
        
        if uncovered_files:
            recommendations.append(f"❌ テストカバレッジが0%のファイル: {len(uncovered_files)}件")
        
        return recommendations

class TestCoverageGenerator:
    """テストカバレッジ生成システム"""
    
    def __init__(self, source_path: str = "src"):
        self.source_path = Path(source_path)
        self.logger = logging.getLogger(__name__)
    
    def generate_test_templates(self, target_file: str) -> List[str]:
        """テストテンプレートを生成"""
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            test_templates = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    test_templates.extend(self._generate_class_test_templates(node))
                elif isinstance(node, ast.FunctionDef):
                    test_templates.extend(self._generate_function_test_templates(node))
            
            return test_templates
            
        except Exception as e:
            self.logger.error(f"テストテンプレート生成エラー: {e}")
            return []
    
    def _generate_class_test_templates(self, class_node: ast.ClassDef) -> List[str]:
        """クラスのテストテンプレートを生成"""
        templates = []
        
        # クラス名からテストクラス名を生成
        test_class_name = f"Test{class_node.name}"
        
        template = f"""
class {test_class_name}:
    \"\"\"{class_node.name}クラスのテスト\"\"\"
    
    def setup_method(self):
        \"\"\"テスト前のセットアップ\"\"\"
        pass
    
    def teardown_method(self):
        \"\"\"テスト後のクリーンアップ\"\"\"
        pass
"""
        
        # メソッドのテストテンプレートを生成
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                method_template = self._generate_method_test_template(node, class_node.name)
                template += method_template
        
        templates.append(template)
        return templates
    
    def _generate_function_test_templates(self, func_node: ast.FunctionDef) -> List[str]:
        """関数のテストテンプレートを生成"""
        templates = []
        
        test_func_name = f"test_{func_node.name}"
        
        template = f"""
def {test_func_name}():
    \"\"\"{func_node.name}関数のテスト\"\"\"
    # TODO: テストケースを実装
    pass
"""
        
        templates.append(template)
        return templates
    
    def _generate_method_test_template(self, method_node: ast.FunctionDef, class_name: str) -> str:
        """メソッドのテストテンプレートを生成"""
        test_method_name = f"test_{method_node.name}"
        
        template = f"""
    def {test_method_name}(self):
        \"\"\"{method_node.name}メソッドのテスト\"\"\"
        # TODO: テストケースを実装
        pass
"""
        
        return template

def main():
    """メイン実行関数"""
    print("🚀 テストカバレッジ分析システム起動")
    
    # カバレッジ分析システム初期化
    analyzer = CoverageAnalyzer()
    
    # カバレッジ測定開始
    analyzer.start_coverage_measurement()
    
    # サンプルテスト実行（実際のテストを実行）
    print("🧪 サンプルテスト実行中...")
    try:
        # 既存のテストを実行
        subprocess.run([
            sys.executable, "-m", "pytest", "src/tests", "-v", "--tb=short"
        ], capture_output=True, text=True, timeout=300)
    except Exception as e:
        print(f"⚠️ テスト実行エラー: {e}")
    
    # カバレッジ測定停止
    analyzer.stop_coverage_measurement()
    
    # カバレッジレポート生成
    print("📊 カバレッジレポート生成中...")
    report = analyzer.generate_coverage_report()
    
    # 結果表示
    print("\n" + "="*50)
    print("📊 テストカバレッジ分析結果")
    print("="*50)
    
    coverage_data = report["coverage_data"]
    print(f"📁 総ファイル数: {coverage_data['total_files']}")
    print(f"✅ カバー済みファイル数: {coverage_data['covered_files']}")
    print(f"📊 総行数: {coverage_data['total_lines']}")
    print(f"✅ カバー済み行数: {coverage_data['covered_lines']}")
    print(f"📈 カバレッジ率: {coverage_data['coverage_percentage']:.1f}%")
    
    # モジュール別カバレッジ
    print(f"\n📦 モジュール別カバレッジ:")
    for module_name, module_data in coverage_data["modules"].items():
        print(f"  {module_name}: {module_data.coverage_percentage:.1f}%")
    
    # 推奨事項
    if report["recommendations"]:
        print(f"\n💡 推奨事項:")
        for rec in report["recommendations"]:
            print(f"  - {rec}")

if __name__ == "__main__":
    main() 