#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parallel Test Runner
並列テスト実行システム

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
License: MIT
"""

import asyncio
import multiprocessing
import subprocess
import sys
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import logging
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from dataclasses import dataclass, field

@dataclass
class ParallelTestConfig:
    """並列テスト設定"""
    max_workers: int = field(default_factory=lambda: min(multiprocessing.cpu_count(), 8))
    test_timeout: int = 300  # 5分
    retry_failed: bool = True
    max_retries: int = 3
    split_by: str = "file"  # file, class, function
    load_scope: str = "session"  # session, module, class, function
    auto_mode: bool = True
    debug: bool = False

@dataclass
class TestExecutionResult:
    """テスト実行結果"""
    test_name: str
    worker_id: int
    success: bool
    execution_time: float
    output: str
    error: Optional[str] = None
    retry_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

class ParallelTestRunner:
    """並列テスト実行システム"""
    
    def __init__(self, config: ParallelTestConfig = None):
        self.config = config or ParallelTestConfig()
        self.logger = logging.getLogger(__name__)
        self.results: List[TestExecutionResult] = []
        self.workers: List[Dict] = []
        
    def setup_workers(self):
        """ワーカー設定"""
        self.logger.info(f"🔧 {self.config.max_workers}個のワーカーを設定中...")
        
        for i in range(self.config.max_workers):
            worker = {
                "id": i,
                "status": "idle",
                "current_test": None,
                "start_time": None,
                "results": []
            }
            self.workers.append(worker)
        
        self.logger.info(f"✅ {len(self.workers)}個のワーカー設定完了")
    
    def discover_tests(self, test_path: str = "src/tests") -> List[str]:
        """テストファイルを発見"""
        test_files = []
        test_path_obj = Path(test_path)
        
        if test_path_obj.exists():
            for test_file in test_path_obj.rglob("test_*.py"):
                test_files.append(str(test_file))
        
        self.logger.info(f"📁 発見されたテストファイル: {len(test_files)}件")
        return test_files
    
    def split_tests_by_file(self, test_files: List[str]) -> List[List[str]]:
        """ファイル単位でテストを分割"""
        chunks = []
        chunk_size = max(1, len(test_files) // self.config.max_workers)
        
        for i in range(0, len(test_files), chunk_size):
            chunk = test_files[i:i + chunk_size]
            chunks.append(chunk)
        
        self.logger.info(f"📦 テストを{len(chunks)}個のチャンクに分割")
        return chunks
    
    def split_tests_by_class(self, test_files: List[str]) -> List[List[str]]:
        """クラス単位でテストを分割"""
        # テストファイルからクラスを抽出
        test_classes = []
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # クラス定義を検索
                import re
                class_pattern = r'class\s+(\w+Test\w*)\s*[:\(]'
                classes = re.findall(class_pattern, content)
                
                for class_name in classes:
                    test_classes.append(f"{test_file}::{class_name}")
                    
            except Exception as e:
                self.logger.warning(f"テストファイル '{test_file}' の解析エラー: {e}")
        
        return self.split_tests_by_file(test_classes)
    
    def execute_test_chunk(self, worker_id: int, test_chunk: List[str]) -> List[TestExecutionResult]:
        """テストチャンクを実行"""
        results = []
        
        for test_file in test_chunk:
            try:
                self.logger.info(f"🔧 ワーカー {worker_id}: {test_file} を実行中...")
                
                start_time = time.time()
                
                # pytestコマンドを構築
                cmd = [
                    sys.executable, "-m", "pytest",
                    test_file,
                    "-v",
                    "--tb=short",
                    "--no-header",
                    "--no-summary"
                ]
                
                if self.config.debug:
                    cmd.append("--capture=no")
                
                # テスト実行
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.test_timeout
                )
                
                execution_time = time.time() - start_time
                
                # 結果を解析
                success = process.returncode == 0
                output = process.stdout
                error = process.stderr if not success else None
                
                result = TestExecutionResult(
                    test_name=test_file,
                    worker_id=worker_id,
                    success=success,
                    execution_time=execution_time,
                    output=output,
                    error=error
                )
                
                results.append(result)
                
                if success:
                    self.logger.info(f"✅ ワーカー {worker_id}: {test_file} 成功 ({execution_time:.2f}秒)")
                else:
                    self.logger.error(f"❌ ワーカー {worker_id}: {test_file} 失敗 ({execution_time:.2f}秒)")
                
            except subprocess.TimeoutExpired:
                self.logger.error(f"⏰ ワーカー {worker_id}: {test_file} タイムアウト")
                result = TestExecutionResult(
                    test_name=test_file,
                    worker_id=worker_id,
                    success=False,
                    execution_time=self.config.test_timeout,
                    output="",
                    error="Test timeout"
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"💥 ワーカー {worker_id}: {test_file} 実行エラー: {e}")
                result = TestExecutionResult(
                    test_name=test_file,
                    worker_id=worker_id,
                    success=False,
                    execution_time=0,
                    output="",
                    error=str(e)
                )
                results.append(result)
        
        return results
    
    async def run_parallel_tests(self, test_path: str = "src/tests") -> Dict:
        """並列テスト実行"""
        self.logger.info("🚀 並列テスト実行開始")
        
        # ワーカー設定
        self.setup_workers()
        
        # テスト発見
        test_files = self.discover_tests(test_path)
        
        if not test_files:
            self.logger.warning("⚠️ テストファイルが見つかりません")
            return {"error": "テストファイルが見つかりません"}
        
        # テスト分割
        if self.config.split_by == "class":
            test_chunks = self.split_tests_by_class(test_files)
        else:
            test_chunks = self.split_tests_by_file(test_files)
        
        # 並列実行
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 各チャンクを並列実行
            future_to_chunk = {
                executor.submit(self.execute_test_chunk, i, chunk): i
                for i, chunk in enumerate(test_chunks)
            }
            
            # 結果を収集
            for future in future_to_chunk:
                worker_id = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    self.results.extend(chunk_results)
                except Exception as e:
                    self.logger.error(f"ワーカー {worker_id} エラー: {e}")
        
        total_time = time.time() - start_time
        
        # 結果集計
        summary = self.generate_summary(total_time)
        
        self.logger.info(f"✅ 並列テスト実行完了: {total_time:.2f}秒")
        
        return {
            "summary": summary,
            "results": [r.__dict__ for r in self.results],
            "workers": self.workers
        }
    
    def generate_summary(self, total_time: float) -> Dict:
        """結果サマリー生成"""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        
        total_execution_time = sum(r.execution_time for r in self.results)
        average_execution_time = total_execution_time / total_tests if total_tests > 0 else 0
        
        # ワーカー別統計
        worker_stats = {}
        for worker in self.workers:
            worker_results = [r for r in self.results if r.worker_id == worker["id"]]
            worker_stats[worker["id"]] = {
                "total_tests": len(worker_results),
                "successful_tests": sum(1 for r in worker_results if r.success),
                "total_time": sum(r.execution_time for r in worker_results)
            }
        
        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0,
            "total_execution_time": total_execution_time,
            "average_execution_time": average_execution_time,
            "parallel_execution_time": total_time,
            "speedup_factor": total_execution_time / total_time if total_time > 0 else 1,
            "worker_stats": worker_stats,
            "max_workers": self.config.max_workers
        }
    
    def save_results(self, output_file: str = "parallel_test_results.json"):
        """結果を保存"""
        try:
            results_data = {
                "timestamp": datetime.now().isoformat(),
                "config": self.config.__dict__,
                "summary": self.generate_summary(0),
                "results": [r.__dict__ for r in self.results]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"✅ 結果を保存: {output_file}")
            
        except Exception as e:
            self.logger.error(f"❌ 結果保存エラー: {e}")

class PytestXdistRunner:
    """pytest-xdistを使用した並列実行"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.logger = logging.getLogger(__name__)
    
    def run_with_xdist(self, test_path: str = "src/tests", additional_args: List[str] = None) -> Dict:
        """pytest-xdistを使用して並列実行"""
        self.logger.info(f"🚀 pytest-xdistで並列実行開始 (ワーカー数: {self.max_workers})")
        
        # コマンド構築
        cmd = [
            sys.executable, "-m", "pytest",
            test_path,
            "-n", str(self.max_workers),  # ワーカー数
            "--dist", "loadfile",  # ファイル単位で分散
            "--tb=short",
            "-v",
            "--junitxml=test-results.xml"
        ]
        
        if additional_args:
            cmd.extend(additional_args)
        
        self.logger.info(f"🔧 実行コマンド: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            # テスト実行
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30分タイムアウト
            )
            
            execution_time = time.time() - start_time
            
            # 結果解析
            success = process.returncode == 0
            output = process.stdout
            error = process.stderr
            
            # JUnit XML結果を解析
            junit_results = self.parse_junit_xml("test-results.xml")
            
            return {
                "success": success,
                "execution_time": execution_time,
                "output": output,
                "error": error,
                "junit_results": junit_results,
                "max_workers": self.max_workers
            }
            
        except subprocess.TimeoutExpired:
            self.logger.error("⏰ テスト実行タイムアウト")
            return {
                "success": False,
                "execution_time": 1800,
                "output": "",
                "error": "Test execution timeout",
                "max_workers": self.max_workers
            }
        
        except Exception as e:
            self.logger.error(f"💥 テスト実行エラー: {e}")
            return {
                "success": False,
                "execution_time": 0,
                "output": "",
                "error": str(e),
                "max_workers": self.max_workers
            }
    
    def parse_junit_xml(self, xml_file: str) -> Dict:
        """JUnit XML結果を解析"""
        try:
            import xml.etree.ElementTree as ET
            
            if not Path(xml_file).exists():
                return {"error": "JUnit XMLファイルが見つかりません"}
            
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # テストスイート情報を抽出
            testsuite = root.find("testsuite")
            if testsuite is None:
                return {"error": "テストスイート情報が見つかりません"}
            
            return {
                "name": testsuite.get("name", ""),
                "tests": int(testsuite.get("tests", 0)),
                "errors": int(testsuite.get("errors", 0)),
                "failures": int(testsuite.get("failures", 0)),
                "skipped": int(testsuite.get("skipped", 0)),
                "time": float(testsuite.get("time", 0))
            }
            
        except Exception as e:
            self.logger.warning(f"JUnit XML解析エラー: {e}")
            return {"error": str(e)}

def main():
    """メイン実行関数"""
    print("🚀 並列テスト実行システム起動")
    
    # 設定
    config = ParallelTestConfig(
        max_workers=4,
        test_timeout=300,
        retry_failed=True,
        debug=True
    )
    
    # 並列テストランナー
    runner = ParallelTestRunner(config)
    
    # テスト実行
    async def run_tests():
        results = await runner.run_parallel_tests()
        
        # 結果表示
        print("\n" + "="*50)
        print("📊 並列テスト実行結果")
        print("="*50)
        
        if "error" in results:
            print(f"❌ エラー: {results['error']}")
            return
        
        summary = results["summary"]
        print(f"✅ 総テスト数: {summary['total_tests']}")
        print(f"✅ 成功テスト数: {summary['successful_tests']}")
        print(f"❌ 失敗テスト数: {summary['failed_tests']}")
        print(f"📈 成功率: {summary['success_rate']:.1f}%")
        print(f"⏱️ 総実行時間: {summary['total_execution_time']:.2f}秒")
        print(f"⚡ 並列実行時間: {summary['parallel_execution_time']:.2f}秒")
        print(f"🚀 高速化率: {summary['speedup_factor']:.2f}x")
        
        # ワーカー別統計
        print(f"\n🔧 ワーカー別統計:")
        for worker_id, stats in summary["worker_stats"].items():
            print(f"  ワーカー {worker_id}: {stats['total_tests']}テスト, {stats['successful_tests']}成功, {stats['total_time']:.2f}秒")
    
    # 非同期実行
    asyncio.run(run_tests())

if __name__ == "__main__":
    main() 