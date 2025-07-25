#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Performance Optimizer
テストパフォーマンス最適化システム

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
License: MIT
"""

import time
import psutil
import gc
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

@dataclass
class TestPerformanceMetrics:
    """テストパフォーマンスメトリクス"""
    test_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gc_collections: int
    gc_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "execution_time": self.execution_time,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "gc_collections": self.gc_collections,
            "gc_time": self.gc_time,
            "timestamp": self.timestamp.isoformat()
        }

class PerformanceProfiler:
    """パフォーマンスプロファイラー"""
    
    def __init__(self):
        self.metrics: List[TestPerformanceMetrics] = []
        self.process = psutil.Process()
        self.logger = logging.getLogger(__name__)
        
    def profile_test(self, test_func: Callable, test_name: str) -> TestPerformanceMetrics:
        """テスト関数のパフォーマンスをプロファイリング"""
        # GC統計をリセット
        gc.collect()
        gc_stats_before = gc.get_stats()
        
        # メモリ使用量を記録
        memory_before = self.process.memory_info().rss / 1024 / 1024
        
        # CPU使用量を記録
        cpu_before = self.process.cpu_percent()
        
        # 実行時間測定
        start_time = time.time()
        
        try:
            # テスト実行
            result = test_func()
            execution_time = time.time() - start_time
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"テスト実行エラー: {e}")
            result = None
        
        # 実行後のメトリクス記録
        memory_after = self.process.memory_info().rss / 1024 / 1024
        cpu_after = self.process.cpu_percent()
        
        # GC統計を取得
        gc_stats_after = gc.get_stats()
        gc_collections = sum(stats['collections'] for stats in gc_stats_after) - sum(stats['collections'] for stats in gc_stats_before)
        gc_time = sum(stats['collections_time'] for stats in gc_stats_after) - sum(stats['collections_time'] for stats in gc_stats_before)
        
        # メトリクス作成
        metrics = TestPerformanceMetrics(
            test_name=test_name,
            execution_time=execution_time,
            memory_usage_mb=memory_after - memory_before,
            cpu_usage_percent=(cpu_before + cpu_after) / 2,
            gc_collections=gc_collections,
            gc_time=gc_time
        )
        
        self.metrics.append(metrics)
        return metrics
    
    def get_slow_tests(self, threshold_seconds: float = 1.0) -> List[TestPerformanceMetrics]:
        """遅いテストを特定"""
        return [m for m in self.metrics if m.execution_time > threshold_seconds]
    
    def get_memory_intensive_tests(self, threshold_mb: float = 100.0) -> List[TestPerformanceMetrics]:
        """メモリ使用量が多いテストを特定"""
        return [m for m in self.metrics if m.memory_usage_mb > threshold_mb]
    
    def generate_performance_report(self) -> Dict:
        """パフォーマンスレポート生成"""
        if not self.metrics:
            return {"error": "メトリクスがありません"}
        
        total_tests = len(self.metrics)
        total_time = sum(m.execution_time for m in self.metrics)
        total_memory = sum(m.memory_usage_mb for m in self.metrics)
        
        slow_tests = self.get_slow_tests()
        memory_intensive_tests = self.get_memory_intensive_tests()
        
        return {
            "summary": {
                "total_tests": total_tests,
                "total_execution_time": total_time,
                "average_execution_time": total_time / total_tests,
                "total_memory_usage_mb": total_memory,
                "average_memory_usage_mb": total_memory / total_tests
            },
            "slow_tests": [m.to_dict() for m in slow_tests],
            "memory_intensive_tests": [m.to_dict() for m in memory_intensive_tests],
            "all_metrics": [m.to_dict() for m in self.metrics]
        }

class TestOptimizer:
    """テスト最適化システム"""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.logger = logging.getLogger(__name__)
        self.optimization_suggestions: List[str] = []
        
    def optimize_test_collection(self, test_paths: List[str]) -> Dict:
        """テスト収集の最適化"""
        suggestions = []
        
        # テストパスの最適化
        if len(test_paths) > 10:
            suggestions.append("テストパスが多すぎます。testpaths設定で特定のディレクトリに限定してください")
        
        # 不要なファイルの除外
        suggestions.append("__pycache__、.pycファイルを除外してください")
        suggestions.append("テストデータファイルは別ディレクトリに配置してください")
        
        return {
            "suggestions": suggestions,
            "optimized_paths": test_paths
        }
    
    def optimize_fixtures(self, fixture_usage: Dict) -> Dict:
        """フィクスチャの最適化"""
        suggestions = []
        
        for fixture_name, usage in fixture_usage.items():
            if usage.get("scope") == "function" and usage.get("setup_time", 0) > 1.0:
                suggestions.append(f"フィクスチャ '{fixture_name}' のスコープを 'module' または 'session' に変更してください")
            
            if usage.get("memory_usage", 0) > 50:
                suggestions.append(f"フィクスチャ '{fixture_name}' のメモリ使用量が高いです。軽量化を検討してください")
        
        return {
            "suggestions": suggestions,
            "optimized_fixtures": fixture_usage
        }
    
    def optimize_parallel_execution(self, test_count: int, cpu_count: int = None) -> Dict:
        """並列実行の最適化"""
        if cpu_count is None:
            cpu_count = multiprocessing.cpu_count()
        
        optimal_workers = min(cpu_count, test_count)
        
        return {
            "optimal_workers": optimal_workers,
            "cpu_count": cpu_count,
            "test_count": test_count,
            "estimated_improvement": f"{(test_count / optimal_workers) / test_count * 100:.1f}% の実行時間短縮が期待できます"
        }
    
    def generate_optimization_report(self) -> Dict:
        """最適化レポート生成"""
        performance_report = self.profiler.generate_performance_report()
        
        return {
            "performance_metrics": performance_report,
            "optimization_suggestions": self.optimization_suggestions,
            "recommendations": {
                "parallel_execution": self.optimize_parallel_execution(len(self.profiler.metrics)),
                "memory_optimization": len(self.profiler.get_memory_intensive_tests()) > 0,
                "slow_test_optimization": len(self.profiler.get_slow_tests()) > 0
            }
        }

class CachingManager:
    """キャッシュ管理システム"""
    
    def __init__(self, cache_dir: str = "test_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def get_cache_key(self, test_name: str, test_data: Dict) -> str:
        """キャッシュキー生成"""
        import hashlib
        data_str = json.dumps(test_data, sort_keys=True)
        return hashlib.md5(f"{test_name}_{data_str}".encode()).hexdigest()
    
    def get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """キャッシュされた結果を取得"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"キャッシュ読み込みエラー: {e}")
        return None
    
    def cache_result(self, cache_key: str, result: Dict):
        """結果をキャッシュ"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"キャッシュ保存エラー: {e}")
    
    def clear_cache(self):
        """キャッシュをクリア"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        self.logger.info("キャッシュをクリアしました")

class TestParallelExecutor:
    """テスト並列実行システム"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.logger = logging.getLogger(__name__)
        
    async def execute_tests_parallel(self, test_functions: List[Callable], test_names: List[str]) -> List[Dict]:
        """テストを並列実行"""
        results = []
        
        # ThreadPoolExecutorを使用して並列実行
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # テスト関数を並列実行
            future_to_test = {
                executor.submit(self._execute_single_test, func, name): name 
                for func, name in zip(test_functions, test_names)
            }
            
            for future in future_to_test:
                test_name = future_to_test[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"テスト '{test_name}' 実行エラー: {e}")
                    results.append({"test_name": test_name, "error": str(e)})
        
        return results
    
    def _execute_single_test(self, test_func: Callable, test_name: str) -> Dict:
        """単一テスト実行"""
        start_time = time.time()
        
        try:
            result = test_func()
            execution_time = time.time() - start_time
            
            return {
                "test_name": test_name,
                "success": True,
                "execution_time": execution_time,
                "result": result
            }
        except Exception as e:
            execution_time = time.time() - start_time
            
            return {
                "test_name": test_name,
                "success": False,
                "execution_time": execution_time,
                "error": str(e)
            }

def main():
    """メイン実行関数"""
    print("🚀 テストパフォーマンス最適化システム起動")
    
    # 最適化システム初期化
    optimizer = TestOptimizer()
    profiler = PerformanceProfiler()
    cache_manager = CachingManager()
    parallel_executor = TestParallelExecutor()
    
    # サンプルテスト関数
    def sample_test_1():
        time.sleep(0.5)
        return {"result": "test1"}
    
    def sample_test_2():
        time.sleep(1.0)
        return {"result": "test2"}
    
    # パフォーマンスプロファイリング
    print("📊 パフォーマンスプロファイリング実行中...")
    profiler.profile_test(sample_test_1, "sample_test_1")
    profiler.profile_test(sample_test_2, "sample_test_2")
    
    # 最適化レポート生成
    print("📈 最適化レポート生成中...")
    optimization_report = optimizer.generate_optimization_report()
    
    # 結果表示
    print("\n" + "="*50)
    print("📊 パフォーマンス最適化レポート")
    print("="*50)
    
    summary = optimization_report["performance_metrics"]["summary"]
    print(f"✅ 総テスト数: {summary['total_tests']}")
    print(f"⏱️ 総実行時間: {summary['total_execution_time']:.2f}秒")
    print(f"📊 平均実行時間: {summary['average_execution_time']:.2f}秒")
    print(f"💾 平均メモリ使用量: {summary['average_memory_usage_mb']:.1f}MB")
    
    # 遅いテストの表示
    slow_tests = optimization_report["performance_metrics"]["slow_tests"]
    if slow_tests:
        print(f"\n🐌 遅いテスト ({len(slow_tests)}件):")
        for test in slow_tests:
            print(f"  - {test['test_name']}: {test['execution_time']:.2f}秒")
    
    # 推奨事項の表示
    recommendations = optimization_report["recommendations"]
    if recommendations["parallel_execution"]["optimal_workers"] > 1:
        print(f"\n⚡ 並列実行推奨: {recommendations['parallel_execution']['optimal_workers']}ワーカー")
        print(f"   期待される改善: {recommendations['parallel_execution']['estimated_improvement']}")

if __name__ == "__main__":
    main() 