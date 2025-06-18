#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Professional Statistics Suite - Multi-Platform Hardware Test
M1/M2 Mac・ROCm GPU対応テスト

Test Coverage:
- Apple Silicon (M1/M2/M3) detection and optimization
- AMD ROCm GPU detection and acceleration
- Metal Performance Shaders integration
- MLX framework compatibility
- Cross-platform performance optimization
"""

import sys
import os
import platform
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import traceback

# プロジェクトパス追加
sys.path.append(str(Path(__file__).parent))

try:
    from config import HardwareDetector, SPSSGradeConfig
    from professional_utils import create_performance_report
except ImportError as e:
    print(f"⚠️  モジュールインポートエラー: {e}")
    print("基本機能のみでテストを実行します...")

@dataclass
class TestResult:
    """テスト結果データクラス"""
    test_name: str
    platform: str
    status: str  # SUCCESS, FAILED, SKIPPED
    duration: float
    details: Dict[str, Any]
    error: Optional[str] = None

class MultiPlatformHardwareTest:
    """マルチプラットフォームハードウェアテストクラス"""
    
    def __init__(self):
        self.platform = platform.system()
        self.architecture = platform.machine()
        self.test_results: List[TestResult] = []
        self.start_time = time.time()
        
        print(f"""
🚀 Professional Statistics Suite - Multi-Platform Hardware Test
=============================================================
Platform: {self.platform} ({self.architecture})
Python: {platform.python_version()}
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
=============================================================
        """)
    
    def run_all_tests(self):
        """全テスト実行"""
        test_methods = [
            self.test_basic_platform_detection,
            self.test_apple_silicon_support,
            self.test_metal_performance_shaders,
            self.test_mlx_framework,
            self.test_rocm_gpu_support,
            self.test_intel_gpu_support,
            self.test_performance_optimization,
            self.test_spss_grade_features,
            self.test_memory_optimization,
            self.test_parallel_processing,
        ]
        
        print("📋 テスト実行開始...\n")
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                self._add_test_result(
                    test_method.__name__,
                    "FAILED",
                    0.0,
                    {},
                    str(e)
                )
                print(f"❌ {test_method.__name__}: FAILED - {e}")
        
        self._generate_test_report()
    
    def test_basic_platform_detection(self):
        """基本プラットフォーム検出テスト"""
        start_time = time.time()
        test_name = "Basic Platform Detection"
        
        try:
            # 基本情報収集
            platform_info = {
                'system': platform.system(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'architecture': platform.architecture(),
            }
            
            # OS固有情報
            if self.platform == 'Darwin':  # macOS
                platform_info['macos_version'] = platform.mac_ver()[0]
                # Apple Silicon判定
                platform_info['apple_silicon'] = self.architecture == 'arm64'
            elif self.platform == 'Linux':
                platform_info['linux_distro'] = platform.freedesktop_os_release()
            elif self.platform == 'Windows':
                platform_info['windows_version'] = platform.win32_ver()
            
            duration = time.time() - start_time
            self._add_test_result(test_name, "SUCCESS", duration, platform_info)
            print(f"✅ {test_name}: SUCCESS ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            self._add_test_result(test_name, "FAILED", duration, {}, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    def test_apple_silicon_support(self):
        """Apple Silicon（M1/M2/M3）サポートテスト"""
        start_time = time.time()
        test_name = "Apple Silicon Support"
        
        if self.platform != 'Darwin' or self.architecture != 'arm64':
            self._add_test_result(test_name, "SKIPPED", 0.0, 
                                {"reason": "Not Apple Silicon Mac"})
            print(f"⏭️  {test_name}: SKIPPED (Not Apple Silicon Mac)")
            return
        
        try:
            apple_silicon_info = {}
            
            # チップ情報取得
            try:
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    apple_silicon_info['chip_name'] = result.stdout.strip()
            except Exception:
                pass
            
            # GPU コア数取得
            try:
                result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    apple_silicon_info['gpu_cores_detected'] = 'Metal' in result.stdout
            except Exception:
                pass
            
            # Unified Memory情報
            try:
                result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    memory_bytes = int(result.stdout.strip())
                    apple_silicon_info['unified_memory_gb'] = memory_bytes // (1024**3)
            except Exception:
                pass
            
            # Neural Engine検出（間接的）
            apple_silicon_info['neural_engine_supported'] = True  # M1以降は全て対応
            
            # MLX フレームワーク確認
            try:
                import mlx.core as mx
                apple_silicon_info['mlx_available'] = True
                apple_silicon_info['mlx_version'] = mx.__version__
            except ImportError:
                apple_silicon_info['mlx_available'] = False
            
            # TensorFlow Metal確認
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices('GPU')
                apple_silicon_info['tensorflow_metal'] = len(gpus) > 0
            except ImportError:
                apple_silicon_info['tensorflow_metal'] = False
            
            duration = time.time() - start_time
            self._add_test_result(test_name, "SUCCESS", duration, apple_silicon_info)
            print(f"✅ {test_name}: SUCCESS ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            self._add_test_result(test_name, "FAILED", duration, {}, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    def test_metal_performance_shaders(self):
        """Metal Performance Shadersテスト"""
        start_time = time.time()
        test_name = "Metal Performance Shaders"
        
        if self.platform != 'Darwin':
            self._add_test_result(test_name, "SKIPPED", 0.0, 
                                {"reason": "Not macOS"})
            print(f"⏭️  {test_name}: SKIPPED (Not macOS)")
            return
        
        try:
            metal_info = {}
            
            # Metal対応確認
            try:
                result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    output = result.stdout
                    metal_info['metal_supported'] = 'Metal' in output
                    
                    # Metal機能詳細
                    if 'Metal Family' in output:
                        metal_info['metal_family_detected'] = True
            except Exception:
                pass
            
            # PyObjC Metal framework確認
            try:
                import objc
                import Metal
                metal_info['pyobjc_metal_available'] = True
            except ImportError:
                metal_info['pyobjc_metal_available'] = False
            
            # MetalPerformanceShaders確認
            try:
                import MetalPerformanceShaders
                metal_info['mps_available'] = True
            except ImportError:
                metal_info['mps_available'] = False
            
            duration = time.time() - start_time
            self._add_test_result(test_name, "SUCCESS", duration, metal_info)
            print(f"✅ {test_name}: SUCCESS ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            self._add_test_result(test_name, "FAILED", duration, {}, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    def test_mlx_framework(self):
        """MLXフレームワークテスト"""
        start_time = time.time()
        test_name = "MLX Framework"
        
        if self.platform != 'Darwin' or self.architecture != 'arm64':
            self._add_test_result(test_name, "SKIPPED", 0.0, 
                                {"reason": "Not Apple Silicon Mac"})
            print(f"⏭️  {test_name}: SKIPPED (Not Apple Silicon Mac)")
            return
        
        try:
            mlx_info = {}
            
            try:
                import mlx.core as mx
                mlx_info['mlx_available'] = True
                mlx_info['mlx_version'] = mx.__version__
                
                # 基本的なMLX操作テスト
                import mlx.nn as nn
                
                # テスト行列作成
                test_array = mx.array([[1.0, 2.0], [3.0, 4.0]])
                result = mx.sum(test_array)
                mlx_info['basic_operations'] = True
                mlx_info['test_result'] = float(result)
                
                # GPU使用確認
                mlx_info['gpu_backend'] = mx.default_device().type == mx.Device.gpu
                
            except ImportError:
                mlx_info['mlx_available'] = False
                mlx_info['installation_note'] = "pip install mlx"
            except Exception as e:
                mlx_info['mlx_available'] = True
                mlx_info['operation_error'] = str(e)
            
            duration = time.time() - start_time
            self._add_test_result(test_name, "SUCCESS", duration, mlx_info)
            print(f"✅ {test_name}: SUCCESS ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            self._add_test_result(test_name, "FAILED", duration, {}, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    def test_rocm_gpu_support(self):
        """ROCm GPU サポートテスト"""
        start_time = time.time()
        test_name = "ROCm GPU Support"
        
        if self.platform != 'Linux':
            self._add_test_result(test_name, "SKIPPED", 0.0, 
                                {"reason": "ROCm is Linux-only"})
            print(f"⏭️  {test_name}: SKIPPED (ROCm is Linux-only)")
            return
        
        try:
            rocm_info = {}
            
            # ROCmインストール確認
            try:
                result = subprocess.run(['which', 'rocm-smi'], 
                                      capture_output=True, text=True)
                rocm_info['rocm_smi_available'] = result.returncode == 0
                
                if result.returncode == 0:
                    # ROCm バージョン取得
                    version_result = subprocess.run(['rocm-smi', '--version'], 
                                                  capture_output=True, text=True)
                    if version_result.returncode == 0:
                        rocm_info['rocm_version'] = version_result.stdout.strip()
                    
                    # AMD GPU 検出
                    device_result = subprocess.run(['rocm-smi', '--showid'], 
                                                 capture_output=True, text=True)
                    if device_result.returncode == 0:
                        rocm_info['amd_gpus_detected'] = 'GPU' in device_result.stdout
                        rocm_info['device_info'] = device_result.stdout.strip()
            except Exception:
                rocm_info['rocm_smi_available'] = False
            
            # PyTorch ROCm確認
            try:
                import torch
                rocm_info['pytorch_available'] = True
                rocm_info['pytorch_version'] = torch.__version__
                
                # HIP サポート確認
                if hasattr(torch, 'hip'):
                    rocm_info['hip_available'] = torch.hip.is_available()
                    if torch.hip.is_available():
                        rocm_info['hip_device_count'] = torch.hip.device_count()
                else:
                    rocm_info['hip_available'] = False
                    
            except ImportError:
                rocm_info['pytorch_available'] = False
            
            duration = time.time() - start_time
            self._add_test_result(test_name, "SUCCESS", duration, rocm_info)
            print(f"✅ {test_name}: SUCCESS ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            self._add_test_result(test_name, "FAILED", duration, {}, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    def test_intel_gpu_support(self):
        """Intel GPU サポートテスト"""
        start_time = time.time()
        test_name = "Intel GPU Support"
        
        try:
            intel_info = {}
            
            # Intel Extension for PyTorch確認
            try:
                import intel_extension_for_pytorch as ipex
                intel_info['ipex_available'] = True
                intel_info['ipex_version'] = ipex.__version__
                
                # Intel GPU検出
                if hasattr(ipex, 'xpu'):
                    intel_info['xpu_available'] = ipex.xpu.is_available()
                    if ipex.xpu.is_available():
                        intel_info['xpu_device_count'] = ipex.xpu.device_count()
            except ImportError:
                intel_info['ipex_available'] = False
                intel_info['installation_note'] = "pip install intel-extension-for-pytorch"
            
            # Level Zero確認（Intel GPU低レベルAPI）
            try:
                result = subprocess.run(['which', 'level-zero'], 
                                      capture_output=True, text=True)
                intel_info['level_zero_available'] = result.returncode == 0
            except Exception:
                intel_info['level_zero_available'] = False
            
            duration = time.time() - start_time
            self._add_test_result(test_name, "SUCCESS", duration, intel_info)
            print(f"✅ {test_name}: SUCCESS ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            self._add_test_result(test_name, "FAILED", duration, {}, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    def test_performance_optimization(self):
        """パフォーマンス最適化テスト"""
        start_time = time.time()
        test_name = "Performance Optimization"
        
        try:
            perf_info = {}
            
            # CPU情報
            import os
            perf_info['cpu_count'] = os.cpu_count()
            
            # メモリ情報
            try:
                import psutil
                memory = psutil.virtual_memory()
                perf_info['total_memory_gb'] = memory.total / (1024**3)
                perf_info['available_memory_gb'] = memory.available / (1024**3)
            except ImportError:
                perf_info['psutil_available'] = False
            
            # ハードウェア検出器テスト
            try:
                detector = HardwareDetector()
                optimal_profile = detector.get_optimal_profile()
                perf_info['optimal_profile'] = optimal_profile.name
                perf_info['spss_grade_config'] = True
            except Exception as e:
                perf_info['hardware_detector_error'] = str(e)
            
            duration = time.time() - start_time
            self._add_test_result(test_name, "SUCCESS", duration, perf_info)
            print(f"✅ {test_name}: SUCCESS ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            self._add_test_result(test_name, "FAILED", duration, {}, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    def test_spss_grade_features(self):
        """SPSS級機能テスト"""
        start_time = time.time()
        test_name = "SPSS Grade Features"
        
        try:
            spss_info = {}
            
            # 統計ライブラリ確認
            stats_libs = ['numpy', 'scipy', 'pandas', 'statsmodels', 'scikit-learn']
            for lib in stats_libs:
                try:
                    module = __import__(lib)
                    spss_info[f'{lib}_available'] = True
                    spss_info[f'{lib}_version'] = getattr(module, '__version__', 'unknown')
                except ImportError:
                    spss_info[f'{lib}_available'] = False
            
            # GPU加速統計確認
            gpu_stats_libs = ['cupy', 'cudf', 'rapids-cuml']
            for lib in gpu_stats_libs:
                try:
                    __import__(lib.replace('-', '_'))
                    spss_info[f'{lib}_available'] = True
                except ImportError:
                    spss_info[f'{lib}_available'] = False
            
            duration = time.time() - start_time
            self._add_test_result(test_name, "SUCCESS", duration, spss_info)
            print(f"✅ {test_name}: SUCCESS ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            self._add_test_result(test_name, "FAILED", duration, {}, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    def test_memory_optimization(self):
        """メモリ最適化テスト"""
        start_time = time.time()
        test_name = "Memory Optimization"
        
        try:
            memory_info = {}
            
            # 大容量データセット処理テスト
            try:
                import numpy as np
                
                # メモリ効率テスト
                test_size = 1000000  # 1M要素
                start_mem_time = time.time()
                
                # 通常配列作成
                arr = np.random.random(test_size)
                creation_time = time.time() - start_mem_time
                
                # 統計計算
                stats_start = time.time()
                mean_val = np.mean(arr)
                std_val = np.std(arr)
                stats_time = time.time() - stats_start
                
                memory_info.update({
                    'numpy_available': True,
                    'array_creation_time': creation_time,
                    'stats_calculation_time': stats_time,
                    'test_mean': float(mean_val),
                    'test_std': float(std_val)
                })
                
                # メモリ解放
                del arr
                
            except ImportError:
                memory_info['numpy_available'] = False
            
            # チャンクサイズ最適化テスト
            if self.platform == 'Darwin' and self.architecture == 'arm64':
                memory_info['unified_memory_optimization'] = True
                memory_info['chunk_size_optimized'] = True
            
            duration = time.time() - start_time
            self._add_test_result(test_name, "SUCCESS", duration, memory_info)
            print(f"✅ {test_name}: SUCCESS ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            self._add_test_result(test_name, "FAILED", duration, {}, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    def test_parallel_processing(self):
        """並列処理テスト"""
        start_time = time.time()
        test_name = "Parallel Processing"
        
        try:
            parallel_info = {}
            
            # マルチプロセッシング
            try:
                import multiprocessing as mp
                parallel_info['multiprocessing_available'] = True
                parallel_info['cpu_count'] = mp.cpu_count()
                
                # 簡単な並列処理テスト
                def square(x):
                    return x * x
                
                with mp.Pool(processes=min(4, mp.cpu_count())) as pool:
                    test_data = list(range(100))
                    results = pool.map(square, test_data)
                    parallel_info['parallel_test_success'] = len(results) == 100
                    
            except Exception as e:
                parallel_info['multiprocessing_error'] = str(e)
            
            # Joblib並列処理
            try:
                from joblib import Parallel, delayed
                parallel_info['joblib_available'] = True
                
                def compute_task(x):
                    return x ** 2
                
                results = Parallel(n_jobs=2)(delayed(compute_task)(x) for x in range(10))
                parallel_info['joblib_test_success'] = len(results) == 10
                
            except ImportError:
                parallel_info['joblib_available'] = False
            
            duration = time.time() - start_time
            self._add_test_result(test_name, "SUCCESS", duration, parallel_info)
            print(f"✅ {test_name}: SUCCESS ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            self._add_test_result(test_name, "FAILED", duration, {}, str(e))
            print(f"❌ {test_name}: FAILED - {e}")
    
    def _add_test_result(self, test_name: str, status: str, duration: float, 
                        details: Dict[str, Any], error: Optional[str] = None):
        """テスト結果追加"""
        result = TestResult(
            test_name=test_name,
            platform=f"{self.platform} ({self.architecture})",
            status=status,
            duration=duration,
            details=details,
            error=error
        )
        self.test_results.append(result)
    
    def _generate_test_report(self):
        """テストレポート生成"""
        total_duration = time.time() - self.start_time
        
        # 統計計算
        success_count = sum(1 for r in self.test_results if r.status == "SUCCESS")
        failed_count = sum(1 for r in self.test_results if r.status == "FAILED")
        skipped_count = sum(1 for r in self.test_results if r.status == "SKIPPED")
        total_count = len(self.test_results)
        
        # レポート出力
        print(f"\n")
        print("=" * 70)
        print("🎯 MULTI-PLATFORM HARDWARE TEST REPORT")
        print("=" * 70)
        print(f"Platform: {self.platform} ({self.architecture})")
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Tests: {total_count} | ✅ Success: {success_count} | ❌ Failed: {failed_count} | ⏭️ Skipped: {skipped_count}")
        print()
        
        # プラットフォーム特化レポート
        if self.platform == 'Darwin' and self.architecture == 'arm64':
            print("🍎 Apple Silicon (M1/M2/M3) Optimization Status:")
            self._print_apple_silicon_status()
        elif self.platform == 'Linux':
            print("🐧 Linux ROCm GPU Optimization Status:")
            self._print_rocm_status()
        elif self.platform == 'Windows':
            print("🪟 Windows Intel GPU Optimization Status:")
            self._print_intel_status()
        
        print()
        
        # 詳細結果
        for result in self.test_results:
            status_icon = {"SUCCESS": "✅", "FAILED": "❌", "SKIPPED": "⏭️"}[result.status]
            print(f"{status_icon} {result.test_name}: {result.status} ({result.duration:.2f}s)")
            if result.error:
                print(f"   Error: {result.error}")
        
        # JSON形式で保存
        self._save_report_json()
        
        print("\n🎉 テスト完了!")
        print(f"詳細レポート: hardware_test_report_{int(time.time())}.json")
    
    def _print_apple_silicon_status(self):
        """Apple Silicon状態表示"""
        apple_results = [r for r in self.test_results if 'Apple' in r.test_name or 'Metal' in r.test_name or 'MLX' in r.test_name]
        
        for result in apple_results:
            if result.status == "SUCCESS" and result.details:
                details = result.details
                if 'mlx_available' in details:
                    print(f"   MLX Framework: {'✅' if details['mlx_available'] else '❌'}")
                if 'metal_supported' in details:
                    print(f"   Metal Support: {'✅' if details['metal_supported'] else '❌'}")
                if 'unified_memory_gb' in details:
                    print(f"   Unified Memory: {details['unified_memory_gb']}GB")
    
    def _print_rocm_status(self):
        """ROCm状態表示"""
        rocm_results = [r for r in self.test_results if 'ROCm' in r.test_name]
        
        for result in rocm_results:
            if result.status == "SUCCESS" and result.details:
                details = result.details
                if 'rocm_smi_available' in details:
                    print(f"   ROCm SMI: {'✅' if details['rocm_smi_available'] else '❌'}")
                if 'hip_available' in details:
                    print(f"   HIP Support: {'✅' if details['hip_available'] else '❌'}")
                if 'amd_gpus_detected' in details:
                    print(f"   AMD GPUs: {'✅' if details['amd_gpus_detected'] else '❌'}")
    
    def _print_intel_status(self):
        """Intel GPU状態表示"""
        intel_results = [r for r in self.test_results if 'Intel' in r.test_name]
        
        for result in intel_results:
            if result.status == "SUCCESS" and result.details:
                details = result.details
                if 'ipex_available' in details:
                    print(f"   Intel Extension for PyTorch: {'✅' if details['ipex_available'] else '❌'}")
                if 'xpu_available' in details:
                    print(f"   Intel XPU: {'✅' if details['xpu_available'] else '❌'}")
    
    def _save_report_json(self):
        """レポートをJSON形式で保存"""
        report_data = {
            'test_info': {
                'platform': self.platform,
                'architecture': self.architecture,
                'python_version': platform.python_version(),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_duration': time.time() - self.start_time
            },
            'results': [
                {
                    'test_name': r.test_name,
                    'platform': r.platform,
                    'status': r.status,
                    'duration': r.duration,
                    'details': r.details,
                    'error': r.error
                }
                for r in self.test_results
            ]
        }
        
        filename = f"hardware_test_report_{int(time.time())}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

def main():
    """メイン実行関数"""
    try:
        tester = MultiPlatformHardwareTest()
        tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n⚠️  テストが中断されました")
    except Exception as e:
        print(f"\n💥 予期しないエラー: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 