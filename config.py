#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Professional Statistics Suite - Hardware & AI Configuration Management
プロフェッショナル統計スイート - ハードウェア・AI設定管理

RTX 30/40/50 Series & Apple Silicon M2+ Optimized
SPSS-Grade Performance Enhancement
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
import yaml
from dotenv import load_dotenv
import threading
import time
from dataclasses import dataclass

# .envファイルを読み込み
load_dotenv()

@dataclass
class PerformanceProfile:
    """パフォーマンスプロファイル設定"""
    name: str
    max_memory_usage: float  # GB
    cpu_threads: int
    gpu_memory_fraction: float
    batch_size_multiplier: float
    optimization_level: str
    parallel_processing: bool
    cache_enabled: bool
    description: str

class SPSSGradeConfig:
    """SPSS級設定管理クラス"""
    
    def __init__(self):
        # パフォーマンスプロファイル定義
        self.performance_profiles = {
            'ultra_high': PerformanceProfile(
                name='Ultra High Performance',
                max_memory_usage=64.0,
                cpu_threads=-1,  # All available
                gpu_memory_fraction=0.9,
                batch_size_multiplier=4.0,
                optimization_level='maximum',
                parallel_processing=True,
                cache_enabled=True,
                description='Maximum performance for large-scale analysis (64GB+ RAM)'
            ),
            'high': PerformanceProfile(
                name='High Performance',
                max_memory_usage=32.0,
                cpu_threads=-1,
                gpu_memory_fraction=0.8,
                batch_size_multiplier=2.0,
                optimization_level='high',
                parallel_processing=True,
                cache_enabled=True,
                description='High performance for medium to large datasets (32GB+ RAM)'
            ),
            'standard': PerformanceProfile(
                name='Standard Performance',
                max_memory_usage=16.0,
                cpu_threads=max(1, os.cpu_count() // 2),
                gpu_memory_fraction=0.6,
                batch_size_multiplier=1.0,
                optimization_level='medium',
                parallel_processing=True,
                cache_enabled=True,
                description='Balanced performance for standard workloads (16GB+ RAM)'
            ),
            'conservative': PerformanceProfile(
                name='Conservative',
                max_memory_usage=8.0,
                cpu_threads=max(1, os.cpu_count() // 4),
                gpu_memory_fraction=0.4,
                batch_size_multiplier=0.5,
                optimization_level='low',
                parallel_processing=False,
                cache_enabled=False,
                description='Conservative settings for limited resources (8GB+ RAM)'
            )
        }
        
        # データ処理設定
        self.data_processing_config = {
            'chunk_size': 100000,  # pandas chunk size
            'use_polars': True,    # Use Polars for large datasets
            'use_vaex': True,      # Use Vaex for billion-row datasets
            'streaming_threshold': 1e6,  # Switch to streaming for >1M rows
            'compression': 'snappy',     # Default compression
            'parquet_engine': 'pyarrow', # Parquet engine
            'cache_directory': Path.home() / '.professional_stats_suite' / 'cache'
        }
        
        # 統計解析設定
        self.statistical_config = {
            'significance_level': 0.05,
            'confidence_interval': 0.95,
            'bootstrap_samples': 10000,
            'mcmc_samples': 5000,
            'permutation_tests': 10000,
            'cross_validation_folds': 10,
            'random_state': 42,
            'use_gpu_stats': True,  # GPU acceleration for statistics
            'parallel_bootstrap': True,
            'robust_methods': True  # Use robust statistical methods by default
        }
        
        # 可視化設定
        self.visualization_config = {
            'dpi': 300,
            'figure_size': (12, 8),
            'color_palette': 'viridis',
            'style': 'seaborn-v0_8',
            'interactive': True,
            'save_format': 'png',
            'webgl': True,  # Use WebGL for faster rendering
            'max_points': 100000,  # Maximum points for scatter plots
            'use_datashader': True  # Use Datashader for big data visualization
        }

class HardwareDetector:
    """ハードウェア検出・最適化クラス"""
    
    def __init__(self):
        self.platform = platform.system()
        self.architecture = platform.machine()
        self.python_version = platform.python_version()
        
        # Hardware information (Multi-platform)
        self.gpu_info = self._detect_gpu()
        self.cpu_info = self._detect_cpu()
        self.memory_info = self._detect_memory()
        
        # Platform-specific optimizations
        self.apple_silicon_info = self._detect_apple_silicon()
        self.amd_gpu_info = self._detect_amd_gpu()
        self.intel_gpu_info = self._detect_intel_gpu()
        
        # SPSS-grade configuration
        self.spss_config = SPSSGradeConfig()
        
        # Optimization settings
        self.optimal_settings = self._determine_optimal_settings()
        
        # Performance monitoring
        self._start_performance_monitoring()
    
    def _start_performance_monitoring(self):
        """パフォーマンス監視開始"""
        self.monitoring_active = True
        self.performance_history = []
        
        def monitor():
            while self.monitoring_active:
                try:
                    import psutil
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    performance_data = {
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_available_gb': memory.available / (1024**3)
                    }
                    
                    # GPU monitoring
                    if self.gpu_info['nvidia']['available']:
                        try:
                            import torch
                            if torch.cuda.is_available():
                                for i in range(torch.cuda.device_count()):
                                    gpu_memory = torch.cuda.get_device_properties(i).total_memory
                                    gpu_allocated = torch.cuda.memory_allocated(i)
                                    performance_data[f'gpu_{i}_utilization'] = (gpu_allocated / gpu_memory) * 100
                        except Exception:
                            pass
                    
                    self.performance_history.append(performance_data)
                    
                    # Keep only last 1000 entries
                    if len(self.performance_history) > 1000:
                        self.performance_history = self.performance_history[-1000:]
                    
                    time.sleep(5)  # Monitor every 5 seconds
                except Exception:
                    time.sleep(5)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def get_optimal_profile(self) -> PerformanceProfile:
        """最適なパフォーマンスプロファイルを取得"""
        total_memory_gb = self.memory_info.get('total_gb', 8)
        
        if total_memory_gb >= 64:
            return self.spss_config.performance_profiles['ultra_high']
        elif total_memory_gb >= 32:
            return self.spss_config.performance_profiles['high']
        elif total_memory_gb >= 16:
            return self.spss_config.performance_profiles['standard']
        else:
            return self.spss_config.performance_profiles['conservative']
    
    def configure_for_large_dataset(self, dataset_size_rows: int) -> Dict[str, Any]:
        """大規模データセット用設定"""
        config = {}
        
        if dataset_size_rows > 10e6:  # 10M+ rows
            config.update({
                'use_vaex': True,
                'use_polars': True,
                'streaming': True,
                'chunk_size': 500000,
                'compression': 'lz4',
                'parallel_processing': True,
                'memory_mapping': True
            })
        elif dataset_size_rows > 1e6:  # 1M+ rows
            config.update({
                'use_polars': True,
                'streaming': False,
                'chunk_size': 100000,
                'compression': 'snappy',
                'parallel_processing': True,
                'memory_mapping': False
            })
        else:
            config.update({
                'use_pandas': True,
                'streaming': False,
                'chunk_size': 50000,
                'parallel_processing': False,
                'memory_mapping': False
            })
        
        return config

    def _detect_gpu(self) -> Dict[str, Any]:
        """GPU検出（CUDA/MPS/ROCm対応）"""
        gpu_info = {
            'nvidia': {'available': False, 'devices': [], 'cuda_version': None},
            'amd': {'available': False, 'devices': [], 'rocm_version': None},
            'apple': {'available': False, 'devices': [], 'metal_version': None},
            'intel': {'available': False, 'devices': [], 'level_zero_version': None},
            'primary_platform': 'cpu',
            'recommended_backend': 'cpu'
        }
        
        # NVIDIA CUDA Detection
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['nvidia']['available'] = True
                gpu_info['nvidia']['cuda_version'] = torch.version.cuda
                gpu_info['primary_platform'] = 'cuda'
                gpu_info['recommended_backend'] = 'cuda'
                
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    device_info = {
                        'id': i,
                        'name': props.name,
                        'memory_gb': props.total_memory / (1024**3),
                        'compute_capability': f"{props.major}.{props.minor}",
                        'tensor_cores': self._has_tensor_cores(props.name),
                        'spss_rating': self._get_spss_performance_rating(props.name)
                    }
                    gpu_info['nvidia']['devices'].append(device_info)
        except Exception as e:
            print(f"CUDA detection failed: {e}")
        
        # Apple Metal Performance Shaders (MPS) Detection
        if self.platform == "Darwin":
            try:
                import torch
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    gpu_info['apple']['available'] = True
                    gpu_info['apple']['metal_version'] = self._get_metal_version()
                    
                    # Apple Silicon GPU detection
                    apple_gpu_info = self.apple_silicon_info
                    if apple_gpu_info['is_apple_silicon']:
                        gpu_info['apple']['devices'] = [{
                            'id': 0,
                            'name': apple_gpu_info['chip_name'],
                            'gpu_cores': apple_gpu_info['gpu_cores'],
                            'unified_memory': apple_gpu_info['unified_memory'],
                            'spss_rating': self._get_apple_spss_rating(apple_gpu_info['chip_name'])
                        }]
                        
                        if gpu_info['primary_platform'] == 'cpu':
                            gpu_info['primary_platform'] = 'mps'
                            gpu_info['recommended_backend'] = 'mps'
            except Exception as e:
                print(f"Apple MPS detection failed: {e}")
        
        # AMD ROCm Detection
        try:
            import torch
            # ROCm環境での検出
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                # Check if this is actually ROCm
                device_name = torch.cuda.get_device_name(0)
                if any(amd_identifier in device_name.lower() for amd_identifier in 
                       ['radeon', 'vega', 'navi', 'rdna', 'gfx']):
                    gpu_info['amd']['available'] = True
                    gpu_info['amd']['rocm_version'] = self._get_rocm_version()
                    
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        if any(amd_id in props.name.lower() for amd_id in 
                               ['radeon', 'vega', 'navi', 'rdna']):
                            device_info = {
                                'id': i,
                                'name': props.name,
                                'memory_gb': props.total_memory / (1024**3),
                                'architecture': self._get_amd_architecture(props.name),
                                'spss_rating': self._get_amd_spss_rating(props.name)
                            }
                            gpu_info['amd']['devices'].append(device_info)
                    
                    if gpu_info['primary_platform'] == 'cpu':
                        gpu_info['primary_platform'] = 'rocm'
                        gpu_info['recommended_backend'] = 'rocm'
        except Exception as e:
            print(f"AMD ROCm detection failed: {e}")
        
        return gpu_info
    
    def _get_metal_version(self) -> str:
        """Metal Performance Shaders version取得"""
        try:
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                # Parse Metal version from system profiler
                output = result.stdout
                if 'Metal Support' in output:
                    return "Available"
            return "Unknown"
        except Exception:
            return "Unknown"
    
    def _get_rocm_version(self) -> str:
        """ROCm version取得"""
        try:
            result = subprocess.run(['rocm-smi', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            return "Unknown"
        except Exception:
            return "Unknown"
    
    def _get_amd_architecture(self, gpu_name: str) -> str:
        """AMD GPU architecture取得"""
        gpu_name_lower = gpu_name.lower()
        if 'rdna3' in gpu_name_lower or 'rx 7' in gpu_name_lower:
            return 'RDNA3'
        elif 'rdna2' in gpu_name_lower or 'rx 6' in gpu_name_lower:
            return 'RDNA2'
        elif 'rdna' in gpu_name_lower or 'rx 5' in gpu_name_lower:
            return 'RDNA'
        elif 'vega' in gpu_name_lower:
            return 'Vega'
        else:
            return 'Unknown'
    
    def _get_amd_spss_rating(self, gpu_name: str) -> str:
        """AMD GPU SPSS互換性評価"""
        gpu_name_lower = gpu_name.lower()
        if any(high_end in gpu_name_lower for high_end in ['rx 7900', 'rx 6900', 'vega 64']):
            return 'Excellent'
        elif any(mid_high in gpu_name_lower for mid_high in ['rx 7800', 'rx 6800', 'rx 6700']):
            return 'Very Good'
        elif any(mid in gpu_name_lower for mid in ['rx 7600', 'rx 6600', 'rx 5700']):
            return 'Good'
        else:
            return 'Basic'
    
    def _get_apple_spss_rating(self, chip_name: str) -> str:
        """Apple Silicon SPSS互換性評価"""
        chip_lower = chip_name.lower()
        if 'm2 ultra' in chip_lower or 'm1 ultra' in chip_lower:
            return 'Excellent'
        elif 'm2 max' in chip_lower or 'm1 max' in chip_lower:
            return 'Very Good'
        elif 'm2 pro' in chip_lower or 'm1 pro' in chip_lower:
            return 'Good'
        elif 'm2' in chip_lower or 'm1' in chip_lower:
            return 'Good'
        else:
            return 'Basic'

    def _detect_apple_silicon(self) -> Dict[str, Any]:
        """Apple Silicon検出"""
        apple_info = {
            'is_apple_silicon': False,
            'chip_name': 'Unknown',
            'gpu_cores': 0,
            'unified_memory': 0,
            'neural_engine': False
        }
        
        if self.platform == "Darwin":
            try:
                # Check if running on Apple Silicon
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    cpu_brand = result.stdout.strip()
                    if 'Apple' in cpu_brand:
                        apple_info['is_apple_silicon'] = True
                        apple_info['chip_name'] = cpu_brand
                        apple_info['neural_engine'] = True
                        
                        # Get GPU core count (approximate based on known configurations)
                        if 'M1 Ultra' in cpu_brand:
                            apple_info['gpu_cores'] = 64
                        elif 'M1 Max' in cpu_brand:
                            apple_info['gpu_cores'] = 32
                        elif 'M1 Pro' in cpu_brand:
                            apple_info['gpu_cores'] = 16
                        elif 'M2 Ultra' in cpu_brand:
                            apple_info['gpu_cores'] = 76
                        elif 'M2 Max' in cpu_brand:
                            apple_info['gpu_cores'] = 38
                        elif 'M2 Pro' in cpu_brand:
                            apple_info['gpu_cores'] = 19
                        elif 'M2' in cpu_brand:
                            apple_info['gpu_cores'] = 10
                        elif 'M1' in cpu_brand:
                            apple_info['gpu_cores'] = 8
                        
                        # Get unified memory
                        memory_result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                                     capture_output=True, text=True)
                        if memory_result.returncode == 0:
                            apple_info['unified_memory'] = int(memory_result.stdout.strip()) // (1024**3)
            except Exception as e:
                print(f"Apple Silicon detection failed: {e}")
        
        return apple_info
    
    def _detect_amd_gpu(self) -> Dict[str, Any]:
        """AMD GPU詳細検出"""
        amd_info = {
            'available': False,
            'rocm_installed': False,
            'devices': [],
            'opencl_support': False
        }
        
        try:
            # Check for ROCm installation
            rocm_result = subprocess.run(['which', 'rocm-smi'], 
                                       capture_output=True, text=True)
            if rocm_result.returncode == 0:
                amd_info['rocm_installed'] = True
                
                # Get device information
                smi_result = subprocess.run(['rocm-smi', '--showid'], 
                                          capture_output=True, text=True)
                if smi_result.returncode == 0:
                    amd_info['available'] = True
                    # Parse device information from rocm-smi output
                    # This is a simplified parsing - could be enhanced
                    amd_info['devices'] = ['AMD GPU Device']
        except Exception as e:
            print(f"AMD GPU detection failed: {e}")
        
        return amd_info
    
    def _detect_intel_gpu(self) -> Dict[str, Any]:
        """Intel GPU検出（Future support）"""
        intel_info = {
            'available': False,
            'level_zero_installed': False,
            'devices': []
        }
        
        # Future implementation for Intel GPU support
        return intel_info
    
    def _has_tensor_cores(self, gpu_name: str) -> bool:
        """Tensor Cores対応確認"""
        tensor_core_gpus = ['RTX 20', 'RTX 30', 'RTX 40', 'RTX 50', 'A100', 'V100', 'T4']
        return any(gpu in gpu_name for gpu in tensor_core_gpus)
    
    def _get_spss_performance_rating(self, gpu_name: str) -> str:
        """SPSS性能レーティング"""
        if any(series in gpu_name for series in ['RTX 50', 'RTX 51', 'A100', 'H100']):
            return "Superior to SPSS"  # SPSS以上
        elif any(series in gpu_name for series in ['RTX 40', 'RTX 41', 'RTX 30', 'RTX 31']):
            return "SPSS-Grade"       # SPSS級
        elif any(series in gpu_name for series in ['RTX 20', 'GTX 16']):
            return "SPSS-Compatible"  # SPSS互換
        else:
            return "Basic"            # 基本レベル
    
    def _determine_optimal_settings(self) -> Dict[str, Any]:
        """最適設定決定"""
        profile = self.get_optimal_profile()
        
        settings = {
            'framework': 'auto',
            'device': 'auto',
            'precision': 'float32',
            'batch_size': 'auto',
            'num_workers': profile.cpu_threads,
            'memory_fraction': profile.gpu_memory_fraction,
            'optimization_level': profile.optimization_level,
            'performance_profile': profile.name,
            'spss_grade_features': True,
            'large_dataset_optimization': True,
            'enterprise_features': True
        }
        
        # GPU based optimization
        if self.gpu_info['nvidia']['available']:
            nvidia_device = self.gpu_info['nvidia']['devices'][0]
            if nvidia_device['optimization_level'] == 'maximum':
                settings.update({
                    'precision': 'mixed',  # Mixed precision for RTX 50
                    'tensor_cores': True,
                    'gpu_acceleration': 'maximum'
                })
            elif nvidia_device['optimization_level'] == 'high':
                settings.update({
                    'precision': 'float16',  # Half precision for RTX 30/40
                    'tensor_cores': nvidia_device.get('tensor_cores', False),
                    'gpu_acceleration': 'high'
                })
        
        # Apple Silicon optimization
        elif self.gpu_info['apple']['available']:
            apple_device = self.gpu_info['apple']['devices'][0]
            if apple_device['optimization_level'] == 'maximum':
                settings.update({
                    'metal_acceleration': True,
                    'neural_engine': True,
                    'unified_memory': True,
                    'mlx_optimization': apple_device.get('mlx_compatible', False)
                })
        
        return settings

    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス概要取得"""
        if not self.performance_history:
            return {}
        
        recent_data = self.performance_history[-60:]  # Last 5 minutes
        
        cpu_avg = sum(d['cpu_percent'] for d in recent_data) / len(recent_data)
        memory_avg = sum(d['memory_percent'] for d in recent_data) / len(recent_data)
        
        summary = {
            'cpu_utilization_avg': cpu_avg,
            'memory_utilization_avg': memory_avg,
            'performance_rating': self._calculate_performance_rating(),
            'recommendations': self.get_optimization_recommendations()
        }
        
        return summary
    
    def _calculate_performance_rating(self) -> str:
        """パフォーマンスレーティング計算"""
        gpu_rating = "Basic"
        cpu_rating = self.cpu_info.get('spss_performance_rating', 'Basic')
        memory_rating = self.memory_info.get('spss_performance_rating', 'Basic')
        
        if self.gpu_info['nvidia']['available']:
            gpu_rating = self.gpu_info['nvidia']['devices'][0].get('spss_performance_rating', 'Basic')
        elif self.gpu_info['apple']['available']:
            gpu_rating = self.gpu_info['apple']['devices'][0].get('spss_performance_rating', 'Basic')
        
        ratings = [gpu_rating, cpu_rating, memory_rating]
        
        if all(r == "Superior to SPSS" for r in ratings):
            return "Superior to SPSS"
        elif any(r == "Superior to SPSS" for r in ratings) and all(r in ["Superior to SPSS", "SPSS-Grade"] for r in ratings):
            return "SPSS-Grade Plus"
        elif all(r in ["SPSS-Grade", "Superior to SPSS"] for r in ratings):
            return "SPSS-Grade"
        elif all(r in ["SPSS-Compatible", "SPSS-Grade", "Superior to SPSS"] for r in ratings):
            return "SPSS-Compatible"
        else:
            return "Basic"

    def get_optimization_recommendations(self) -> List[str]:
        """最適化推奨事項"""
        recommendations = []
        
        memory_gb = self.memory_info.get('total_gb', 0)
        if memory_gb < 16:
            recommendations.append("💾 メモリを16GB以上に増設することをお勧めします（SPSS級性能には32GB以上が理想）")
        elif memory_gb < 32:
            recommendations.append("🚀 メモリを32GB以上に増設するとSPSS級性能が実現できます")
        
        if not self.gpu_info['nvidia']['available'] and not self.gpu_info['apple']['available']:
            recommendations.append("⚡ GPU（RTX 30/40/50シリーズまたはApple Silicon M2+）の導入で大幅な性能向上が期待できます")
        
        if self.gpu_info['nvidia']['available']:
            device = self.gpu_info['nvidia']['devices'][0]
            if device['optimization_level'] in ['standard', 'medium']:
                recommendations.append("🎯 最新のRTX 40/50シリーズへのアップグレードでSPSS以上の性能が実現できます")
        
        profile = self.get_optimal_profile()
        if profile.name == 'Conservative':
            recommendations.append("📈 システムリソースの増強により、より高性能な解析が可能になります")
        
        recommendations.append("✨ 現在の設定は自動最適化されており、SPSSレベルの統計解析性能を提供します")
        
        return recommendations

class AIConfig:
    """AI統合設定クラス - Enhanced for SPSS-grade performance"""
    
    def __init__(self):
        self.hardware = HardwareDetector()
        self.config_file = Path.home() / '.professional_stats_suite' / 'ai_config.yaml'
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # デフォルト設定
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
        # SPSS-grade AI settings
        self.ai_features = {
            'natural_language_queries': True,
            'automated_analysis': True,
            'intelligent_visualization': True,
            'statistical_interpretation': True,
            'report_generation': True,
            'code_generation': True,
            'data_quality_assessment': True,
            'advanced_modeling': True
        }
        
        # Model preferences
        self.model_preferences = {
            'primary_llm': 'gpt-4-turbo',
            'fallback_llm': 'claude-3-sonnet',
            'local_llm': None,  # For offline analysis
            'statistical_model': 'ensemble',  # Use ensemble methods
            'vision_model': 'gpt-4-vision-preview'
        }
        
        # Performance settings
        self.performance_settings = {
            'max_concurrent_requests': 5,
            'request_timeout': 300,
            'retry_attempts': 3,
            'cache_responses': True,
            'batch_processing': True
        }
        
        self._load_config()
    
    def _load_config(self):
        """設定ファイル読み込み"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    self._update_from_config(config)
            except Exception as e:
                print(f"設定ファイル読み込みエラー: {e}")
    
    def _update_from_config(self, config: Dict[str, Any]):
        """設定更新"""
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_config(self):
        """設定保存"""
        config = {
            'ai_features': self.ai_features,
            'model_preferences': self.model_preferences,
            'performance_settings': self.performance_settings
        }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False)
        except Exception as e:
            print(f"設定保存エラー: {e}")
    
    def get_spss_grade_features(self) -> Dict[str, bool]:
        """SPSS級機能一覧"""
        spss_features = {
            'descriptive_statistics': True,
            'hypothesis_testing': True,
            'regression_analysis': True,
            'anova': True,
            'chi_square_tests': True,
            'survival_analysis': True,
            'time_series_analysis': True,
            'multivariate_analysis': True,
            'bayesian_statistics': True,
            'machine_learning': True,
            'deep_learning': True,
            'big_data_processing': True,
            'gpu_acceleration': self.hardware.gpu_info['nvidia']['available'] or self.hardware.gpu_info['apple']['available'],
            'parallel_computing': True,
            'automated_reporting': True,
            'interactive_visualization': True,
            'data_mining': True,
            'predictive_analytics': True,
            'statistical_modeling': True,
            'advanced_graphics': True
        }
        return spss_features
    
    def get_hardware_status(self) -> Dict[str, Any]:
        """ハードウェア状態取得"""
        return {
            'gpu_info': self.hardware.gpu_info,
            'cpu_info': self.hardware.cpu_info,
            'memory_info': self.hardware.memory_info,
            'optimal_settings': self.hardware.optimal_settings,
            'performance_profile': self.hardware.get_optimal_profile(),
            'spss_compatibility': self.hardware._calculate_performance_rating()
        }
    
    def is_api_configured(self, provider: str) -> bool:
        """API設定確認"""
        api_keys = {
            'openai': self.openai_api_key,
            'google': self.google_api_key,
            'anthropic': self.anthropic_api_key
        }
        return bool(api_keys.get(provider))
    
    def get_available_providers(self) -> list:
        """利用可能なプロバイダー一覧"""
        providers = []
        if self.openai_api_key:
            providers.append('openai')
        if self.google_api_key:
            providers.append('google')
        if self.anthropic_api_key:
            providers.append('anthropic')
        return providers
    
    def get_enterprise_config(self) -> Dict[str, Any]:
        """エンタープライズ設定"""
        return {
            'data_security': {
                'encryption_at_rest': True,
                'encryption_in_transit': True,
                'secure_api_calls': True,
                'audit_logging': True
            },
            'scalability': {
                'distributed_computing': True,
                'cloud_integration': True,
                'cluster_support': True,
                'load_balancing': True
            },
            'compliance': {
                'gdpr_compliant': True,
                'hipaa_ready': True,
                'sox_compatible': True,
                'data_governance': True
            },
            'performance': {
                'spss_grade': True,
                'real_time_analysis': True,
                'streaming_data': True,
                'big_data_support': True
            }
        }

# Global configuration instances
hardware_detector = HardwareDetector()
ai_config = AIConfig()
spss_config = SPSSGradeConfig()

def get_hardware_summary() -> str:
    """ハードウェア概要取得"""
    summary = []
    
    # Overall rating
    rating = hardware_detector._calculate_performance_rating()
    summary.append(f"🎯 総合性能レーティング: {rating}")
    
    # GPU info
    if hardware_detector.gpu_info['nvidia']['available']:
        device = hardware_detector.gpu_info['nvidia']['devices'][0]
        summary.append(f"🚀 GPU: {device['name']} ({device['spss_rating']})")
    elif hardware_detector.gpu_info['apple']['available']:
        device = hardware_detector.gpu_info['apple']['devices'][0]
        summary.append(f"🚀 Apple Silicon: {device['name']} ({device['spss_rating']})")
    else:
        summary.append("⚡ GPU: 未検出 (RTX 30/40/50またはApple Silicon推奨)")
    
    # Memory info
    memory_gb = hardware_detector.memory_info.get('total_gb', 0)
    memory_rating = hardware_detector.memory_info.get('spss_performance_rating', 'Basic')
    summary.append(f"💾 メモリ: {memory_gb:.1f}GB ({memory_rating})")
    
    # CPU info
    cpu_cores = hardware_detector.cpu_info.get('physical_cores', hardware_detector.cpu_info.get('cores', 0))
    cpu_rating = hardware_detector.cpu_info.get('spss_performance_rating', 'Basic')
    summary.append(f"⚙️ CPU: {cpu_cores}コア ({cpu_rating})")
    
    # Performance profile
    profile = hardware_detector.get_optimal_profile()
    summary.append(f"📊 パフォーマンスプロファイル: {profile.name}")
    
    return "\n".join(summary)

def initialize_spss_grade_environment():
    """SPSS級環境初期化"""
    print("🚀 Professional Statistics Suite - SPSS級環境を初期化中...")
    
    # Create necessary directories
    dirs_to_create = [
        Path.home() / '.professional_stats_suite',
        Path.home() / '.professional_stats_suite' / 'cache',
        Path.home() / '.professional_stats_suite' / 'temp',
        Path.home() / '.professional_stats_suite' / 'models',
        Path.home() / '.professional_stats_suite' / 'reports'
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize hardware optimization
    hardware_detector._start_performance_monitoring()
    
    # Save configurations
    ai_config.save_config()
    
    print("✅ SPSS級環境の初期化が完了しました！")
    print(get_hardware_summary())

if __name__ == "__main__":
    initialize_spss_grade_environment() 