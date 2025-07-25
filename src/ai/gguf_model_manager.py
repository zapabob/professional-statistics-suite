#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GGUF Model Manager - 2025 July 25th Edition
GGUFモデル管理システム - 2025年7月25日版
"""

import os
import json
import hashlib
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import asyncio
import aiohttp
from tqdm import tqdm

@dataclass
class GGUFModelInfo:
    """GGUFモデル情報"""
    name: str
    filename: str
    size_bytes: int
    url: str
    description: str
    tags: List[str]
    quantization: str
    context_length: int
    parameters: str
    license: str
    last_updated: str
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class GGUFModelManager:
    """GGUFモデル管理システム"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.models_info_file = self.models_dir / "models_info.json"
        self.download_cache_file = self.models_dir / "download_cache.json"
        
        # 最新のGGUFモデル情報（2025年7月25日現在）
        self.available_models = {
            "llama3-8b-instruct": GGUFModelInfo(
                name="Llama 3 8B Instruct",
                filename="llama3-8b-instruct.Q8_0.gguf",
                size_bytes=8_000_000_000,  # 約8GB
                url="https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q8_0.gguf",
                description="Meta社のLlama 3 8Bパラメータモデルの指示チューニング版",
                tags=["llama3", "instruct", "8b", "quantized"],
                quantization="Q8_0",
                context_length=8192,
                parameters="8B",
                license="Meta License",
                last_updated="2024-07-25"
            ),
            "llama3-70b-instruct": GGUFModelInfo(
                name="Llama 3 70B Instruct",
                filename="llama3-70b-instruct.Q4_K_M.gguf",
                size_bytes=40_000_000_000,  # 約40GB
                url="https://huggingface.co/QuantFactory/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct.Q4_K_M.gguf",
                description="Meta社のLlama 3 70Bパラメータモデルの指示チューニング版（高精度）",
                tags=["llama3", "instruct", "70b", "quantized", "high-performance"],
                quantization="Q4_K_M",
                context_length=8192,
                parameters="70B",
                license="Meta License",
                last_updated="2024-07-25"
            ),
            "phi3-mini-128k": GGUFModelInfo(
                name="Phi-3 Mini 128K",
                filename="phi3-mini-128k-instruct.Q8_0.gguf",
                size_bytes=4_000_000_000,  # 約4GB
                url="https://huggingface.co/QuantFactory/Microsoft-Phi-3-mini-128k-instruct-GGUF/resolve/main/Microsoft-Phi-3-mini-128k-instruct.Q8_0.gguf",
                description="Microsoft社のPhi-3 Mini 128Kコンテキストモデル",
                tags=["phi3", "mini", "128k", "instruct", "quantized"],
                quantization="Q8_0",
                context_length=131072,
                parameters="3.8B",
                license="Microsoft License",
                last_updated="2024-07-25"
            ),
            "mistral-7b-instruct": GGUFModelInfo(
                name="Mistral 7B Instruct",
                filename="mistral-7b-instruct.Q8_0.gguf",
                size_bytes=7_000_000_000,  # 約7GB
                url="https://huggingface.co/QuantFactory/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/Mistral-7B-Instruct-v0.2.Q8_0.gguf",
                description="Mistral AI社の7Bパラメータ指示チューニングモデル",
                tags=["mistral", "instruct", "7b", "quantized"],
                quantization="Q8_0",
                context_length=8192,
                parameters="7B",
                license="Apache 2.0",
                last_updated="2024-07-25"
            ),
            "codestral-22b": GGUFModelInfo(
                name="Codestral 22B",
                filename="codestral-22b.Q4_K_M.gguf",
                size_bytes=12_000_000_000,  # 約12GB
                url="https://huggingface.co/QuantFactory/Codestral-22B-v0.1-GGUF/resolve/main/Codestral-22B-v0.1.Q4_K_M.gguf",
                description="コード生成特化の22Bパラメータモデル",
                tags=["codestral", "code-generation", "22b", "quantized"],
                quantization="Q4_K_M",
                context_length=16384,
                parameters="22B",
                license="Mistral License",
                last_updated="2024-07-25"
            )
        }
        
        self._load_download_cache()
    
    def _load_download_cache(self):
        """ダウンロードキャッシュ読み込み"""
        try:
            if self.download_cache_file.exists():
                with open(self.download_cache_file, 'r', encoding='utf-8') as f:
                    self.download_cache = json.load(f)
            else:
                self.download_cache = {}
        except Exception as e:
            logging.warning(f"ダウンロードキャッシュ読み込みエラー: {e}")
            self.download_cache = {}
    
    def _save_download_cache(self):
        """ダウンロードキャッシュ保存"""
        try:
            with open(self.download_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.download_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"ダウンロードキャッシュ保存エラー: {e}")
    
    def get_available_models(self) -> Dict[str, GGUFModelInfo]:
        """利用可能モデル一覧取得"""
        return self.available_models
    
    def get_model_info(self, model_name: str) -> Optional[GGUFModelInfo]:
        """モデル情報取得"""
        return self.available_models.get(model_name)
    
    def list_downloaded_models(self) -> List[str]:
        """ダウンロード済みモデル一覧"""
        downloaded = []
        for model_name, model_info in self.available_models.items():
            model_path = self.models_dir / model_info.filename
            if model_path.exists():
                downloaded.append(model_name)
        return downloaded
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """モデルファイルパス取得"""
        model_info = self.get_model_info(model_name)
        if model_info:
            model_path = self.models_dir / model_info.filename
            if model_path.exists():
                return model_path
        return None
    
    def verify_model_integrity(self, model_name: str) -> bool:
        """モデル整合性検証"""
        model_path = self.get_model_path(model_name)
        if not model_path:
            return False
        
        model_info = self.get_model_info(model_name)
        if not model_info:
            return False
        
        # ファイルサイズチェック
        actual_size = model_path.stat().st_size
        expected_size = model_info.size_bytes
        
        # 10%の許容誤差
        tolerance = expected_size * 0.1
        if abs(actual_size - expected_size) > tolerance:
            logging.warning(f"モデルサイズ不一致: {model_name}")
            return False
        
        # チェックサム検証（キャッシュから）
        if model_name in self.download_cache:
            expected_checksum = self.download_cache[model_name].get('checksum')
            if expected_checksum:
                actual_checksum = self._calculate_file_checksum(model_path)
                if actual_checksum != expected_checksum:
                    logging.warning(f"チェックサム不一致: {model_name}")
                    return False
        
        return True
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """ファイルチェックサム計算"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def download_model(self, model_name: str, progress_callback=None) -> bool:
        """モデルダウンロード（非同期）"""
        model_info = self.get_model_info(model_name)
        if not model_info:
            logging.error(f"モデル情報が見つかりません: {model_name}")
            return False
        
        model_path = self.models_dir / model_info.filename
        
        # 既にダウンロード済みの場合
        if model_path.exists() and self.verify_model_integrity(model_name):
            logging.info(f"モデルは既にダウンロード済みです: {model_name}")
            return True
        
        try:
            logging.info(f"モデルダウンロード開始: {model_name}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(model_info.url) as response:
                    if response.status != 200:
                        logging.error(f"ダウンロードエラー: {response.status}")
                        return False
                    
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(model_path, 'wb') as f:
                        downloaded_size = 0
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            if progress_callback and total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                progress_callback(progress)
            
            # ダウンロード完了後の検証
            if self.verify_model_integrity(model_name):
                # キャッシュに保存
                self.download_cache[model_name] = {
                    'downloaded_at': datetime.now().isoformat(),
                    'file_size': model_path.stat().st_size,
                    'checksum': self._calculate_file_checksum(model_path)
                }
                self._save_download_cache()
                
                logging.info(f"モデルダウンロード完了: {model_name}")
                return True
            else:
                logging.error(f"モデル整合性検証失敗: {model_name}")
                model_path.unlink(missing_ok=True)
                return False
                
        except Exception as e:
            logging.error(f"ダウンロードエラー: {model_name} - {e}")
            model_path.unlink(missing_ok=True)
            return False
    
    def download_model_sync(self, model_name: str) -> bool:
        """モデルダウンロード（同期）"""
        try:
            return asyncio.run(self.download_model(model_name))
        except Exception as e:
            logging.error(f"同期ダウンロードエラー: {e}")
            return False
    
    def get_model_stats(self) -> Dict[str, Any]:
        """モデル統計情報"""
        stats = {
            'total_available': len(self.available_models),
            'downloaded': len(self.list_downloaded_models()),
            'total_size_gb': 0,
            'models': {}
        }
        
        for model_name in self.list_downloaded_models():
            model_info = self.get_model_info(model_name)
            model_path = self.get_model_path(model_name)
            
            if model_info and model_path:
                size_gb = model_path.stat().st_size / (1024**3)
                stats['total_size_gb'] += size_gb
                
                stats['models'][model_name] = {
                    'size_gb': round(size_gb, 2),
                    'integrity_verified': self.verify_model_integrity(model_name),
                    'last_updated': model_info.last_updated
                }
        
        stats['total_size_gb'] = round(stats['total_size_gb'], 2)
        return stats
    
    def cleanup_incomplete_downloads(self) -> int:
        """不完全ダウンロードのクリーンアップ"""
        cleaned_count = 0
        
        for file_path in self.models_dir.glob("*.gguf"):
            if file_path.name.endswith('.tmp') or file_path.name.endswith('.part'):
                try:
                    file_path.unlink()
                    cleaned_count += 1
                    logging.info(f"不完全ダウンロードファイル削除: {file_path.name}")
                except Exception as e:
                    logging.error(f"ファイル削除エラー: {file_path.name} - {e}")
        
        return cleaned_count
    
    def get_recommended_models(self, use_case: str = "general") -> List[str]:
        """推奨モデル取得"""
        recommendations = {
            "general": ["llama3-8b-instruct", "phi3-mini-128k"],
            "code": ["codestral-22b", "llama3-8b-instruct"],
            "high_performance": ["llama3-70b-instruct", "codestral-22b"],
            "lightweight": ["phi3-mini-128k", "mistral-7b-instruct"],
            "statistics": ["llama3-8b-instruct", "phi3-mini-128k"]
        }
        
        return recommendations.get(use_case, ["llama3-8b-instruct"])

# 使用例
def main():
    """メイン関数"""
    manager = GGUFModelManager()
    
    print("=== GGUF Model Manager ===")
    print(f"利用可能モデル数: {len(manager.get_available_models())}")
    print(f"ダウンロード済みモデル数: {len(manager.list_downloaded_models())}")
    
    # 統計分析用推奨モデル
    recommended = manager.get_recommended_models("statistics")
    print(f"統計分析推奨モデル: {recommended}")
    
    # モデル統計
    stats = manager.get_model_stats()
    print(f"総ダウンロードサイズ: {stats['total_size_gb']}GB")
    
    # 整合性検証
    for model_name in manager.list_downloaded_models():
        is_valid = manager.verify_model_integrity(model_name)
        print(f"{model_name}: {'✅' if is_valid else '❌'}")

if __name__ == "__main__":
    main() 