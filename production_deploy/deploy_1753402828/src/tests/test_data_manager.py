#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Data Manager
テストデータ管理システム

Author: Ryo Minegishi
Email: r.minegishi1987@gmail.com
License: MIT
"""

import json
import os
import sys
import shutil
import hashlib
import zipfile
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import sqlite3
import pickle
import yaml
from abc import ABC, abstractmethod
import threading
import time
import random
import string

@dataclass
class TestDataSet:
    """テストデータセット"""
    name: str
    description: str
    data_type: str  # csv, json, sqlite, pickle, yaml
    file_path: str
    size_bytes: int
    created_at: datetime
    updated_at: datetime
    version: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""

@dataclass
class DataGenerationConfig:
    """データ生成設定"""
    data_type: str
    size: int
    columns: List[str]
    data_types: Dict[str, str]  # column_name -> data_type
    constraints: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None

class DataGenerator:
    """テストデータ生成器"""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed or int(time.time())
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.logger = logging.getLogger(__name__)
    
    def generate_numeric_data(self, size: int, data_type: str = "float", **kwargs) -> List[Union[int, float]]:
        """数値データを生成"""
        if data_type == "int":
            min_val = kwargs.get("min_val", 0)
            max_val = kwargs.get("max_val", 1000)
            return np.random.randint(min_val, max_val, size).tolist()
        elif data_type == "float":
            min_val = kwargs.get("min_val", 0.0)
            max_val = kwargs.get("max_val", 1000.0)
            return np.random.uniform(min_val, max_val, size).tolist()
        elif data_type == "normal":
            mean = kwargs.get("mean", 0.0)
            std = kwargs.get("std", 1.0)
            return np.random.normal(mean, std, size).tolist()
        else:
            raise ValueError(f"Unsupported numeric data type: {data_type}")
    
    def generate_categorical_data(self, size: int, categories: List[str], **kwargs) -> List[str]:
        """カテゴリカルデータを生成"""
        weights = kwargs.get("weights", None)
        if weights and len(weights) == len(categories):
            return np.random.choice(categories, size, p=weights).tolist()
        else:
            return np.random.choice(categories, size).tolist()
    
    def generate_text_data(self, size: int, min_length: int = 5, max_length: int = 20, **kwargs) -> List[str]:
        """テキストデータを生成"""
        texts = []
        for _ in range(size):
            length = random.randint(min_length, max_length)
            text = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
            texts.append(text)
        return texts
    
    def generate_datetime_data(self, size: int, start_date: str = "2020-01-01", end_date: str = "2024-12-31", **kwargs) -> List[str]:
        """日時データを生成"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        dates = []
        for _ in range(size):
            random_date = start + timedelta(days=random.randint(0, (end - start).days))
            dates.append(random_date.strftime("%Y-%m-%d"))
        return dates
    
    def generate_boolean_data(self, size: int, true_probability: float = 0.5) -> List[bool]:
        """ブールデータを生成"""
        return np.random.choice([True, False], size, p=[true_probability, 1-true_probability]).tolist()
    
    def generate_dataframe(self, config: DataGenerationConfig) -> pd.DataFrame:
        """設定に基づいてDataFrameを生成"""
        data = {}
        
        for column in config.columns:
            data_type = config.data_types.get(column, "float")
            
            if data_type in ["int", "float", "normal"]:
                data[column] = self.generate_numeric_data(config.size, data_type, **config.constraints.get(column, {}))
            elif data_type == "categorical":
                categories = config.constraints.get(column, {}).get("categories", ["A", "B", "C"])
                data[column] = self.generate_categorical_data(config.size, categories, **config.constraints.get(column, {}))
            elif data_type == "text":
                data[column] = self.generate_text_data(config.size, **config.constraints.get(column, {}))
            elif data_type == "datetime":
                data[column] = self.generate_datetime_data(config.size, **config.constraints.get(column, {}))
            elif data_type == "boolean":
                data[column] = self.generate_boolean_data(config.size, **config.constraints.get(column, {}))
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
        
        return pd.DataFrame(data)

class DataStorage:
    """データストレージ管理"""
    
    def __init__(self, base_dir: str = "test_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # データベース初期化
        self.db_path = self.base_dir / "test_data.db"
        self._init_database()
    
    def _init_database(self):
        """データベースを初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                data_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                size_bytes INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                version TEXT NOT NULL,
                tags TEXT,
                metadata TEXT,
                checksum TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_dataset(self, dataset: TestDataSet) -> bool:
        """データセットを保存"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO test_datasets 
                (name, description, data_type, file_path, size_bytes, created_at, updated_at, version, tags, metadata, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                dataset.name,
                dataset.description,
                dataset.data_type,
                dataset.file_path,
                dataset.size_bytes,
                dataset.created_at.isoformat(),
                dataset.updated_at.isoformat(),
                dataset.version,
                json.dumps(dataset.tags),
                json.dumps(dataset.metadata),
                dataset.checksum
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ データセット保存完了: {dataset.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ データセット保存エラー: {e}")
            return False
    
    def load_dataset(self, name: str) -> Optional[TestDataSet]:
        """データセットを読み込み"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM test_datasets WHERE name = ?', (name,))
            row = cursor.fetchone()
            
            conn.close()
            
            if row:
                return TestDataSet(
                    name=row[1],
                    description=row[2],
                    data_type=row[3],
                    file_path=row[4],
                    size_bytes=row[5],
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                    version=row[8],
                    tags=json.loads(row[9]) if row[9] else [],
                    metadata=json.loads(row[10]) if row[10] else {},
                    checksum=row[11]
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ データセット読み込みエラー: {e}")
            return None
    
    def list_datasets(self, tags: Optional[List[str]] = None) -> List[TestDataSet]:
        """データセット一覧を取得"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if tags:
                # タグでフィルタリング
                placeholders = ','.join(['?' for _ in tags])
                cursor.execute(f'''
                    SELECT * FROM test_datasets 
                    WHERE tags LIKE '%{placeholders}%'
                ''', tags)
            else:
                cursor.execute('SELECT * FROM test_datasets')
            
            rows = cursor.fetchall()
            conn.close()
            
            datasets = []
            for row in rows:
                dataset = TestDataSet(
                    name=row[1],
                    description=row[2],
                    data_type=row[3],
                    file_path=row[4],
                    size_bytes=row[5],
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                    version=row[8],
                    tags=json.loads(row[9]) if row[9] else [],
                    metadata=json.loads(row[10]) if row[10] else {},
                    checksum=row[11]
                )
                datasets.append(dataset)
            
            return datasets
            
        except Exception as e:
            self.logger.error(f"❌ データセット一覧取得エラー: {e}")
            return []
    
    def delete_dataset(self, name: str) -> bool:
        """データセットを削除"""
        try:
            # データセット情報を取得
            dataset = self.load_dataset(name)
            if not dataset:
                return False
            
            # ファイルを削除
            file_path = Path(dataset.file_path)
            if file_path.exists():
                file_path.unlink()
            
            # データベースから削除
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM test_datasets WHERE name = ?', (name,))
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ データセット削除完了: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ データセット削除エラー: {e}")
            return False

class DataSerializer:
    """データシリアライザー"""
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, file_path: str, data_type: str) -> bool:
        """DataFrameをファイルに保存"""
        try:
            if data_type == "csv":
                df.to_csv(file_path, index=False)
            elif data_type == "json":
                df.to_json(file_path, orient="records", indent=2)
            elif data_type == "pickle":
                df.to_pickle(file_path)
            elif data_type == "parquet":
                df.to_parquet(file_path, index=False)
            elif data_type == "excel":
                df.to_excel(file_path, index=False)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            return True
            
        except Exception as e:
            logging.error(f"データ保存エラー: {e}")
            return False
    
    @staticmethod
    def load_dataframe(file_path: str, data_type: str) -> Optional[pd.DataFrame]:
        """ファイルからDataFrameを読み込み"""
        try:
            if data_type == "csv":
                return pd.read_csv(file_path)
            elif data_type == "json":
                return pd.read_json(file_path)
            elif data_type == "pickle":
                return pd.read_pickle(file_path)
            elif data_type == "parquet":
                return pd.read_parquet(file_path)
            elif data_type == "excel":
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
        except Exception as e:
            logging.error(f"データ読み込みエラー: {e}")
            return None

class TestDataManager:
    """テストデータ管理システム"""
    
    def __init__(self, base_dir: str = "test_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # コンポーネント初期化
        self.generator = DataGenerator()
        self.storage = DataStorage(base_dir)
        self.serializer = DataSerializer()
        
        # ロック（スレッドセーフ）
        self.lock = threading.Lock()
    
    def generate_test_data(self, config: DataGenerationConfig, name: str, description: str = "", tags: List[str] = None) -> Optional[TestDataSet]:
        """テストデータを生成"""
        try:
            with self.lock:
                # DataFrameを生成
                df = self.generator.generate_dataframe(config)
                
                # ファイルパスを決定
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp}.{config.data_type}"
                file_path = self.base_dir / filename
                
                # データを保存
                if not self.serializer.save_dataframe(df, str(file_path), config.data_type):
                    return None
                
                # チェックサムを計算
                checksum = self._calculate_checksum(file_path)
                
                # データセット情報を作成
                dataset = TestDataSet(
                    name=name,
                    description=description,
                    data_type=config.data_type,
                    file_path=str(file_path),
                    size_bytes=file_path.stat().st_size,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    version="1.0.0",
                    tags=tags or [],
                    metadata={
                        "size": config.size,
                        "columns": config.columns,
                        "data_types": config.data_types,
                        "constraints": config.constraints,
                        "seed": config.seed
                    },
                    checksum=checksum
                )
                
                # データベースに保存
                if self.storage.save_dataset(dataset):
                    self.logger.info(f"✅ テストデータ生成完了: {name}")
                    return dataset
                else:
                    # 失敗した場合はファイルを削除
                    file_path.unlink()
                    return None
                
        except Exception as e:
            self.logger.error(f"❌ テストデータ生成エラー: {e}")
            return None
    
    def load_test_data(self, name: str) -> Optional[pd.DataFrame]:
        """テストデータを読み込み"""
        try:
            dataset = self.storage.load_dataset(name)
            if not dataset:
                return None
            
            # チェックサムを検証
            current_checksum = self._calculate_checksum(Path(dataset.file_path))
            if current_checksum != dataset.checksum:
                self.logger.warning(f"チェックサム不一致: {name}")
            
            return self.serializer.load_dataframe(dataset.file_path, dataset.data_type)
            
        except Exception as e:
            self.logger.error(f"❌ テストデータ読み込みエラー: {e}")
            return None
    
    def list_test_data(self, tags: Optional[List[str]] = None) -> List[TestDataSet]:
        """テストデータ一覧を取得"""
        return self.storage.list_datasets(tags)
    
    def delete_test_data(self, name: str) -> bool:
        """テストデータを削除"""
        return self.storage.delete_dataset(name)
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """古いデータをクリーンアップ"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            datasets = self.storage.list_datasets()
            
            deleted_count = 0
            for dataset in datasets:
                if dataset.created_at < cutoff_date:
                    if self.storage.delete_dataset(dataset.name):
                        deleted_count += 1
            
            self.logger.info(f"✅ {deleted_count}件の古いデータを削除しました")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"❌ データクリーンアップエラー: {e}")
            return 0
    
    def export_data(self, name: str, export_path: str, export_format: str = "zip") -> bool:
        """データをエクスポート"""
        try:
            dataset = self.storage.load_dataset(name)
            if not dataset:
                return False
            
            if export_format == "zip":
                with zipfile.ZipFile(export_path, 'w') as zipf:
                    zipf.write(dataset.file_path, os.path.basename(dataset.file_path))
                    
                    # メタデータも含める
                    metadata = {
                        "dataset": dataset.__dict__,
                        "export_date": datetime.now().isoformat()
                    }
                    zipf.writestr("metadata.json", json.dumps(metadata, indent=2, default=str))
            
            self.logger.info(f"✅ データエクスポート完了: {name} -> {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ データエクスポートエラー: {e}")
            return False
    
    def import_data(self, import_path: str, name: str = None) -> Optional[TestDataSet]:
        """データをインポート"""
        try:
            if import_path.endswith('.zip'):
                with zipfile.ZipFile(import_path, 'r') as zipf:
                    # メタデータを読み込み
                    metadata_content = zipf.read("metadata.json")
                    metadata = json.loads(metadata_content)
                    
                    # データファイルを抽出
                    data_files = [f for f in zipf.namelist() if not f.endswith('.json')]
                    if not data_files:
                        return None
                    
                    data_file = data_files[0]
                    temp_dir = tempfile.mkdtemp()
                    zipf.extract(data_file, temp_dir)
                    
                    # データセット情報を復元
                    dataset_dict = metadata["dataset"]
                    dataset_dict["created_at"] = datetime.fromisoformat(dataset_dict["created_at"])
                    dataset_dict["updated_at"] = datetime.fromisoformat(dataset_dict["updated_at"])
                    
                    dataset = TestDataSet(**dataset_dict)
                    
                    # 新しいファイルパスを設定
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_filename = f"{dataset.name}_{timestamp}.{dataset.data_type}"
                    new_file_path = self.base_dir / new_filename
                    
                    # ファイルを移動
                    shutil.move(os.path.join(temp_dir, data_file), new_file_path)
                    shutil.rmtree(temp_dir)
                    
                    # データセット情報を更新
                    dataset.file_path = str(new_file_path)
                    dataset.size_bytes = new_file_path.stat().st_size
                    dataset.updated_at = datetime.now()
                    dataset.checksum = self._calculate_checksum(new_file_path)
                    
                    # データベースに保存
                    if self.storage.save_dataset(dataset):
                        self.logger.info(f"✅ データインポート完了: {dataset.name}")
                        return dataset
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ データインポートエラー: {e}")
            return None
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """ファイルのチェックサムを計算"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

class TestDataFactory:
    """テストデータファクトリー"""
    
    @staticmethod
    def create_sample_data_config() -> DataGenerationConfig:
        """サンプルデータ設定を作成"""
        return DataGenerationConfig(
            data_type="csv",
            size=1000,
            columns=["id", "name", "age", "salary", "department", "hire_date", "is_active"],
            data_types={
                "id": "int",
                "name": "text",
                "age": "int",
                "salary": "float",
                "department": "categorical",
                "hire_date": "datetime",
                "is_active": "boolean"
            },
            constraints={
                "id": {"min_val": 1, "max_val": 10000},
                "age": {"min_val": 18, "max_val": 65},
                "salary": {"min_val": 30000, "max_val": 150000},
                "department": {"categories": ["Engineering", "Sales", "Marketing", "HR", "Finance"]},
                "name": {"min_length": 5, "max_length": 15}
            },
            seed=42
        )
    
    @staticmethod
    def create_performance_test_config() -> DataGenerationConfig:
        """パフォーマンステスト用データ設定を作成"""
        return DataGenerationConfig(
            data_type="csv",
            size=100000,
            columns=["id", "value", "category", "timestamp"],
            data_types={
                "id": "int",
                "value": "normal",
                "category": "categorical",
                "timestamp": "datetime"
            },
            constraints={
                "id": {"min_val": 1, "max_val": 1000000},
                "value": {"mean": 100, "std": 20},
                "category": {"categories": ["A", "B", "C", "D", "E"]},
                "timestamp": {"start_date": "2020-01-01", "end_date": "2024-12-31"}
            },
            seed=123
        )

def main():
    """メイン実行関数"""
    print("🚀 テストデータ管理システム起動")
    
    # テストデータマネージャー初期化
    manager = TestDataManager()
    
    # サンプルデータを生成
    print("📊 サンプルデータを生成中...")
    sample_config = TestDataFactory.create_sample_data_config()
    sample_dataset = manager.generate_test_data(
        sample_config,
        name="sample_employee_data",
        description="従業員データのサンプル",
        tags=["sample", "employee", "hr"]
    )
    
    if sample_dataset:
        print(f"✅ サンプルデータ生成完了: {sample_dataset.name}")
        
        # データを読み込み
        df = manager.load_test_data(sample_dataset.name)
        if df is not None:
            print(f"📋 データ形状: {df.shape}")
            print(f"📋 カラム: {list(df.columns)}")
            print(f"📋 最初の5行:")
            print(df.head())
    
    # パフォーマンステストデータを生成
    print("\n⚡ パフォーマンステストデータを生成中...")
    perf_config = TestDataFactory.create_performance_test_config()
    perf_dataset = manager.generate_test_data(
        perf_config,
        name="performance_test_data",
        description="パフォーマンステスト用大規模データ",
        tags=["performance", "large", "test"]
    )
    
    if perf_dataset:
        print(f"✅ パフォーマンステストデータ生成完了: {perf_dataset.name}")
    
    # データ一覧を表示
    print("\n📋 生成されたデータセット一覧:")
    datasets = manager.list_test_data()
    for dataset in datasets:
        print(f"  - {dataset.name}: {dataset.description} ({dataset.data_type}, {dataset.size_bytes} bytes)")
    
    # クリーンアップ（古いデータを削除）
    print("\n🧹 古いデータをクリーンアップ中...")
    deleted_count = manager.cleanup_old_data(days=1)  # 1日以上古いデータを削除
    print(f"✅ {deleted_count}件のデータを削除しました")

if __name__ == "__main__":
    main() 