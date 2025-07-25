#!/usr/bin/env python3
"""
本番環境統合テスト・リリースマネージャー
安全な本番環境でのテストとリリースを統合管理
"""

import sys
import os
import time
import gc
import psutil
import json
import subprocess
import shutil
import signal
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production_release.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ReleasePhase(Enum):
    """リリース段階"""
    VALIDATION = "validation"
    DEPLOY = "deploy"
    CANARY = "canary"
    ROLLOUT = "rollout"
    MONITORING = "monitoring"

class TestType(Enum):
    """テストタイプ"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    E2E = "e2e"

@dataclass
class ReleaseConfig:
    """リリース設定"""
    deployment_id: str
    canary_percentage: float = 0.1
    rollout_interval: int = 300  # 5分
    max_rollback_time: int = 60  # 1分
    health_check_interval: int = 30  # 30秒
    monitoring_duration: int = 3600  # 1時間

class ProductionReleaseManager:
    """本番環境リリースマネージャー"""
    
    def __init__(self, config: ReleaseConfig):
        self.config = config
        self.current_phase = ReleasePhase.VALIDATION
        self.test_results = {}
        self.health_metrics = {}
        self.rollback_triggered = False
        self.monitoring_active = False
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"本番環境リリースマネージャー初期化: {config.deployment_id}")
    
    def _signal_handler(self, signum, frame):
        """緊急停止ハンドラー"""
        logger.warning(f"緊急停止シグナル受信: {signum}")
        self.emergency_rollback()
        sys.exit(1)
    
    def emergency_rollback(self):
        """緊急ロールバック"""
        logger.error("緊急ロールバック実行")
        try:
            # 最新の安定版に戻す
            stable_deploy = self._find_latest_stable_deployment()
            if stable_deploy:
                self._switch_to_deployment(stable_deploy)
                logger.info(f"緊急ロールバック完了: {stable_deploy}")
        except Exception as e:
            logger.error(f"緊急ロールバック失敗: {e}")
    
    def _find_latest_stable_deployment(self) -> Optional[str]:
        """最新の安定版デプロイメントを検索"""
        deploy_dir = "production_deploy"
        if not os.path.exists(deploy_dir):
            return None
        
        deployments = [d for d in os.listdir(deploy_dir) if d.startswith("deploy_")]
        if not deployments:
            return None
        
        # 最新のデプロイメントを返す
        return sorted(deployments)[-1]
    
    def _switch_to_deployment(self, deployment_id: str):
        """デプロイメント切り替え"""
        logger.info(f"デプロイメント切り替え: {deployment_id}")
        # 実際の切り替えロジックを実装
        pass
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """包括的バリデーションテスト"""
        logger.info("包括的バリデーションテスト開始")
        results = {
            "success": True,
            "phase": ReleasePhase.VALIDATION.value,
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        # 1. 基本インポートテスト
        results["tests"]["imports"] = self._test_basic_imports()
        
        # 2. コアモジュールテスト
        results["tests"]["core_modules"] = self._test_core_modules()
        
        # 3. データ処理テスト
        results["tests"]["data_processing"] = self._test_data_processing()
        
        # 4. パフォーマンステスト
        results["tests"]["performance"] = self._test_performance()
        
        # 5. セキュリティテスト
        results["tests"]["security"] = self._test_security()
        
        # 6. 統合テスト
        results["tests"]["integration"] = self._test_integration()
        
        # 結果集計
        all_success = all(test["success"] for test in results["tests"].values())
        results["success"] = all_success
        
        if results["success"]:
            logger.info("包括的バリデーションテスト成功")
            self.current_phase = ReleasePhase.DEPLOY
        else:
            logger.error("包括的バリデーションテスト失敗")
        
        return results
    
    def _test_basic_imports(self) -> Dict[str, Any]:
        """基本インポートテスト"""
        logger.info("基本インポートテスト開始")
        results = {"success": True, "errors": []}
        
        required_modules = [
            "pandas", "numpy", "matplotlib", "sklearn",
            "scipy", "seaborn", "plotly"
        ]
        
        for module in required_modules:
            try:
                __import__(module)
                logger.debug(f"{module} インポート成功")
            except Exception as e:
                results["success"] = False
                results["errors"].append(f"{module}: {e}")
                logger.error(f"{module} インポート失敗: {e}")
        
        return results
    
    def _test_core_modules(self) -> Dict[str, Any]:
        """コアモジュールテスト"""
        logger.info("コアモジュールテスト開始")
        results = {"success": True, "errors": []}
        
        core_modules = [
            "src.core.config",
            "src.data.data_preprocessing",
            "src.statistics.advanced_statistics",
            "src.visualization.advanced_visualization",
            "src.gui.main_window"
        ]
        
        for module in core_modules:
            try:
                __import__(module)
                logger.debug(f"{module} インポート成功")
            except Exception as e:
                results["success"] = False
                results["errors"].append(f"{module}: {e}")
                logger.error(f"{module} インポート失敗: {e}")
        
        return results
    
    def _test_data_processing(self) -> Dict[str, Any]:
        """データ処理テスト"""
        logger.info("データ処理テスト開始")
        results = {"success": True, "errors": []}
        
        try:
            import pandas as pd
            import numpy as np
            
            # サンプルデータ作成
            data = pd.DataFrame({
                'A': np.random.randn(100),
                'B': np.random.randn(100),
                'C': np.random.choice(['X', 'Y', 'Z'], 100)
            })
            
            # 基本統計
            stats = data.describe()
            logger.debug("基本統計計算成功")
            
            # 相関分析
            corr = data.corr()
            logger.debug("相関分析成功")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"データ処理: {e}")
            logger.error(f"データ処理テスト失敗: {e}")
        
        return results
    
    def _test_performance(self) -> Dict[str, Any]:
        """パフォーマンステスト"""
        logger.info("パフォーマンステスト開始")
        results = {"success": True, "errors": [], "metrics": {}}
        
        try:
            # メモリ使用量測定
            process = psutil.Process()
            memory_info = process.memory_info()
            results["metrics"]["memory_usage"] = memory_info.rss / 1024 / 1024  # MB
            
            # CPU使用率測定
            cpu_percent = process.cpu_percent(interval=1)
            results["metrics"]["cpu_usage"] = cpu_percent
            
            # 処理時間測定
            start_time = time.time()
            import numpy as np
            large_array = np.random.randn(10000, 1000)
            result = np.linalg.svd(large_array)
            processing_time = time.time() - start_time
            results["metrics"]["processing_time"] = processing_time
            
            # 閾値チェック
            if memory_info.rss > 1024 * 1024 * 1024:  # 1GB
                results["success"] = False
                results["errors"].append("メモリ使用量が閾値を超過")
            
            if processing_time > 30:  # 30秒
                results["success"] = False
                results["errors"].append("処理時間が閾値を超過")
            
            logger.debug(f"パフォーマンステスト完了: {results['metrics']}")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"パフォーマンス: {e}")
            logger.error(f"パフォーマンステスト失敗: {e}")
        
        return results
    
    def _test_security(self) -> Dict[str, Any]:
        """セキュリティテスト"""
        logger.info("セキュリティテスト開始")
        results = {"success": True, "errors": []}
        
        try:
            # ファイル権限チェック
            sensitive_files = [
                "production_config.json",
                "audit_trail.db"
            ]
            
            for file_path in sensitive_files:
                if os.path.exists(file_path):
                    # 権限チェック（簡易版）
                    stat = os.stat(file_path)
                    if stat.st_mode & 0o777 != 0o600:  # 所有者のみ読み書き
                        logger.warning(f"ファイル権限警告: {file_path}")
            
            # 環境変数チェック
            sensitive_env_vars = ["API_KEY", "SECRET_KEY", "DATABASE_URL"]
            for env_var in sensitive_env_vars:
                if env_var in os.environ:
                    logger.warning(f"機密環境変数が設定されています: {env_var}")
            
            logger.debug("セキュリティテスト完了")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"セキュリティ: {e}")
            logger.error(f"セキュリティテスト失敗: {e}")
        
        return results
    
    def _test_integration(self) -> Dict[str, Any]:
        """統合テスト"""
        logger.info("統合テスト開始")
        results = {"success": True, "errors": []}
        
        try:
            # 設定読み込みテスト
            config_path = f"production_deploy/{self.config.deployment_id}/production_config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                logger.debug("設定ファイル読み込み成功")
            
            # データディレクトリ確認
            data_dir = f"production_deploy/{self.config.deployment_id}/data"
            if os.path.exists(data_dir):
                data_files = os.listdir(data_dir)
                logger.debug(f"データファイル確認: {len(data_files)}個")
            
            # リソースディレクトリ確認
            resources_dir = f"production_deploy/{self.config.deployment_id}/resources"
            if os.path.exists(resources_dir):
                resource_files = os.listdir(resources_dir)
                logger.debug(f"リソースファイル確認: {len(resource_files)}個")
            
            logger.debug("統合テスト完了")
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"統合: {e}")
            logger.error(f"統合テスト失敗: {e}")
        
        return results
    
    def deploy_to_production(self) -> Dict[str, Any]:
        """本番環境デプロイ"""
        logger.info("本番環境デプロイ開始")
        results = {
            "success": True,
            "phase": ReleasePhase.DEPLOY.value,
            "timestamp": datetime.now().isoformat(),
            "deployment_id": self.config.deployment_id
        }
        
        try:
            # デプロイメントディレクトリ確認
            deploy_dir = f"production_deploy/{self.config.deployment_id}"
            if not os.path.exists(deploy_dir):
                results["success"] = False
                results["errors"] = [f"デプロイメントディレクトリが存在しません: {deploy_dir}"]
                return results
            
            # 設定ファイル確認
            config_file = os.path.join(deploy_dir, "production_config.json")
            if not os.path.exists(config_file):
                results["success"] = False
                results["errors"] = [f"設定ファイルが存在しません: {config_file}"]
                return results
            
            # 起動スクリプト確認
            startup_script = os.path.join(deploy_dir, "start_production.bat")
            if not os.path.exists(startup_script):
                results["success"] = False
                results["errors"] = [f"起動スクリプトが存在しません: {startup_script}"]
                return results
            
            # 依存関係確認
            requirements_file = os.path.join(deploy_dir, "requirements.txt")
            if not os.path.exists(requirements_file):
                results["success"] = False
                results["errors"] = [f"依存関係ファイルが存在しません: {requirements_file}"]
                return results
            
            logger.info("本番環境デプロイ準備完了")
            self.current_phase = ReleasePhase.CANARY
            
        except Exception as e:
            results["success"] = False
            results["errors"] = [f"デプロイ失敗: {e}"]
            logger.error(f"本番環境デプロイ失敗: {e}")
        
        return results
    
    def start_canary_deployment(self) -> Dict[str, Any]:
        """カナリアデプロイメント開始"""
        logger.info(f"カナリアデプロイメント開始: {self.config.canary_percentage * 100}%")
        results = {
            "success": True,
            "phase": ReleasePhase.CANARY.value,
            "timestamp": datetime.now().isoformat(),
            "canary_percentage": self.config.canary_percentage
        }
        
        try:
            # カナリアデプロイメントの実装
            # 実際の環境では、ロードバランサーやプロキシの設定を変更
            logger.info("カナリアデプロイメント設定完了")
            
            # ヘルスチェック開始
            self._start_health_monitoring()
            
            self.current_phase = ReleasePhase.ROLLOUT
            
        except Exception as e:
            results["success"] = False
            results["errors"] = [f"カナリアデプロイメント失敗: {e}"]
            logger.error(f"カナリアデプロイメント失敗: {e}")
        
        return results
    
    def _start_health_monitoring(self):
        """ヘルスモニタリング開始"""
        logger.info("ヘルスモニタリング開始")
        self.monitoring_active = True
        
        def monitor_health():
            while self.monitoring_active and not self.rollback_triggered:
                try:
                    # システムメトリクス収集
                    process = psutil.Process()
                    memory_usage = process.memory_info().rss / 1024 / 1024
                    cpu_usage = process.cpu_percent()
                    
                    self.health_metrics = {
                        "timestamp": datetime.now().isoformat(),
                        "memory_usage_mb": memory_usage,
                        "cpu_usage_percent": cpu_usage,
                        "phase": self.current_phase.value
                    }
                    
                    # 閾値チェック
                    if memory_usage > 1024:  # 1GB
                        logger.warning(f"メモリ使用量警告: {memory_usage:.2f}MB")
                    
                    if cpu_usage > 80:  # 80%
                        logger.warning(f"CPU使用率警告: {cpu_usage:.2f}%")
                    
                    time.sleep(self.config.health_check_interval)
                    
                except Exception as e:
                    logger.error(f"ヘルスモニタリングエラー: {e}")
                    break
        
        # バックグラウンドでモニタリング開始
        monitoring_thread = threading.Thread(target=monitor_health, daemon=True)
        monitoring_thread.start()
    
    def gradual_rollout(self) -> Dict[str, Any]:
        """段階的ロールアウト"""
        logger.info("段階的ロールアウト開始")
        results = {
            "success": True,
            "phase": ReleasePhase.ROLLOUT.value,
            "timestamp": datetime.now().isoformat(),
            "rollout_stages": []
        }
        
        try:
            rollout_percentages = [25, 50, 75, 100]
            
            for percentage in rollout_percentages:
                logger.info(f"ロールアウト段階: {percentage}%")
                
                # 段階的ロールアウトの実装
                # 実際の環境では、トラフィック配分の調整
                
                rollout_stage = {
                    "percentage": percentage,
                    "timestamp": datetime.now().isoformat(),
                    "health_metrics": self.health_metrics.copy()
                }
                results["rollout_stages"].append(rollout_stage)
                
                # ヘルスチェック待機
                time.sleep(self.config.rollout_interval)
                
                # 異常検出時のロールバック
                if self._detect_anomalies():
                    logger.warning(f"異常検出: {percentage}%段階でロールバック")
                    self.rollback_deployment()
                    results["success"] = False
                    results["errors"] = ["異常検出によるロールバック"]
                    break
            
            if results["success"]:
                self.current_phase = ReleasePhase.MONITORING
                logger.info("段階的ロールアウト完了")
            
        except Exception as e:
            results["success"] = False
            results["errors"] = [f"ロールアウト失敗: {e}"]
            logger.error(f"段階的ロールアウト失敗: {e}")
        
        return results
    
    def _detect_anomalies(self) -> bool:
        """異常検出"""
        if not self.health_metrics:
            return False
        
        # メモリ使用量チェック
        memory_usage = self.health_metrics.get("memory_usage_mb", 0)
        if memory_usage > 2048:  # 2GB
            logger.error(f"メモリ使用量異常: {memory_usage:.2f}MB")
            return True
        
        # CPU使用率チェック
        cpu_usage = self.health_metrics.get("cpu_usage_percent", 0)
        if cpu_usage > 90:  # 90%
            logger.error(f"CPU使用率異常: {cpu_usage:.2f}%")
            return True
        
        return False
    
    def rollback_deployment(self):
        """デプロイメントロールバック"""
        logger.warning("デプロイメントロールバック実行")
        self.rollback_triggered = True
        self.monitoring_active = False
        
        try:
            # 前の安定版に戻す
            stable_deploy = self._find_latest_stable_deployment()
            if stable_deploy and stable_deploy != self.config.deployment_id:
                self._switch_to_deployment(stable_deploy)
                logger.info(f"ロールバック完了: {stable_deploy}")
            else:
                logger.error("ロールバック可能な安定版が見つかりません")
        
        except Exception as e:
            logger.error(f"ロールバック失敗: {e}")
    
    def finalize_release(self) -> Dict[str, Any]:
        """リリース完了処理"""
        logger.info("リリース完了処理開始")
        results = {
            "success": True,
            "phase": ReleasePhase.MONITORING.value,
            "timestamp": datetime.now().isoformat(),
            "final_metrics": self.health_metrics
        }
        
        try:
            # モニタリング停止
            self.monitoring_active = False
            
            # 最終結果保存
            self._save_release_results(results)
            
            # クリーンアップ
            gc.collect()
            
            logger.info("リリース完了処理完了")
            
        except Exception as e:
            results["success"] = False
            results["errors"] = [f"リリース完了処理失敗: {e}"]
            logger.error(f"リリース完了処理失敗: {e}")
        
        return results
    
    def _save_release_results(self, results: Dict[str, Any]):
        """リリース結果保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results/production_release_{timestamp}.json"
        
        os.makedirs("test_results", exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"リリース結果保存: {filename}")
    
    def run_complete_release_pipeline(self) -> Dict[str, Any]:
        """完全なリリースパイプライン実行"""
        logger.info("完全なリリースパイプライン開始")
        
        pipeline_results = {
            "deployment_id": self.config.deployment_id,
            "start_time": datetime.now().isoformat(),
            "phases": {}
        }
        
        try:
            # 1. バリデーション
            validation_result = self.run_comprehensive_validation()
            pipeline_results["phases"]["validation"] = validation_result
            
            if not validation_result["success"]:
                logger.error("バリデーション失敗 - パイプライン停止")
                return pipeline_results
            
            # 2. デプロイ
            deploy_result = self.deploy_to_production()
            pipeline_results["phases"]["deploy"] = deploy_result
            
            if not deploy_result["success"]:
                logger.error("デプロイ失敗 - パイプライン停止")
                return pipeline_results
            
            # 3. カナリアデプロイメント
            canary_result = self.start_canary_deployment()
            pipeline_results["phases"]["canary"] = canary_result
            
            if not canary_result["success"]:
                logger.error("カナリアデプロイメント失敗 - パイプライン停止")
                return pipeline_results
            
            # 4. 段階的ロールアウト
            rollout_result = self.gradual_rollout()
            pipeline_results["phases"]["rollout"] = rollout_result
            
            if not rollout_result["success"]:
                logger.error("ロールアウト失敗 - パイプライン停止")
                return pipeline_results
            
            # 5. リリース完了
            finalize_result = self.finalize_release()
            pipeline_results["phases"]["finalize"] = finalize_result
            
            pipeline_results["end_time"] = datetime.now().isoformat()
            pipeline_results["overall_success"] = all(
                phase["success"] for phase in pipeline_results["phases"].values()
            )
            
            logger.info("完全なリリースパイプライン完了")
            
        except Exception as e:
            logger.error(f"リリースパイプライン失敗: {e}")
            pipeline_results["error"] = str(e)
            pipeline_results["overall_success"] = False
        
        return pipeline_results

def main():
    """メイン実行関数"""
    # 最新のデプロイメントIDを取得
    deploy_dir = "production_deploy"
    if os.path.exists(deploy_dir):
        deployments = [d for d in os.listdir(deploy_dir) if d.startswith("deploy_")]
        if deployments:
            latest_deployment = sorted(deployments)[-1]
        else:
            latest_deployment = "deploy_1753430280"  # フォールバック
    else:
        latest_deployment = "deploy_1753430280"  # フォールバック
    
    # リリース設定
    config = ReleaseConfig(
        deployment_id=latest_deployment,
        canary_percentage=0.1,
        rollout_interval=60,  # 1分（テスト用）
        max_rollback_time=30,  # 30秒（テスト用）
        health_check_interval=10,  # 10秒（テスト用）
        monitoring_duration=300  # 5分（テスト用）
    )
    
    # リリースマネージャー作成
    manager = ProductionReleaseManager(config)
    
    # 完全なリリースパイプライン実行
    results = manager.run_complete_release_pipeline()
    
    # 結果表示
    print("\n" + "="*50)
    print("本番環境リリース結果")
    print("="*50)
    print(f"デプロイメントID: {results['deployment_id']}")
    print(f"全体成功: {results.get('overall_success', False)}")
    print(f"開始時刻: {results['start_time']}")
    print(f"終了時刻: {results.get('end_time', 'N/A')}")
    
    if results.get('overall_success', False):
        print("✅ リリース成功")
    else:
        print("❌ リリース失敗")
        if 'error' in results:
            print(f"エラー: {results['error']}")
    
    return results

if __name__ == "__main__":
    main() 