#!/usr/bin/env python3
"""
本番環境動作確認テスト
実際の運用環境での包括的なテスト
"""

import sys
import os
import time
import gc
import psutil
import json
import subprocess
from datetime import datetime
from typing import Dict, Any, List

def test_production_startup() -> Dict[str, Any]:
    """本番環境起動テスト"""
    print("本番環境起動テスト開始")
    results = {"success": True, "errors": []}
    
    try:
        # 本番環境ディレクトリの存在確認
        production_dir = "production_deploy/deploy_1753430280"
        if not os.path.exists(production_dir):
            results["success"] = False
            results["errors"].append(f"本番環境ディレクトリが存在しません: {production_dir}")
            print(f"本番環境ディレクトリが存在しません: {production_dir}")
            return results
        
        # 本番設定ファイルの確認
        config_file = os.path.join(production_dir, "production_config.json")
        if not os.path.exists(config_file):
            results["success"] = False
            results["errors"].append(f"本番設定ファイルが存在しません: {config_file}")
            print(f"本番設定ファイルが存在しません: {config_file}")
        else:
            print("本番設定ファイル確認成功")
        
        # 起動スクリプトの確認
        startup_script = os.path.join(production_dir, "start_production.bat")
        if not os.path.exists(startup_script):
            results["success"] = False
            results["errors"].append(f"起動スクリプトが存在しません: {startup_script}")
            print(f"起動スクリプトが存在しません: {startup_script}")
        else:
            print("起動スクリプト確認成功")
        
        # requirements.txtの確認
        requirements_file = os.path.join(production_dir, "requirements.txt")
        if not os.path.exists(requirements_file):
            results["success"] = False
            results["errors"].append(f"依存関係ファイルが存在しません: {requirements_file}")
            print(f"依存関係ファイルが存在しません: {requirements_file}")
        else:
            print("依存関係ファイル確認成功")
        
        if results["success"]:
            print("本番環境起動テスト成功")
        
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"ProductionStartup: {e}")
        print(f"本番環境起動テスト失敗: {e}")
    
    return results

def test_production_data_access() -> Dict[str, Any]:
    """本番環境データアクセステスト"""
    print("本番環境データアクセステスト開始")
    results = {"success": True, "errors": []}
    
    try:
        # 本番環境のデータディレクトリ確認
        data_dir = "production_deploy/deploy_1753430280/data"
        if not os.path.exists(data_dir):
            results["success"] = False
            results["errors"].append(f"データディレクトリが存在しません: {data_dir}")
            print(f"データディレクトリが存在しません: {data_dir}")
            return results
        
        # データファイルの確認
        data_files = [
            "basic_statistics_sample.csv",
            "machine_learning_sample.csv",
            "medical_survival_sample.csv",
            "regression_sample.csv",
            "timeseries_sample.csv"
        ]
        
        for data_file in data_files:
            file_path = os.path.join(data_dir, data_file)
            if not os.path.exists(file_path):
                results["success"] = False
                results["errors"].append(f"データファイルが存在しません: {file_path}")
                print(f"データファイルが存在しません: {file_path}")
            else:
                # ファイルサイズの確認
                file_size = os.path.getsize(file_path)
                print(f"データファイル確認成功: {data_file} ({file_size} bytes)")
        
        if results["success"]:
            print("本番環境データアクセステスト成功")
        
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"ProductionDataAccess: {e}")
        print(f"本番環境データアクセステスト失敗: {e}")
    
    return results

def test_production_module_imports() -> Dict[str, Any]:
    """本番環境モジュールインポートテスト"""
    print("本番環境モジュールインポートテスト開始")
    results = {"success": True, "errors": []}
    
    try:
        # 本番環境のsrcディレクトリ確認
        src_dir = "production_deploy/deploy_1753430280/src"
        if not os.path.exists(src_dir):
            results["success"] = False
            results["errors"].append(f"ソースディレクトリが存在しません: {src_dir}")
            print(f"ソースディレクトリが存在しません: {src_dir}")
            return results
        
        # 主要モジュールのインポートテスト
        modules_to_test = [
            "src.core.config",
            "src.data.data_preprocessing",
            "src.statistics.advanced_statistics",
            "src.gui.professional_statistics_gui",
            "src.ai.ai_integration"
        ]
        
        for module_name in modules_to_test:
            try:
                # 本番環境のパスを追加
                sys.path.insert(0, "production_deploy/deploy_1753430280")
                __import__(module_name)
                print(f"モジュールインポート成功: {module_name}")
            except Exception as e:
                results["success"] = False
                results["errors"].append(f"{module_name}: {e}")
                print(f"モジュールインポート失敗: {module_name} - {e}")
        
        if results["success"]:
            print("本番環境モジュールインポートテスト成功")
        
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"ProductionModuleImports: {e}")
        print(f"本番環境モジュールインポートテスト失敗: {e}")
    
    return results

def test_production_performance() -> Dict[str, Any]:
    """本番環境パフォーマンステスト"""
    print("本番環境パフォーマンステスト開始")
    results = {"success": True, "errors": []}
    
    try:
        # システムリソース監視
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        print("システムリソース監視:")
        print(f"   - メモリ使用量: {memory.used / (1024**3):.2f}GB / {memory.total / (1024**3):.2f}GB")
        print(f"   - CPU使用率: {cpu_percent:.1f}%")
        print(f"   - ディスク使用率: {disk.percent:.1f}%")
        
        # パフォーマンス基準の確認
        if memory.percent > 90:
            results["warnings"] = [f"メモリ使用率が高い: {memory.percent:.1f}%"]
            print(f"メモリ使用率警告: {memory.percent:.1f}% > 90%")
        
        if cpu_percent > 80:
            results["warnings"] = results.get("warnings", []) + [f"CPU使用率が高い: {cpu_percent:.1f}%"]
            print(f"CPU使用率警告: {cpu_percent:.1f}% > 80%")
        
        if disk.percent > 90:
            results["warnings"] = results.get("warnings", []) + [f"ディスク使用率が高い: {disk.percent:.1f}%"]
            print(f"ディスク使用率警告: {disk.percent:.1f}% > 90%")
        
        print("本番環境パフォーマンステスト成功")
        
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"ProductionPerformance: {e}")
        print(f"本番環境パフォーマンステスト失敗: {e}")
    
    return results

def test_production_security() -> Dict[str, Any]:
    """本番環境セキュリティテスト"""
    print("本番環境セキュリティテスト開始")
    results = {"success": True, "errors": []}
    
    try:
        # 本番環境ディレクトリの権限確認
        production_dir = "production_deploy/deploy_1753430280"
        
        # 重要なファイルの存在確認
        critical_files = [
            "production_config.json",
            "requirements.txt",
            "src/security/audit_compliance_system.py",
            "src/security/booth_protection.py"
        ]
        
        for file_name in critical_files:
            file_path = os.path.join(production_dir, file_name)
            if not os.path.exists(file_path):
                results["success"] = False
                results["errors"].append(f"セキュリティファイルが存在しません: {file_path}")
                print(f"セキュリティファイルが存在しません: {file_path}")
            else:
                print(f"セキュリティファイル確認成功: {file_name}")
        
        # 設定ファイルの内容確認
        config_file = os.path.join(production_dir, "production_config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # セキュリティ設定の確認
                if "security" in config_data:
                    print("セキュリティ設定確認成功")
                else:
                    results["warnings"] = results.get("warnings", []) + ["セキュリティ設定が不足しています"]
                    print("セキュリティ設定が不足しています")
            except Exception as e:
                results["errors"].append(f"設定ファイル読み込みエラー: {e}")
                print(f"設定ファイル読み込みエラー: {e}")
        
        if results["success"]:
            print("本番環境セキュリティテスト成功")
        
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"ProductionSecurity: {e}")
        print(f"本番環境セキュリティテスト失敗: {e}")
    
    return results

def test_production_backup_integrity() -> Dict[str, Any]:
    """本番環境バックアップ整合性テスト"""
    print("本番環境バックアップ整合性テスト開始")
    results = {"success": True, "errors": []}
    
    try:
        # バックアップディレクトリの確認
        backup_dir = "production_backups/deploy_1753430280"
        if not os.path.exists(backup_dir):
            results["success"] = False
            results["errors"].append(f"バックアップディレクトリが存在しません: {backup_dir}")
            print(f"バックアップディレクトリが存在しません: {backup_dir}")
            return results
        
        # バックアップファイルの確認
        backup_files = [
            "IMPLEMENTATION_LOG.md",
            "requirements.txt",
            "src/",
            "data/",
            "resources/"
        ]
        
        for backup_item in backup_files:
            backup_path = os.path.join(backup_dir, backup_item)
            if not os.path.exists(backup_path):
                results["success"] = False
                results["errors"].append(f"バックアップアイテムが存在しません: {backup_path}")
                print(f"バックアップアイテムが存在しません: {backup_path}")
            else:
                if os.path.isdir(backup_path):
                    file_count = len([f for f in os.listdir(backup_path) if os.path.isfile(os.path.join(backup_path, f))])
                    print(f"バックアップディレクトリ確認成功: {backup_item} ({file_count} files)")
                else:
                    file_size = os.path.getsize(backup_path)
                    print(f"バックアップファイル確認成功: {backup_item} ({file_size} bytes)")
        
        if results["success"]:
            print("本番環境バックアップ整合性テスト成功")
        
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"ProductionBackupIntegrity: {e}")
        print(f"本番環境バックアップ整合性テスト失敗: {e}")
    
    return results

def run_production_validation_test() -> Dict[str, Any]:
    """本番環境動作確認テスト実行"""
    print("本番環境動作確認テスト開始")
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    start_time = time.time()
    test_results = []
    
    # テスト実行
    tests = [
        ("本番環境起動", test_production_startup),
        ("本番環境データアクセス", test_production_data_access),
        ("本番環境モジュールインポート", test_production_module_imports),
        ("本番環境パフォーマンス", test_production_performance),
        ("本番環境セキュリティ", test_production_security),
        ("本番環境バックアップ整合性", test_production_backup_integrity)
    ]
    
    successful_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"{test_name}テスト実行中...")
        
        # メモリクリーンアップ
        gc.collect()
        
        result = test_func()
        test_results.append({
            "name": test_name,
            "result": result
        })
        
        if result["success"]:
            successful_tests += 1
            print(f"{test_name}テスト成功")
        else:
            print(f"{test_name}テスト失敗")
            for error in result.get("errors", []):
                print(f"   - エラー: {error}")
        
        # 警告の表示
        for warning in result.get("warnings", []):
            print(f"   - 警告: {warning}")
    
    execution_time = time.time() - start_time
    success_rate = (successful_tests / total_tests) * 100
    
    # 結果表示
    print("\n" + "="*60)
    print("本番環境動作確認テスト結果")
    print("="*60)
    print(f"成功テスト数: {successful_tests}/{total_tests}")
    print(f"成功率: {success_rate:.1f}%")
    print(f"実行時間: {execution_time:.1f}秒")
    
    overall_success = success_rate >= 80.0
    
    if overall_success:
        print("本番環境動作確認テスト成功！運用環境準備完了")
    else:
        print("本番環境動作確認テスト失敗。追加の修正が必要")
    
    # 結果をJSONファイルに保存
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "overall_success": overall_success,
        "success_rate": success_rate,
        "successful_tests": successful_tests,
        "total_tests": total_tests,
        "execution_time": execution_time,
        "test_results": test_results
    }
    
    with open("production_validation_test_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n結果保存: production_validation_test_results.json")
    
    return final_results

if __name__ == "__main__":
    results = run_production_validation_test()
    
    if results["overall_success"]:
        print("次のステップ:")
        print("1. 実際のユーザーでのテスト開始")
        print("2. パフォーマンス監視の継続")
        print("3. セキュリティ監査の実施")
        sys.exit(0)
    else:
        print("テスト失敗。修正が必要です。")
        sys.exit(1)
