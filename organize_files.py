#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ファイル整理スクリプト
リポジトリのファイルを新しいディレクトリ構造に移動します
"""
import os
import shutil
import logging
from pathlib import Path

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """新しいディレクトリ構造を作成"""
    directories = [
        "src/core",
        "src/gui", 
        "src/statistics",
        "src/ai",
        "src/data",
        "src/visualization",
        "src/performance",
        "src/security",
        "src/distribution",
        "src/tests",
        "src/runners",
        "documentation/implementation_logs",
        "resources"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"ディレクトリ作成: {directory}")

def move_files():
    """ファイルを適切なディレクトリに移動"""
    file_mappings = {
        # コアモジュール
        "main.py": "src/core/",
        "config.py": "src/core/",
        "professional_utils.py": "src/core/",
        
        # GUIモジュール
        "HAD_Statistics_GUI.py": "src/gui/",
        "kiro_integrated_gui.py": "src/gui/",
        "professional_statistics_gui.py": "src/gui/",
        "unified_ai_landing_gui.py": "src/gui/",
        
        # 統計解析モジュール
        "advanced_statistics.py": "src/statistics/",
        "assumption_validator.py": "src/statistics/",
        "bayesian_analysis.py": "src/statistics/",
        "statistical_method_advisor.py": "src/statistics/",
        "statistical_power_analysis.py": "src/statistics/",
        "survival_analysis.py": "src/statistics/",
        
        # AI統合モジュール
        "ai_integration.py": "src/ai/",
        "contextual_retriever.py": "src/ai/",
        "demo_statistical_gguf.py": "src/ai/",
        "gguf_model_selector.py": "src/ai/",
        "local_llm_statistical_assistant.py": "src/ai/",
        
        # データ処理モジュール
        "data_preprocessing.py": "src/data/",
        "sample_data.py": "src/data/",
        
        # 可視化モジュール
        "advanced_visualization.py": "src/visualization/",
        "professional_reports.py": "src/visualization/",
        "web_dashboard.py": "src/visualization/",
        
        # パフォーマンス最適化
        "parallel_optimization.py": "src/performance/",
        
        # セキュリティモジュール
        "audit_compliance_system.py": "src/security/",
        "booth_protection.py": "src/security/",
        "trial_license_system.py": "src/security/",
        
        # 配布関連モジュール
        "booth_build_system.py": "src/distribution/",
        "booth_deployment_automation.py": "src/distribution/",
        "booth_license_generator.py": "src/distribution/",
        "booth_sales_manager.py": "src/distribution/",
        "build_exe_auto.py": "src/distribution/",
        "exe_builder_system.py": "src/distribution/",
        "generate_booth_content.py": "src/distribution/",
        
        # 実行スクリプト
        "context_manager_demo.py": "src/runners/",
        "interactive_analysis_app.py": "src/runners/",
        "launch_booth_gui.py": "src/runners/",
        "run_kiro_gui.py": "src/runners/",
        "run_professional_gui.py": "src/runners/",
        "run_unified_ai_landing.py": "src/runners/",
        "run_web_dashboard.py": "src/runners/",
        "statistical_advisor_demo.py": "src/runners/"
    }
    
    for source_file, dest_dir in file_mappings.items():
        if os.path.exists(source_file):
            dest_path = os.path.join(dest_dir, source_file)
            try:
                shutil.move(source_file, dest_path)
                logger.info(f"移動完了: {source_file} -> {dest_path}")
            except Exception as e:
                logger.error(f"移動エラー: {source_file} -> {dest_path}: {e}")
        else:
            logger.warning(f"ファイルが見つかりません: {source_file}")

def copy_resources():
    """リソースファイルをコピー"""
    resource_files = [
        "edition_config_gpu_accelerated.json",
        "edition_config_lite.json", 
        "edition_config_professional.json",
        "edition_config_standard.json",
        "gguf_model_config.json",
        "env_template.txt",
        "requirements.txt",
        "requirements_booth.txt",
        "requirements_exe.txt",
        "run_gemini_test.bat",
        "run_gemini_test.sh"
    ]
    
    for file in resource_files:
        if os.path.exists(file):
            dest_path = os.path.join("resources", file)
            try:
                shutil.copy2(file, dest_path)
                logger.info(f"コピー完了: {file} -> {dest_path}")
            except Exception as e:
                logger.error(f"コピーエラー: {file} -> {dest_path}: {e}")
        else:
            logger.warning(f"リソースファイルが見つかりません: {file}")

def create_init_files():
    """__init__.pyファイルを作成"""
    init_dirs = [
        "src",
        "src/core",
        "src/gui",
        "src/statistics", 
        "src/ai",
        "src/data",
        "src/visualization",
        "src/performance",
        "src/security",
        "src/distribution",
        "src/tests",
        "src/runners"
    ]
    
    for directory in init_dirs:
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_file):
            package_name = os.path.basename(directory)
            content = f'"""{package_name} package"""\n\n'
            with open(init_file, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"__init__.py作成: {init_file}")

def main():
    """メイン処理"""
    logger.info("🔄 ファイル整理を開始します...")
    
    try:
        create_directories()
        move_files()
        copy_resources()
        create_init_files()
        
        logger.info("✅ ファイル整理が完了しました")
        
    except Exception as e:
        logger.error(f"❌ ファイル整理中にエラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    main()
