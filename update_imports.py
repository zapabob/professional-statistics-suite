#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
インポートパス修正スクリプト
ファイル移動後のインポートパスを更新します
"""

import os
import re
import glob
from pathlib import Path

# モジュールのマッピング（元のモジュール名 -> 新しいインポートパス）
module_mapping = {
    # コアモジュール
    "main": "src.core.main",
    "config": "src.core.config",
    "professional_utils": "src.core.professional_utils",
    
    # GUIモジュール
    "HAD_Statistics_GUI": "src.gui.HAD_Statistics_GUI",
    "professional_statistics_gui": "src.gui.professional_statistics_gui",
    "unified_ai_landing_gui": "src.gui.unified_ai_landing_gui",
    "kiro_integrated_gui": "src.gui.kiro_integrated_gui",
    "booth_license_gui": "src.gui.booth_license_gui",
    "booth_license_simple_gui": "src.gui.booth_license_simple_gui",
    
    # 統計モジュール
    "advanced_statistics": "src.statistics.advanced_statistics",
    "bayesian_analysis": "src.statistics.bayesian_analysis",
    "survival_analysis": "src.statistics.survival_analysis",
    "statistical_power_analysis": "src.statistics.statistical_power_analysis",
    "statistical_method_advisor": "src.statistics.statistical_method_advisor",
    "assumption_validator": "src.statistics.assumption_validator",
    
    # AIモジュール
    "ai_integration": "src.ai.ai_integration",
    "contextual_retriever": "src.ai.contextual_retriever",
    "gguf_model_selector": "src.ai.gguf_model_selector",
    "local_llm_statistical_assistant": "src.ai.local_llm_statistical_assistant",
    "demo_statistical_gguf": "src.ai.demo_statistical_gguf",
    
    # データモジュール
    "data_preprocessing": "src.data.data_preprocessing",
    "sample_data": "src.data.sample_data",
    
    # 可視化モジュール
    "advanced_visualization": "src.visualization.advanced_visualization",
    "professional_reports": "src.visualization.professional_reports",
    "web_dashboard": "src.visualization.web_dashboard",
    
    # パフォーマンスモジュール
    "parallel_optimization": "src.performance.parallel_optimization",
    
    # セキュリティモジュール
    "audit_compliance_system": "src.security.audit_compliance_system",
    "booth_protection": "src.security.booth_protection",
    "trial_license_system": "src.security.trial_license_system",
    
    # 配布モジュール
    "exe_builder_system": "src.distribution.exe_builder_system",
    "booth_build_system": "src.distribution.booth_build_system",
    "booth_deployment_automation": "src.distribution.booth_deployment_automation",
    "booth_sales_manager": "src.distribution.booth_sales_manager",
    "booth_license_generator": "src.distribution.booth_license_generator",
    "generate_booth_content": "src.distribution.generate_booth_content",
    "build_exe_auto": "src.distribution.build_exe_auto"
}

# ランナーモジュールのインポートパスを更新
def update_runner_imports():
    runner_files = glob.glob("src/runners/*.py")
    
    for file_path in runner_files:
        print(f"更新中: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # インポートパターンを検索して置換
        for module, new_path in module_mapping.items():
            # 'from module import' パターン
            pattern1 = rf'from\s+{module}\s+import'
            replacement1 = f'from {new_path} import'
            content = re.sub(pattern1, replacement1, content)
            
            # 'import module' パターン
            pattern2 = rf'import\s+{module}(?!\w)'
            replacement2 = f'import {new_path}'
            content = re.sub(pattern2, replacement2, content)
            
            # 'import module as' パターン
            pattern3 = rf'import\s+{module}\s+as'
            replacement3 = f'import {new_path} as'
            content = re.sub(pattern3, replacement3, content)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

# すべてのPythonファイルのインポートパスを更新
def update_all_imports():
    python_files = []
    for root_dir in ["src"]:
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
    
    for file_path in python_files:
        print(f"更新中: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # インポートパターンを検索して置換
        for module, new_path in module_mapping.items():
            # 'from module import' パターン
            pattern1 = rf'from\s+{module}\s+import'
            replacement1 = f'from {new_path} import'
            content = re.sub(pattern1, replacement1, content)
            
            # 'import module' パターン
            pattern2 = rf'import\s+{module}(?!\w)'
            replacement2 = f'import {new_path}'
            content = re.sub(pattern2, replacement2, content)
            
            # 'import module as' パターン
            pattern3 = rf'import\s+{module}\s+as'
            replacement3 = f'import {new_path} as'
            content = re.sub(pattern3, replacement3, content)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

# メインスクリプトを更新して新しいパスを使用
def create_main_script():
    main_script = """#!/usr/bin/env python
# -*- coding: utf-8 -*-
\"\"\"
Professional Statistics Suite - メインスクリプト
新しいディレクトリ構造に対応した起動スクリプト
\"\"\"

import sys
import os

# パスを追加して新しいディレクトリ構造からのインポートを可能に
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ランナーをインポート
from src.runners.run_professional_gui import main as run_professional_gui
from src.runners.run_unified_ai_landing import main as run_unified_ai_landing
from src.runners.run_kiro_gui import main as run_kiro_gui
from src.runners.run_web_dashboard import main as run_web_dashboard

def main():
    \"\"\"メイン関数\"\"\"
    import argparse
    
    parser = argparse.ArgumentParser(description="Professional Statistics Suite")
    parser.add_argument("--mode", choices=["professional", "unified", "kiro", "web"], 
                      default="professional", help="起動モード")
    args = parser.parse_args()
    
    if args.mode == "professional":
        run_professional_gui()
    elif args.mode == "unified":
        run_unified_ai_landing()
    elif args.mode == "kiro":
        run_kiro_gui()
    elif args.mode == "web":
        run_web_dashboard()

if __name__ == "__main__":
    main()
"""
    
    with open("main.py", "w", encoding="utf-8") as f:
        f.write(main_script)
    
    print("作成: main.py (新しいディレクトリ構造対応)")

def main():
    print("🔄 インポートパスの更新を開始します...")
    
    # ランナーのインポートパスを更新
    update_runner_imports()
    
    # すべてのPythonファイルのインポートパスを更新
    update_all_imports()
    
    # メインスクリプトを作成
    create_main_script()
    
    print("✅ インポートパスの更新が完了しました")

if __name__ == "__main__":
    main() 