#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__init__.pyファイル作成スクリプト
"""

import os

# 作成するディレクトリのリスト
directories = [
    "../src",
    "../src/core",
    "../src/gui",
    "../src/statistics",
    "../src/ai",
    "../src/data",
    "../src/visualization",
    "../src/performance",
    "../src/security",
    "../src/distribution",
    "../src/tests",
    "../src/runners"
]

# 各ディレクトリに__init__.pyファイルを作成
for directory in directories:
    # ディレクトリが存在するか確認
    if not os.path.exists(directory):
        print(f"ディレクトリが存在しません: {directory}")
        continue
    
    # __init__.pyファイルのパス
    init_file = os.path.join(directory, "__init__.py")
    
    # パッケージ名（ディレクトリ名の最後の部分）
    package_name = os.path.basename(directory)
    
    # __init__.pyファイルの内容
    content = f'"""\n{package_name} package\n"""\n\n'
    
    # ファイルを作成して内容を書き込む
    with open(init_file, "w") as f:
        f.write(content)
    
    print(f"作成: {init_file}")

print("__init__.pyファイルの作成が完了しました") 