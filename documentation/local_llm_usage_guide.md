# ローカルLLM統計補助機能 使用方法ガイド

## 概要

このガイドでは、LMStudioのPythonライブラリを使用してGGUFファイルを推論し、統計分析のサポートを行うローカルLLM統計補助機能の使用方法を説明します。

## 前提条件

### 1. 必要なソフトウェア

- **LMStudio**: [https://lmstudio.ai/](https://lmstudio.ai/) からダウンロード
- **Python 3.8以上**
- **GGUFモデルファイル**: `models/` ディレクトリに配置

### 2. 必要なPythonライブラリ

```bash
pip install lmstudio pandas numpy scipy scikit-learn
```

### 3. GGUFモデルファイル

以下のGGUFファイルが `models/` ディレクトリに配置されていることを確認：

- `mathstral-7B-v0.1.Q8_0.gguf` (7.2GB) - 数学特化モデル
- `Phi-4-mini-reasoning-Q8_0.gguf` (3.8GB) - 推論特化モデル

## セットアップ手順

### 1. LMStudioの起動

1. LMStudioを起動
2. 左側のモデルリストから使用したいGGUFファイルを選択
3. 「Start Server」ボタンをクリック
4. サーバーが起動したら、APIエンドポイントが `http://localhost:1234` で利用可能になります

### 2. Pythonスクリプトでの使用

#### 基本的な使用方法

```python
import asyncio
from ai_integration import LMStudioProvider

async def main():
    # LMStudioプロバイダーを初期化
    provider = LMStudioProvider(models_dir="./models")
    
    # 統計クエリを分析
    result = await provider.analyze_statistical_query(
        query="データの平均値と標準偏差を計算する方法を教えてください",
        data_info={
            "rows": 100,
            "columns": 5,
            "column_names": ["group", "score", "age", "satisfaction", "time"]
        },
        user_expertise="intermediate"
    )
    
    if result["success"]:
        print(f"回答: {result['answer']}")
        print(f"信頼度: {result['confidence']}")
        print(f"推奨手法: {result['suggested_methods']}")
    else:
        print(f"エラー: {result['error']}")

# 実行
asyncio.run(main())
```

#### 統計補助システムの使用

```python
import asyncio
from local_llm_statistical_assistant import (
    LocalLLMStatisticalAssistant, 
    StatisticalQuery, 
    GGUFModelConfig
)

async def main():
    # モデル設定
    model_config = GGUFModelConfig(
        model_path="./models/mathstral-7B-v0.1.Q8_0.gguf",
        model_name="mathstral-7B-v0.1.Q8_0",
        context_size=4096,
        temperature=0.3,
        max_tokens=512
    )
    
    # 統計補助システムを初期化
    assistant = LocalLLMStatisticalAssistant(model_config)
    assistant.initialize()
    
    # 統計クエリを作成
    query = StatisticalQuery(
        query="3群の平均値の差を検定するにはどの手法を使いますか？",
        user_expertise="intermediate",
        data_info={
            "rows": 30,
            "columns": 3,
            "column_names": ["group", "score", "age"]
        }
    )
    
    # 統計クエリを分析
    response = await assistant.analyze_statistical_query(query)
    
    print(f"回答: {response.answer}")
    print(f"信頼度: {response.confidence}")
    print(f"推奨手法: {response.suggested_methods}")
    print(f"処理時間: {response.processing_time:.2f}秒")

# 実行
asyncio.run(main())
```

## 利用可能な機能

### 1. 統計クエリ分析

```python
# 記述統計
result = await provider.analyze_statistical_query(
    "データの基本統計を計算する方法を教えてください",
    user_expertise="beginner"
)

# 推論統計
result = await provider.analyze_statistical_query(
    "t検定の使い方を説明してください",
    user_expertise="intermediate"
)

# 予測分析
result = await provider.analyze_statistical_query(
    "回帰分析の手順を教えてください",
    user_expertise="advanced"
)
```

### 2. 統計手法提案

```python
# データ特性に基づく統計手法提案
data_characteristics = {
    "data_type": "continuous",
    "n_groups": 3,
    "n_samples": 100
}

suggestions = provider.suggest_statistical_method(
    data_characteristics, 
    "3群の平均値の差を検定したい"
)

for suggestion in suggestions:
    print(f"手法: {suggestion['method_name']}")
    print(f"適合性スコア: {suggestion['compatibility_score']:.2f}")
    print(f"説明: {suggestion['description']}")
```

### 3. 利用可能な統計手法

- **t検定**: 2群の平均値の差を検定
- **カイ二乗検定**: カテゴリカルデータの独立性を検定
- **相関分析**: 2変数間の関係性を分析
- **回帰分析**: 従属変数を独立変数で予測
- **分散分析**: 3群以上の平均値の差を検定
- **マンホイットニー検定**: ノンパラメトリックな2群比較

## トラブルシューティング

### 1. LMStudioサーバーが起動しない

**症状**: `Connection refused` エラーが発生

**解決方法**:
1. LMStudioが起動していることを確認
2. サーバーが `http://localhost:1234` で起動していることを確認
3. ファイアウォールの設定を確認

### 2. GGUFモデルファイルが見つからない

**症状**: `GGUFモデルファイルが見つかりません` エラー

**解決方法**:
1. `models/` ディレクトリにGGUFファイルが配置されていることを確認
2. ファイル名が正しいことを確認
3. ファイルの権限を確認

### 3. メモリ不足エラー

**症状**: `Out of memory` エラー

**解決方法**:
1. より小さいGGUFモデルを使用（例：Phi-4-mini-reasoning）
2. システムのメモリ使用量を確認
3. 他のアプリケーションを終了

### 4. 推論速度が遅い

**解決方法**:
1. GPUを使用可能なGGUFモデルを使用
2. より小さいモデルを使用
3. コンテキストサイズを調整

## パフォーマンス最適化

### 1. モデル選択

- **高速推論**: Phi-4-mini-reasoning (3.8GB)
- **高精度推論**: mathstral-7B (7.2GB)

### 2. パラメータ調整

```python
model_config = GGUFModelConfig(
    model_path="./models/Phi-4-mini-reasoning-Q8_0.gguf",
    context_size=2048,  # 小さいコンテキストで高速化
    temperature=0.1,    # 低い温度で一貫性向上
    max_tokens=256      # 短い応答で高速化
)
```

### 3. バッチ処理

```python
# 複数のクエリを一度に処理
queries = [
    "平均値の計算方法",
    "標準偏差の計算方法",
    "相関分析の手順"
]

results = []
for query in queries:
    result = await provider.analyze_statistical_query(query)
    results.append(result)
```

## 応用例

### 1. データ分析ワークフロー

```python
import pandas as pd

# データ読み込み
data = pd.read_csv("data.csv")

# データ特性を分析
data_info = {
    "rows": len(data),
    "columns": len(data.columns),
    "column_names": list(data.columns),
    "dtypes": data.dtypes.to_dict()
}

# 統計手法を提案
suggestions = provider.suggest_statistical_method(
    data_info, 
    "グループ間の差を検定したい"
)

# 推奨手法に基づいて分析を実行
for suggestion in suggestions[:3]:
    print(f"推奨手法: {suggestion['method_name']}")
    print(f"Pythonコード: {suggestion['python_code']}")
```

### 2. 教育用コンテンツ生成

```python
# 初心者向けの説明を生成
beginner_result = await provider.analyze_statistical_query(
    "t検定とは何ですか？",
    user_expertise="beginner"
)

# 上級者向けの説明を生成
advanced_result = await provider.analyze_statistical_query(
    "t検定の仮定とその検証方法を教えてください",
    user_expertise="advanced"
)
```

### 3. コード生成

```python
# 統計分析のコード例を生成
result = await provider.analyze_statistical_query(
    "Pythonでt検定を実行するコードを教えてください",
    user_expertise="intermediate"
)

if result["code_example"]:
    print("生成されたコード:")
    print(result["code_example"])
```

## まとめ

ローカルLLM統計補助機能を使用することで、以下のメリットが得られます：

1. **プライバシー保護**: データを外部に送信せずに分析
2. **高速応答**: ローカルでの推論により高速な応答
3. **カスタマイズ可能**: 統計学専用のプロンプトとデータベース
4. **教育効果**: ユーザーレベルに応じた説明

この機能を活用して、効率的で安全な統計分析を行ってください。 