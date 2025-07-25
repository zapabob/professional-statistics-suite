# リポジトリ再構成 - 2025-07-25

## 📋 作業概要

**作業日時**: 2025年7月25日  
**担当者**: Professional Statistics Suite 開発チーム  
**作業内容**: リポジトリ構造の整理と最適化

## 🧹 整理作業内容

### 1. ディレクトリ構造の最適化
- モジュール別にディレクトリを分割
- 関連機能のグループ化
- インポートパスの整理

### 2. 新しいディレクトリ構造
```
professional-statistics-suite/
├── src/                      # ソースコード
│   ├── core/                 # コアモジュール
│   ├── gui/                  # GUIモジュール
│   ├── statistics/           # 統計解析モジュール
│   ├── ai/                   # AI統合モジュール
│   ├── data/                 # データ処理モジュール
│   ├── visualization/        # 可視化モジュール
│   ├── performance/          # パフォーマンス最適化
│   ├── security/             # セキュリティモジュール
│   ├── distribution/         # 配布関連モジュール
│   ├── tests/                # テストモジュール
│   └── runners/              # 実行スクリプト
├── documentation/            # ドキュメント
│   └── implementation_logs/  # 実装ログ
├── resources/                # リソースファイル
│   ├── templates/            # テンプレート
│   └── models/               # モデルファイル
├── reports/                  # レポート出力
├── logs/                     # ログファイル
├── checkpoints/              # チェックポイント
├── backup/                   # バックアップ
└── cache/                    # キャッシュ
```

## 📊 整理効果

### ✅ 開発効率向上
- **コード可読性**: 構造化による向上
- **保守性**: モジュール分離による向上
- **拡張性**: アーキテクチャ明確化

### ✅ 品質向上
- **専門性**: 機能別の明確な分離
- **テスト容易性**: モジュール単位でのテスト
- **ドキュメント整理**: 実装ログの集約

## 🔍 確認項目

- [x] ディレクトリ構造の作成
- [x] ファイルの適切な移動
- [x] __init__.pyファイルの作成
- [x] ドキュメントの整理
- [x] リソースファイルの整理
- [x] データストレージディレクトリの設定

## 🚀 移行手順

1. **ディレクトリ構造作成**
   ```powershell
   mkdir src; mkdir src\core; mkdir src\gui; mkdir src\statistics; mkdir src\ai; mkdir src\data; mkdir src\visualization; mkdir src\performance; mkdir src\security; mkdir src\distribution; mkdir src\tests; mkdir src\runners; mkdir documentation; mkdir resources
   ```

2. **ファイル移動**
   - コアモジュール移動
     ```powershell
     cp main.py src\core\; cp config.py src\core\; cp professional_utils.py src\core\
     ```
   - GUI モジュール移動
     ```powershell
     cp HAD_Statistics_GUI.py src\gui\; cp professional_statistics_gui.py src\gui\; cp unified_ai_landing_gui.py src\gui\; cp kiro_integrated_gui.py src\gui\
     ```
   - 統計モジュール移動
     ```powershell
     cp advanced_statistics.py src\statistics\; cp bayesian_analysis.py src\statistics\; cp survival_analysis.py src\statistics\; cp statistical_method_advisor.py src\statistics\; cp assumption_validator.py src\statistics\
     ```
   - AI モジュール移動
     ```powershell
     cp ai_integration.py src\ai\; cp contextual_retriever.py src\ai\; cp gguf_model_selector.py src\ai\; cp local_llm_statistical_assistant.py src\ai\
     ```

3. **インポートパス更新**
   - 各モジュールのインポートパスを新しい構造に合わせて更新
   - 相対インポートから絶対インポートへの変更

4. **__init__.pyファイル作成**
   - 各パッケージディレクトリに__init__.pyファイルを作成
   - パッケージ内のモジュールをエクスポート

5. **ドキュメント整理**
   - _docsディレクトリの内容をdocumentation/implementation_logsに移動
   - README.mdとARCHITECTURE.mdをdocumentationに移動

6. **リソースファイル整理**
   - テンプレートとモデルファイルをresourcesディレクトリに移動
   - 設定ファイルをresourcesディレクトリに移動

## 📝 今後の課題

- **インポートパスの完全移行**: すべてのファイルで新しいインポートパスを使用するように更新
- **テストの更新**: 新しいディレクトリ構造に合わせてテストを更新
- **ドキュメントの更新**: 新しいディレクトリ構造に関するドキュメントの更新
- **ビルドスクリプトの更新**: 新しいディレクトリ構造に合わせてビルドスクリプトを更新 