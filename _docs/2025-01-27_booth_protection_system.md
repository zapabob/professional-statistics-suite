# Booth版保護システム実装ログ

## 実施日
2025年1月27日

## 概要
商用配布（Booth版）向けの包括的なリバースエンジニアリング対策システムを実装しました。

## 🔒 保護機能一覧

### 1. アンチデバッグ保護
- **プロセス監視**: x64dbg, IDA Pro, CheatEngine等の検出
- **Windows API活用**: IsDebuggerPresent, CheckRemoteDebuggerPresent
- **Python固有**: sys.gettrace() によるデバッガー検出
- **実行時間監視**: ステップ実行検出（異常に遅い実行の検出）

### 2. ライセンス管理システム
- **マシン固有ID**: CPU、MAC、マザーボード、HDDシリアルからID生成
- **暗号化ライセンス**: PBKDF2 + Fernet暗号化
- **オンライン認証**: ライセンスサーバーとの連携
- **期限管理**: 有効期限チェック

### 3. コード難読化
- **文字列難読化**: hex エンコーディング
- **変数名難読化**: 意味のない名前への置換
- **ダミーコード挿入**: 解析を困難にする無意味な関数

### 4. 整合性チェック
- **ファイルハッシュ検証**: SHA256による改ざん検出
- **実行時チェック**: スタックトレース検証

### 5. ビルドシステム
- **PyInstaller統合**: 単一実行ファイル化
- **UPX圧縮**: 追加圧縮・難読化
- **隠しインポート**: 必要なライブラリの自動包含

## 📁 ファイル構成

```
booth_protection.py          # 保護システム本体
booth_build_system.py        # ビルドシステム
requirements_booth.txt       # Booth版用依存関係
```

## 🛡️ 保護レベル

### MAXIMUM（最高レベル）
- 全保護機能有効
- 5秒間隔でのデバッガー監視
- 即座にプロセス終了

### HIGH（高レベル）
- 主要保護機能有効
- 10秒間隔での監視

### MEDIUM（中レベル）
- 基本保護機能のみ

## 🔧 使用方法

### 1. 保護システムの統合

```python
from booth_protection import booth_protection, require_license, anti_debug

# 保護システム初期化
protection_result = booth_protection.initialize_protection()
if not protection_result["success"]:
    print(f"ライセンスエラー: {protection_result['error']}")
    sys.exit(1)

# ライセンス必須機能
@require_license("premium")
def premium_analysis():
    pass

# アンチデバッグ保護
@anti_debug
def sensitive_function():
    pass
```

### 2. Booth版ビルド

```bash
# 依存関係インストール
pip install -r requirements_booth.txt

# ビルド実行
python booth_build_system.py
```

### 3. ライセンスサーバー（別途実装が必要）

```python
# FastAPI サーバー例
@app.post("/api/verify")
async def verify_license(request: LicenseRequest):
    # ライセンス検証ロジック
    return {"valid": True, "license_data": {...}}
```

## 🔐 セキュリティ対策詳細

### デバッガー対策
1. **プロセス名検出**: 一般的なリバースエンジニアリングツール
2. **API フック**: Windows Kernel32 API
3. **実行時間分析**: ステップ実行の検出

### コード保護
1. **文字列暗号化**: 重要な文字列をhex化
2. **制御フロー難読化**: ダミー関数による混乱
3. **バイナリパッキング**: UPX圧縮

### ライセンス認証
1. **ハードウェアフィンガープリンティング**: マシン固有識別
2. **時限ライセンス**: 期限切れ自動無効化
3. **オンライン検証**: サーバー側認証

## ⚠️ 制限事項

### 1. Python特有の制限
- バイトコード復元可能性（uncompyle6等）
- 動的解析に対する限界

### 2. 対策の限界
- 完全な保護は不可能
- 高度な攻撃者に対する時間稼ぎが目的

### 3. パフォーマンス影響
- アンチデバッグ機能による若干のオーバーヘッド
- 暗号化処理による起動時間増加

## 🚀 追加推奨対策

### 1. サーバーサイド処理
```python
# 重要な計算をサーバーで実行
async def server_side_analysis(data):
    response = await api_client.post("/analyze", data)
    return response.json()
```

### 2. 分割配布
- 重要なアルゴリズムを別ライブラリで配布
- DLL/SO形式での提供

### 3. 定期アップデート
- 保護機能の定期更新
- 新しい解析ツールへの対応

## 📋 Booth配布チェックリスト

### ビルド前
- [ ] 保護システムの動作確認
- [ ] ライセンスサーバーの準備
- [ ] アイコン・リソースファイルの準備
- [ ] ドキュメントの準備

### ビルド実行
- [ ] `python booth_build_system.py` 実行
- [ ] 生成されたexeファイルの動作確認
- [ ] ライセンス認証のテスト
- [ ] ウイルス対策ソフトでの誤検知確認

### 配布前
- [ ] パッケージの圧縮・暗号化
- [ ] 購入者向けドキュメント作成
- [ ] サポート体制の構築

## 💡 今後の改善予定

1. **より高度な難読化**: LLVM Obfuscator 統合
2. **仮想マシン検出**: VMware, VirtualBox 検出
3. **クラウド認証**: AWS/Azure 連携
4. **ハードウェア暗号化**: TPM チップ活用

## 📞 サポート情報

### トラブルシューティング
- ライセンス認証エラー
- ウイルス対策ソフトでの誤検知
- 特定環境での動作不良

### 更新情報
- 新しい保護機能の追加
- 既知の脆弱性への対応
- パフォーマンス改善 