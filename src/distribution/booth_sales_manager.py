#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BOOTH Sales Manager
販売戦略統合管理システム
"""

import uuid
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import sqlite3
import qrcode
import matplotlib.pyplot as plt

class BoothSalesManager:
    """BOOTH販売管理システム"""
    
    def __init__(self, db_path: str = "booth_sales.db"):
        self.db_path = db_path
        self.init_database()
        self.editions = {
            "Lite": {
                "price": 2980,
                "features": ["基本統計", "可視化", "CSV出力"],
                "limitations": ["プロジェクト保存不可", "AI機能制限", "GPU非対応"]
            },
            "Standard": {
                "price": 9800,
                "features": ["全統計機能", "プロジェクト保存", "PDF出力", "基本AI機能"],
                "limitations": ["GPU制限", "高度AI機能制限"]
            },
            "Professional": {
                "price": 29800,
                "features": ["全機能", "GPU加速", "AI統合", "技術サポート"],
                "limitations": []
            },
            "GPU_Accelerated": {
                "price": 49800,
                "features": ["最高性能", "RTX最適化", "専用サポート", "カスタマイズ"],
                "limitations": []
            }
        }
        
    def init_database(self):
        """データベース初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # ライセンステーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS licenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                license_key TEXT UNIQUE NOT NULL,
                edition TEXT NOT NULL,
                customer_email TEXT,
                machine_id TEXT,
                purchase_date DATETIME,
                expiry_date DATETIME,
                status TEXT DEFAULT 'active',
                coupon_used TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # クーポンテーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS coupons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                discount_percent INTEGER NOT NULL,
                valid_from DATETIME,
                valid_until DATETIME,
                usage_limit INTEGER DEFAULT 1,
                used_count INTEGER DEFAULT 0,
                target_edition TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 売上統計テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sales_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                edition TEXT,
                quantity INTEGER,
                revenue REAL,
                coupon_code TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 顧客管理テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE,
                name TEXT,
                university TEXT,
                purchase_history TEXT,
                referral_code TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def generate_license_key(self, edition: str, customer_email: str = None) -> str:
        """ライセンスキー生成"""
        # 形式: PSS-EDITION-XXXXXXXX-XXXXXXXX
        prefix = "PSS"
        edition_code = {
            "Lite": "LT",
            "Standard": "ST",
            "Professional": "PR",
            "GPU_Accelerated": "GA"
        }.get(edition, "XX")
        
        # ランダム部分生成
        unique_data = f"{datetime.now().isoformat()}{customer_email or ''}{uuid.uuid4()}"
        hash_obj = hashlib.sha256(unique_data.encode())
        hash_hex = hash_obj.hexdigest()[:16].upper()
        
        license_key = f"{prefix}-{edition_code}-{hash_hex[:8]}-{hash_hex[8:]}"
        
        # データベースに保存
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        expiry_date = datetime.now() + timedelta(days=365)  # 1年間有効
        
        cursor.execute("""
            INSERT INTO licenses (license_key, edition, customer_email, expiry_date)
            VALUES (?, ?, ?, ?)
        """, (license_key, edition, customer_email, expiry_date))
        
        conn.commit()
        conn.close()
        
        return license_key
    
    def create_launch_coupons(self) -> Dict[str, str]:
        """ローンチクーポン生成"""
        coupons = {}
        
        # 30%OFFローンチクーポン（500本限定）
        launch_code = "LAUNCH30"
        coupons[launch_code] = self.create_coupon(
            code=launch_code,
            discount_percent=30,
            valid_days=30,
            usage_limit=500,
            description="ローンチ記念30%OFF"
        )
        
        # 学生割10%追加クーポン
        student_code = "STUDENT10"
        coupons[student_code] = self.create_coupon(
            code=student_code,
            discount_percent=10,
            valid_days=365,
            usage_limit=1000,
            description="学生限定追加10%OFF"
        )
        
        # リファラルクーポン
        referral_code = "REFER20"
        coupons[referral_code] = self.create_coupon(
            code=referral_code,
            discount_percent=20,
            valid_days=90,
            usage_limit=100,
            description="紹介者限定20%OFF"
        )
        
        return coupons
    
    def create_coupon(self, code: str, discount_percent: int, valid_days: int, 
                     usage_limit: int = 1, target_edition: str = None,
                     description: str = "") -> str:
        """クーポン作成"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        valid_from = datetime.now()
        valid_until = valid_from + timedelta(days=valid_days)
        
        try:
            cursor.execute("""
                INSERT INTO coupons (code, discount_percent, valid_from, valid_until, 
                                   usage_limit, target_edition)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (code, discount_percent, valid_from, valid_until, usage_limit, target_edition))
            
            conn.commit()
            return f"✅ クーポン '{code}' を作成しました ({discount_percent}% OFF, {valid_days}日間有効)"
            
        except sqlite3.IntegrityError:
            return f"❌ クーポン '{code}' は既に存在します"
        finally:
            conn.close()
    
    def calculate_price_with_coupon(self, edition: str, coupon_code: str = None) -> Dict[str, Any]:
        """クーポン適用価格計算"""
        base_price = self.editions[edition]["price"]
        final_price = base_price
        discount = 0
        
        if coupon_code:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT discount_percent, valid_until, usage_limit, used_count, target_edition
                FROM coupons WHERE code = ? AND valid_until > datetime('now')
            """, (coupon_code,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                discount_percent, valid_until, usage_limit, used_count, target_edition = result
                
                # 使用制限チェック
                if used_count < usage_limit:
                    # エディション制限チェック
                    if not target_edition or target_edition == edition:
                        discount = base_price * (discount_percent / 100)
                        final_price = base_price - discount
        
        return {
            "edition": edition,
            "base_price": base_price,
            "discount": discount,
            "final_price": final_price,
            "coupon_code": coupon_code,
            "savings_percent": (discount / base_price * 100) if discount > 0 else 0
        }
    
    def generate_booth_listing_content(self) -> str:
        """BOOTH商品ページコンテンツ生成"""
        content = """
# 🎯 IBM SPSS 1/20価格でGPU加速！日本語統計ソフト Professional Statistics Suite v2.0

## 💡 なぜ選ばれるのか？

**IBM SPSS**: ¥500,000+ / 年額  
**Professional Statistics Suite**: ¥2,980～ (買い切り)

### 🚀 主な特徴
- ✅ **GPU加速対応** - RTX30/40シリーズで最大10倍高速
- ✅ **完全日本語対応** - UI・ドキュメント・サポート全て日本語
- ✅ **AI統合分析** - ChatGPT/Claude連携で高度な解釈
- ✅ **卒論対応** - 大学院生が開発、研究現場のニーズを完全理解

---

## 📊 エディション比較

| 機能 | Lite<br/>¥2,980 | Standard<br/>¥9,800 | Professional<br/>¥29,800 | GPU Accelerated<br/>¥49,800 |
|------|:---:|:---:|:---:|:---:|
| 基本統計 | ✅ | ✅ | ✅ | ✅ |
| 高度統計 | ❌ | ✅ | ✅ | ✅ |
| プロジェクト保存 | ❌ | ✅ | ✅ | ✅ |
| AI機能 | 制限有 | 基本 | 完全 | 完全 |
| GPU加速 | ❌ | ❌ | ✅ | ✅ |
| 技術サポート | Discord | メール | 優先 | 専用 |

---

## 🎁 限定特典

### 🔥 ローンチ記念 30%OFF
**クーポンコード: LAUNCH30**  
※先着500本限定

### 🎓 学生限定 追加10%OFF
**クーポンコード: STUDENT10**  
※.ac.jpメールで自動適用

### 👥 紹介プログラム
3人紹介でStandard無償アップグレード！

---

## 📱 デモ動画
[60秒で分かる！起動→解析→PDF出力](動画URL)

## 💾 無料体験
[Lite版体験ダウンロード](ダウンロードURL)
※機能制限版、保存不可

---

## 📞 サポート
- 📧 メール: support@statistics-suite.com
- 💬 Discord: [コミュニティ参加](Discord URL)
- 📖 ドキュメント: [オンラインマニュアル](マニュアルURL)

---

## 🛡️ 返金保証
**Lite版**: 7日間無条件返金保証

---

**🔥 今すぐ始めよう！卒論・研究・ビジネス分析を革新**
"""
        
        return content
    
    def create_activation_email_template(self, license_key: str, edition: str, 
                                      customer_email: str) -> str:
        """アクティベーションメールテンプレート生成"""
        template = f"""
件名: ✅ Professional Statistics Suite ライセンス認証情報

{customer_email} 様

この度は Professional Statistics Suite v2.0 ({edition}) をご購入いただき、
誠にありがとうございます！

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔑 ライセンス情報
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ライセンスキー: {license_key}
エディション: {edition}
有効期限: {(datetime.now() + timedelta(days=365)).strftime('%Y年%m月%d日')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📥 ダウンロード・インストール
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. BOOTHから実行ファイルをダウンロード
2. StatisticsSuite_Booth.exe を実行
3. 初回起動時にライセンスキーを入力
4. 認証完了後、全機能が利用可能になります

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 はじめ方
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📖 クイックスタートガイド: [URL]
🎥 チュートリアル動画: [URL]
💬 Discordコミュニティ: [URL]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📞 サポート
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

技術的なご質問やトラブルがございましたら、
お気軽にお問い合わせください。

📧 メール: support@statistics-suite.com
💬 Discord: [コミュニティURL]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

今後ともProfessional Statistics Suiteをよろしくお願いいたします。

Professional Statistics Suite開発チーム
"""
        return template
    
    def generate_qr_code_for_activation(self, license_key: str, output_path: str = None) -> str:
        """アクティベーション用QRコード生成"""
        activation_url = f"https://your-domain.com/activate?key={license_key}"
        
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(activation_url)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        if not output_path:
            output_path = f"qr_codes/activation_{license_key.replace('-', '_')}.png"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        
        return output_path
    
    def generate_sales_dashboard(self) -> str:
        """販売ダッシュボード生成"""
        conn = sqlite3.connect(self.db_path)
        
        # 売上データ取得
        import pandas as pd
        
        licenses_df = pd.read_sql("""
            SELECT edition, DATE(created_at) as date, COUNT(*) as sales,
                   SUM(CASE 
                       WHEN edition = 'Lite' THEN 2980
                       WHEN edition = 'Standard' THEN 9800  
                       WHEN edition = 'Professional' THEN 29800
                       WHEN edition = 'GPU_Accelerated' THEN 49800
                       ELSE 0
                   END) as revenue
            FROM licenses 
            GROUP BY edition, DATE(created_at)
            ORDER BY date DESC
        """, conn)
        
        coupons_df = pd.read_sql("""
            SELECT code, used_count, usage_limit, discount_percent
            FROM coupons
            ORDER BY used_count DESC
        """, conn)
        
        conn.close()
        
        # グラフ作成
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. エディション別売上
        if not licenses_df.empty:
            edition_sales = licenses_df.groupby('edition')['sales'].sum()
            ax1.pie(edition_sales.values, labels=edition_sales.index, autopct='%1.1f%%')
            ax1.set_title('Edition Sales Distribution')
        
        # 2. 日別売上推移
        if not licenses_df.empty:
            daily_revenue = licenses_df.groupby('date')['revenue'].sum()
            ax2.plot(daily_revenue.index, daily_revenue.values)
            ax2.set_title('Daily Revenue Trend')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. クーポン使用状況
        if not coupons_df.empty:
            ax3.bar(coupons_df['code'], coupons_df['used_count'])
            ax3.set_title('Coupon Usage')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. 累計売上
        if not licenses_df.empty:
            cumulative_revenue = licenses_df.groupby('date')['revenue'].sum().cumsum()
            ax4.plot(cumulative_revenue.index, cumulative_revenue.values)
            ax4.set_title('Cumulative Revenue')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        dashboard_path = "reports/sales_dashboard.png"
        Path(dashboard_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return dashboard_path

def main():
    """販売管理システム実行"""
    manager = BoothSalesManager()
    
    print("🎯 BOOTH Sales Manager v1.0")
    print("=" * 50)
    
    while True:
        print("\n📋 メニュー:")
        print("1. ライセンスキー生成")
        print("2. クーポン作成")
        print("3. 価格計算（クーポン適用）")
        print("4. ローンチクーポン生成")
        print("5. BOOTH商品ページ生成")
        print("6. アクティベーションメール生成")
        print("7. 販売ダッシュボード生成")
        print("8. QRコード生成")
        print("0. 終了")
        
        choice = input("\n選択してください: ").strip()
        
        if choice == "1":
            edition = input("エディション (Lite/Standard/Professional/GPU_Accelerated): ").strip()
            email = input("顧客メール (オプション): ").strip() or None
            
            if edition in manager.editions:
                license_key = manager.generate_license_key(edition, email)
                print(f"✅ ライセンスキー生成: {license_key}")
            else:
                print("❌ 無効なエディションです")
        
        elif choice == "2":
            code = input("クーポンコード: ").strip()
            discount = int(input("割引率 (%): "))
            days = int(input("有効日数: "))
            limit = int(input("使用制限回数: "))
            
            result = manager.create_coupon(code, discount, days, limit)
            print(result)
        
        elif choice == "3":
            edition = input("エディション: ").strip()
            coupon = input("クーポンコード (オプション): ").strip() or None
            
            if edition in manager.editions:
                pricing = manager.calculate_price_with_coupon(edition, coupon)
                print("\n💰 価格計算結果:")
                print(f"エディション: {pricing['edition']}")
                print(f"定価: ¥{pricing['base_price']:,}")
                print(f"割引: ¥{pricing['discount']:,}")
                print(f"最終価格: ¥{pricing['final_price']:,}")
                print(f"割引率: {pricing['savings_percent']:.1f}%")
            else:
                print("❌ 無効なエディションです")
        
        elif choice == "4":
            coupons = manager.create_launch_coupons()
            print("🎁 ローンチクーポン生成完了:")
            for code, message in coupons.items():
                print(f"  {code}: {message}")
        
        elif choice == "5":
            content = manager.generate_booth_listing_content()
            with open("booth_listing_content.md", "w", encoding="utf-8") as f:
                f.write(content)
            print("✅ BOOTH商品ページコンテンツを booth_listing_content.md に保存しました")
        
        elif choice == "6":
            license_key = input("ライセンスキー: ").strip()
            edition = input("エディション: ").strip()
            email = input("顧客メール: ").strip()
            
            email_content = manager.create_activation_email_template(license_key, edition, email)
            with open(f"activation_email_{license_key.replace('-', '_')}.txt", "w", encoding="utf-8") as f:
                f.write(email_content)
            print("✅ アクティベーションメールテンプレートを生成しました")
        
        elif choice == "7":
            dashboard_path = manager.generate_sales_dashboard()
            print(f"✅ 販売ダッシュボードを {dashboard_path} に生成しました")
        
        elif choice == "8":
            license_key = input("ライセンスキー: ").strip()
            qr_path = manager.generate_qr_code_for_activation(license_key)
            print(f"✅ QRコードを {qr_path} に生成しました")
        
        elif choice == "0":
            print("👋 お疲れ様でした！")
            break
        
        else:
            print("❌ 無効な選択です")

if __name__ == "__main__":
    main() 