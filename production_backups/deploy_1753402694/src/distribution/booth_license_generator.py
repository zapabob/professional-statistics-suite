#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BOOTH License Generator
BOOTH商品番号と購入者固有番号を使用したライセンス発行システム
"""

import json
import hashlib
import re
import sqlite3
import smtplib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import qrcode

class BoothLicenseGenerator:
    """BOOTH購入情報からライセンス自動発行"""
    
    def __init__(self, db_path: str = "booth_licenses.db"):
        self.db_path = db_path
        self.init_database()
        
        # BOOTH商品番号とエディションのマッピング
        self.product_mapping = {
            "1234567": {  # 例：Lite版の商品番号
                "edition": "Lite",
                "price": 2980,
                "trial_days": 7
            },
            "1234568": {  # 例：Standard版の商品番号
                "edition": "Standard", 
                "price": 9800,
                "trial_days": 0
            },
            "1234569": {  # 例：Professional版の商品番号
                "edition": "Professional",
                "price": 29800,
                "trial_days": 0
            },
            "1234570": {  # 例：GPU Accelerated版の商品番号
                "edition": "GPU_Accelerated",
                "price": 49800,
                "trial_days": 0
            }
        }
        
        # メール設定
        self.email_config = {
            "imap_server": "imap.gmail.com",
            "imap_port": 993,
            "smtp_server": "smtp.gmail.com", 
            "smtp_port": 587,
            "username": "",  # 設定ファイルから読み込み
            "password": "",  # 設定ファイルから読み込み
            "from_address": "licenses@statistics-suite.com"
        }
        
        self.load_email_config()
    
    def init_database(self):
        """データベース初期化"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # BOOTH購入情報テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS booth_purchases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_number TEXT NOT NULL,
                buyer_id TEXT NOT NULL,
                buyer_email TEXT,
                purchase_date DATETIME,
                amount INTEGER,
                processed BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 発行済みライセンステーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS issued_licenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                license_key TEXT UNIQUE NOT NULL,
                product_number TEXT NOT NULL,
                buyer_id TEXT NOT NULL,
                buyer_email TEXT,
                edition TEXT NOT NULL,
                issue_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                expiry_date DATETIME,
                activation_status TEXT DEFAULT 'pending',
                machine_id TEXT,
                activation_date DATETIME,
                download_count INTEGER DEFAULT 0
            )
        """)
        
        # メール処理ログテーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS email_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT,
                sender TEXT,
                subject TEXT,
                body TEXT,
                processed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                license_issued BOOLEAN DEFAULT FALSE,
                error_message TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def load_email_config(self):
        """メール設定読み込み"""
        config_file = "booth_email_config.json"
        if Path(config_file).exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.email_config.update(config)
    
    def generate_booth_license_key(self, product_number: str, buyer_id: str, 
                                 edition: Optional[str] = None) -> str:
        """BOOTH購入情報を使用したライセンスキー生成"""
        
        # エディション情報取得
        if not edition and product_number in self.product_mapping:
            edition = self.product_mapping[product_number]["edition"]
        
        # エディションコード
        edition_codes = {
            "Lite": "LT",
            "Standard": "ST", 
            "Professional": "PR",
            "GPU_Accelerated": "GA"
        }
        edition_code = edition_codes.get(edition if edition else "Unknown", "XX")
        
        # BOOTH購入情報を使用したハッシュ生成
        hash_input = f"{product_number}{buyer_id}{datetime.now().isoformat()}"
        hash_obj = hashlib.sha256(hash_input.encode())
        
        # ライセンスキー形式: PSS-ED-PRODUCT-BUYER
        # 例: PSS-PR-1234567-3F5BBBA4
        license_key = f"PSS-{edition_code}-{product_number}-{buyer_id.upper()}"
        
        return license_key
    
    def parse_booth_notification_email(self, email_content: str) -> Optional[Dict[str, str]]:
        """BOOTH通知メールを解析して購入情報抽出"""
        
        # BOOTHメール形式のパターンマッチング
        patterns = {
            "product_number": r"商品番号[：:]\s*(\d{7})",
            "buyer_id": r"購入者固有番号[：:]\s*([a-fA-F0-9]{8})",
            "buyer_email": r"購入者メール[：:]\s*([\w\.-]+@[\w\.-]+\.\w+)",
            "amount": r"金額[：:]\s*¥?([\d,]+)",
            "purchase_date": r"購入日時[：:]\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"
        }
        
        # 代替パターン（英語版BOOTH対応）
        alt_patterns = {
            "product_number": r"Product\s+ID[：:]\s*(\d{7})",
            "buyer_id": r"Buyer\s+ID[：:]\s*([a-fA-F0-9]{8})",
            "buyer_email": r"Buyer\s+Email[：:]\s*([\w\.-]+@[\w\.-]+\.\w+)",
            "amount": r"Amount[：:]\s*¥?([\d,]+)",
            "purchase_date": r"Purchase\s+Date[：:]\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"
        }
        
        extracted_data = {}
        
        # 日本語パターンで抽出
        for key, pattern in patterns.items():
            match = re.search(pattern, email_content, re.IGNORECASE)
            if match:
                extracted_data[key] = match.group(1)
        
        # 英語パターンで補完
        for key, pattern in alt_patterns.items():
            if key not in extracted_data:
                match = re.search(pattern, email_content, re.IGNORECASE)
                if match:
                    extracted_data[key] = match.group(1)
        
        # 必須フィールドチェック
        required_fields = ["product_number", "buyer_id"]
        if all(field in extracted_data for field in required_fields):
            return extracted_data
        
        return None
    
    def process_booth_purchase(self, purchase_data: Dict[str, str]) -> Dict[str, Any]:
        """BOOTH購入情報を処理してライセンス発行"""
        
        try:
            product_number = purchase_data["product_number"]
            buyer_id = purchase_data["buyer_id"]
            buyer_email = purchase_data.get("buyer_email")
            
            # 商品情報取得
            if product_number not in self.product_mapping:
                return {
                    "success": False,
                    "error": f"未知の商品番号: {product_number}"
                }
            
            product_info = self.product_mapping[product_number]
            edition = product_info["edition"]
            
            # 重複購入チェック
            if self._is_duplicate_purchase(product_number, buyer_id):
                return {
                    "success": False,
                    "error": "既に処理済みの購入です"
                }
            
            # ライセンスキー生成
            license_key = self.generate_booth_license_key(
                product_number, buyer_id, edition
            )
            
            # 有効期限設定
            expiry_date = None
            if product_info["trial_days"] > 0:
                expiry_date = datetime.now() + timedelta(days=product_info["trial_days"])
            else:
                expiry_date = datetime.now() + timedelta(days=365)  # 1年間
            
            # データベースに記録
            self._save_purchase_and_license(
                purchase_data, license_key, edition, expiry_date
            )
            
            # ライセンス配布メール送信
            if buyer_email:
                self._send_license_email(
                    buyer_email, license_key, edition, purchase_data
                )
            
            return {
                "success": True,
                "license_key": license_key,
                "edition": edition,
                "buyer_id": buyer_id,
                "expiry_date": expiry_date.isoformat(),
                "email_sent": bool(buyer_email)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _is_duplicate_purchase(self, product_number: str, buyer_id: str) -> bool:
        """重複購入チェック"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM booth_purchases 
            WHERE product_number = ? AND buyer_id = ?
        """, (product_number, buyer_id))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def _save_purchase_and_license(self, purchase_data: Dict[str, str], 
                                 license_key: str, edition: str, 
                                 expiry_date: datetime):
        """購入情報とライセンス情報をデータベースに保存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 購入情報保存
        cursor.execute("""
            INSERT INTO booth_purchases 
            (product_number, buyer_id, buyer_email, purchase_date, amount, processed)
            VALUES (?, ?, ?, ?, ?, TRUE)
        """, (
            purchase_data["product_number"],
            purchase_data["buyer_id"], 
            purchase_data.get("buyer_email"),
            purchase_data.get("purchase_date"),
            purchase_data.get("amount", "").replace(",", "")
        ))
        
        # ライセンス情報保存
        cursor.execute("""
            INSERT INTO issued_licenses
            (license_key, product_number, buyer_id, buyer_email, edition, expiry_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            license_key,
            purchase_data["product_number"],
            purchase_data["buyer_id"],
            purchase_data.get("buyer_email"),
            edition,
            expiry_date
        ))
        
        conn.commit()
        conn.close()
    
    def _send_license_email(self, buyer_email: str, license_key: str, 
                          edition: str, purchase_data: Dict[str, str]):
        """ライセンス配布メール送信"""
        try:
            # メール内容生成
            subject = f"✅ Professional Statistics Suite {edition} Edition - ライセンス発行完了"
            
            body = self._generate_license_email_body(
                buyer_email, license_key, edition, purchase_data
            )
            
            # メール送信
            if self.email_config["username"] and self.email_config["password"]:
                self._send_email(buyer_email, subject, body)
                
                # QRコード添付版も送信
                self._send_license_email_with_qr(
                    buyer_email, license_key, edition, subject, body
                )
                
        except Exception as e:
            print(f"⚠️ メール送信失敗: {str(e)}")
    
    def _generate_license_email_body(self, buyer_email: str, license_key: str,
                                   edition: str, purchase_data: Dict[str, str]) -> str:
        """ライセンスメール本文生成"""
        
        product_info = self.product_mapping.get(purchase_data["product_number"], {})
        price = product_info.get("price", "不明")
        
        expiry_info = ""
        if product_info.get("trial_days", 0) > 0:
            expiry_date = datetime.now() + timedelta(days=product_info["trial_days"])
            expiry_info = f"\n有効期限: {expiry_date.strftime('%Y年%m月%d日')} ({product_info['trial_days']}日間)"
        else:
            expiry_date = datetime.now() + timedelta(days=365)
            expiry_info = f"\n有効期限: {expiry_date.strftime('%Y年%m月%d日')} (1年間)"
        
        body = f"""
{buyer_email} 様

この度は Professional Statistics Suite v2.0 ({edition} Edition) をご購入いただき、
誠にありがとうございます！

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔑 ライセンス情報
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ライセンスキー: {license_key}
エディション: {edition} Edition
購入金額: ¥{price:,}{expiry_info}

BOOTH購入情報:
- 商品番号: {purchase_data['product_number']}
- 購入者ID: {purchase_data['buyer_id']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📥 ダウンロード・インストール
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. BOOTHから実行ファイルをダウンロード
2. StatisticsSuite_Booth.exe を実行
3. 初回起動時に上記ライセンスキーを入力
4. 認証完了後、{edition}版の全機能が利用可能になります

📱 QRコードでの認証も可能です（別途添付）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 {edition} Edition の主な機能
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{self._get_edition_features_for_email(edition)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎓 クイックスタート
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. サンプルデータで動作確認
2. 基本統計から始める
3. グラフ作成で可視化
4. PDF レポート出力

📖 詳細マニュアル: https://statistics-suite.com/manual
🎥 チュートリアル動画: https://youtube.com/@statistics-suite

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📞 サポート
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

技術的なご質問やトラブルがございましたら、
お気軽にお問い合わせください。

📧 メール: support@statistics-suite.com
💬 Discord: https://discord.gg/statistics-suite
📖 FAQ: https://statistics-suite.com/faq

サポート対象: {self._get_support_level(edition)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎁 特典情報
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

・次回アップデート無料提供
・Discord プレミアムコミュニティ参加権
・統計解析テンプレート無料配布
・論文作成支援ツール（Professional版以上）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

今後とも Professional Statistics Suite をよろしくお願いいたします。

Professional Statistics Suite 開発チーム
https://statistics-suite.com
"""
        return body.strip()
    
    def _get_edition_features_for_email(self, edition: str) -> str:
        """メール用エディション機能説明"""
        features = {
            "Lite": """
✅ 基本統計（平均、分散、相関など）
✅ 基本的なグラフ作成（ヒストグラム、散布図）
✅ CSV データ出力
⏰ 7日間無料体験版
🔄 Standard版へのアップグレード可能""",
            
            "Standard": """
✅ 全統計機能（t検定、ANOVA、回帰分析等）
✅ プロジェクト保存・管理
✅ PDF/HTML レポート自動生成
✅ 基本AI機能（解析結果の自動解釈）
✅ Excel データ連携
💎 Professional版へのアップグレード特典あり""",
            
            "Professional": """
✅ Standard版の全機能
⚡ GPU加速対応（RTX30/40シリーズで最大10倍高速）
🧠 AI統合分析（ChatGPT/Claude連携）
🎨 高度な可視化（3D グラフ、アニメーション）
📞 優先技術サポート
🔧 カスタムプラグイン対応""",
            
            "GPU_Accelerated": """
✅ Professional版の全機能
🚀 RTX最適化による究極のパフォーマンス
🔧 専用1対1技術サポート
⚙️ カスタマイズ機能開発対応
🆕 新機能ベータ版先行アクセス
👑 プレミアムコミュニティ特別招待"""
        }
        
        return features.get(edition, "機能情報を取得中...")
    
    def _get_support_level(self, edition: str) -> str:
        """エディション別サポートレベル"""
        support_levels = {
            "Lite": "Discord コミュニティサポート",
            "Standard": "メール 24時間以内返信保証",
            "Professional": "メール優先対応 + Discord プレミアム",
            "GPU_Accelerated": "専用1対1サポート + リモート設定支援"
        }
        
        return support_levels.get(edition, "標準サポート")
    
    def _send_email(self, to_email: str, subject: str, body: str):
        """メール送信"""
        if not self.email_config["username"]:
            print("⚠️ メール設定が不完全です")
            return
        
        msg = MIMEMultipart()
        msg['From'] = self.email_config["from_address"]
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        server = smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"])
        server.starttls()
        server.login(self.email_config["username"], self.email_config["password"])
        
        text = msg.as_string()
        server.sendmail(self.email_config["from_address"], to_email, text)
        server.quit()
    
    def _send_license_email_with_qr(self, buyer_email: str, license_key: str,
                                   edition: str, subject: str, body: str):
        """QRコード付きライセンスメール送信"""
        try:
            # QRコード生成
            qr_path = self._generate_license_qr_code(license_key)
            
            msg = MIMEMultipart()
            msg['From'] = self.email_config["from_address"]
            msg['To'] = buyer_email
            msg['Subject'] = subject + " (QRコード付き)"
            
            # 本文にQRコード説明追加
            qr_body = body + """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📱 QRコード認証
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

添付のQRコードをスマートフォンで読み取ると、
ライセンス認証ページに直接アクセスできます。

手動入力の手間を省けて便利です！
"""
            
            msg.attach(MIMEText(qr_body, 'plain', 'utf-8'))
            
            # QRコード添付
            if Path(qr_path).exists():
                with open(qr_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= "license_qr_{license_key.replace("-", "_")}.png"'
                )
                msg.attach(part)
            
            # 送信
            server = smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"])
            server.starttls()
            server.login(self.email_config["username"], self.email_config["password"])
            
            text = msg.as_string()
            server.sendmail(self.email_config["from_address"], buyer_email, text)
            server.quit()
            
        except Exception as e:
            print(f"⚠️ QRコード付きメール送信失敗: {str(e)}")
    
    def _generate_license_qr_code(self, license_key: str) -> str:
        """ライセンス用QRコード生成"""
        activation_url = f"https://statistics-suite.com/activate?key={license_key}"
        
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        qr.add_data(activation_url)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # QRコードディレクトリ作成
        qr_dir = Path("qr_codes")
        qr_dir.mkdir(exist_ok=True)
        
        qr_path = qr_dir / f"license_{license_key.replace('-', '_')}.png"
        img.save(str(qr_path))
        
        return str(qr_path)
    
    def manual_license_issue(self, product_number: str, buyer_id: str, 
                           buyer_email: str = None) -> Dict[str, Any]:
        """手動ライセンス発行"""
        purchase_data = {
            "product_number": product_number,
            "buyer_id": buyer_id,
            "buyer_email": buyer_email,
            "purchase_date": datetime.now().isoformat(),
            "amount": str(self.product_mapping.get(product_number, {}).get("price", 0))
        }
        
        return self.process_booth_purchase(purchase_data)
    
    def get_license_status(self, license_key: str) -> Optional[Dict[str, Any]]:
        """ライセンス状態取得"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT license_key, edition, issue_date, expiry_date, 
                   activation_status, machine_id, activation_date, download_count
            FROM issued_licenses WHERE license_key = ?
        """, (license_key,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "license_key": result[0],
                "edition": result[1], 
                "issue_date": result[2],
                "expiry_date": result[3],
                "activation_status": result[4],
                "machine_id": result[5],
                "activation_date": result[6],
                "download_count": result[7]
            }
        
        return None
    
    def generate_sales_report(self) -> str:
        """売上レポート生成"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 基本統計
        cursor.execute("""
            SELECT 
                COUNT(*) as total_licenses,
                COUNT(DISTINCT buyer_id) as unique_buyers,
                SUM(CASE WHEN processed = TRUE THEN 1 ELSE 0 END) as processed_purchases
            FROM booth_purchases
        """)
        
        basic_stats = cursor.fetchone()
        
        # エディション別売上
        cursor.execute("""
            SELECT 
                il.edition,
                COUNT(*) as count,
                SUM(bp.amount) as revenue
            FROM issued_licenses il
            JOIN booth_purchases bp ON il.buyer_id = bp.buyer_id AND il.product_number = bp.product_number
            GROUP BY il.edition
            ORDER BY revenue DESC
        """)
        
        edition_stats = cursor.fetchall()
        
        conn.close()
        
        # レポート生成
        report = f"""
# BOOTH ライセンス発行レポート

**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 基本統計

- **総ライセンス発行数**: {basic_stats[0]:,}
- **ユニーク購入者数**: {basic_stats[1]:,}
- **処理済み購入**: {basic_stats[2]:,}

## 💰 エディション別売上

| エディション | 販売数 | 売上額 |
|-------------|--------|--------|
"""
        
        total_revenue = 0
        for edition, count, revenue in edition_stats:
            revenue = revenue or 0
            total_revenue += revenue
            report += f"| {edition} | {count:,} | ¥{revenue:,} |\n"
        
        report += f"""

**総売上**: ¥{total_revenue:,}

---

*自動生成レポート - BOOTH License Generator v1.0*
"""
        
        # レポートファイル保存
        report_path = f"reports/booth_license_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        Path("reports").mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report_path

def main():
    """BOOTH ライセンス発行システム実行"""
    generator = BoothLicenseGenerator()
    
    print("🎯 BOOTH License Generator v1.0")
    print("=" * 50)
    
    while True:
        print("\n📋 メニュー:")
        print("1. 手動ライセンス発行")
        print("2. ライセンス状態確認")
        print("3. 売上レポート生成")
        print("4. 商品番号マッピング表示")
        print("5. テスト用購入情報処理")
        print("0. 終了")
        
        choice = input("\n選択してください: ").strip()
        
        if choice == "1":
            product_number = input("商品番号 (7桁): ").strip()
            buyer_id = input("購入者固有番号 (8桁英数字): ").strip()
            buyer_email = input("購入者メール (オプション): ").strip() or None
            
            result = generator.manual_license_issue(product_number, buyer_id, buyer_email)
            
            if result["success"]:
                print("✅ ライセンス発行完了:")
                print(f"   ライセンスキー: {result['license_key']}")
                print(f"   エディション: {result['edition']}")
                print(f"   有効期限: {result['expiry_date']}")
                print(f"   メール送信: {'済' if result['email_sent'] else '未'}")
            else:
                print(f"❌ ライセンス発行失敗: {result['error']}")
        
        elif choice == "2":
            license_key = input("ライセンスキー: ").strip()
            status = generator.get_license_status(license_key)
            
            if status:
                print("✅ ライセンス情報:")
                for key, value in status.items():
                    print(f"   {key}: {value}")
            else:
                print("❌ ライセンスが見つかりません")
        
        elif choice == "3":
            report_path = generator.generate_sales_report()
            print(f"✅ 売上レポートを {report_path} に生成しました")
        
        elif choice == "4":
            print("\n📋 商品番号マッピング:")
            for product_id, info in generator.product_mapping.items():
                print(f"   {product_id}: {info['edition']} (¥{info['price']:,})")
        
        elif choice == "5":
            # テスト用
            test_data = {
                "product_number": "1234569",
                "buyer_id": "3f5bbba4",
                "buyer_email": "test@example.com",
                "purchase_date": datetime.now().isoformat(),
                "amount": "29800"
            }
            
            result = generator.process_booth_purchase(test_data)
            print(f"テスト結果: {result}")
        
        elif choice == "0":
            print("👋 お疲れ様でした！")
            break
        
        else:
            print("❌ 無効な選択です")

if __name__ == "__main__":
    main() 