#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BOOTH商品ページコンテンツ自動生成
"""

from .booth_sales_manager import BoothSalesManager

def main():
    print("🎯 BOOTH商品ページコンテンツ生成中...")
    
    # 販売管理システム初期化
    manager = BoothSalesManager()
    
    # 1. ローンチクーポン生成
    print("🎁 ローンチクーポン生成中...")
    coupons = manager.create_launch_coupons()
    print("✅ ローンチクーポン生成完了:")
    for code, message in coupons.items():
        print(f"   {code}: {message}")
    
    # 2. BOOTH商品ページコンテンツ生成
    print("\n📝 BOOTH商品ページコンテンツ生成中...")
    content = manager.generate_booth_listing_content()
    
    # ファイル保存
    with open("booth_listing_content.md", "w", encoding="utf-8") as f:
        f.write(content)
    
    print("✅ BOOTH商品ページコンテンツを booth_listing_content.md に保存しました")
    
    # 3. 価格計算例
    print("\n💰 価格計算例（クーポン適用）:")
    
    editions = ["Lite", "Standard", "Professional", "GPU_Accelerated"]
    coupon = "LAUNCH30"
    
    for edition in editions:
        pricing = manager.calculate_price_with_coupon(edition, coupon)
        print(f"   {edition}: ¥{pricing['base_price']:,} → ¥{int(pricing['final_price']):,} ({pricing['savings_percent']:.0f}% OFF)")
    
    # 4. サンプルライセンス生成
    print("\n🔑 サンプルライセンスキー生成:")
    for edition in editions:
        license_key = manager.generate_license_key(edition, "demo@example.com")
        print(f"   {edition}: {license_key}")
    
    print("\n🎉 BOOTH販売準備完了！")
    print("📄 商品ページコンテンツ: booth_listing_content.md")
    print("💾 販売データベース: booth_sales.db")
    print("🎁 ローンチクーポン: LAUNCH30 (30% OFF, 500本限定)")

if __name__ == "__main__":
    main() 