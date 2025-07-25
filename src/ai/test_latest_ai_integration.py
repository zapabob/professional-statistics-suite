#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Latest AI Integration - 2025 July 25th Edition
最新AI統合システムテスト - 2025年7月25日版
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ai.latest_ai_integration import LatestAIOrchestrator
from src.ai.gguf_model_manager import GGUFModelManager

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class LatestAIIntegrationTester:
    """最新AI統合システムテストクラス"""
    
    def __init__(self):
        self.orchestrator = LatestAIOrchestrator()
        self.model_manager = GGUFModelManager()
        self.test_results = []
    
    async def test_provider_availability(self):
        """プロバイダー利用可能性テスト"""
        print("🔍 プロバイダー利用可能性テスト開始...")
        
        providers = self.orchestrator.get_available_providers()
        print(f"利用可能プロバイダー: {providers}")
        
        status = self.orchestrator.get_provider_status()
        print(f"プロバイダー状態: {status}")
        
        for provider, is_available in status.items():
            result = {
                'test': 'provider_availability',
                'provider': provider,
                'available': is_available,
                'timestamp': datetime.now().isoformat()
            }
            self.test_results.append(result)
            
            status_icon = "✅" if is_available else "❌"
            print(f"{status_icon} {provider}: {'利用可能' if is_available else '利用不可'}")
    
    async def test_basic_query(self, query: str = "t検定について説明してください"):
        """基本クエリテスト"""
        print(f"\n🔍 基本クエリテスト開始: {query}")
        
        try:
            response = await self.orchestrator.analyze_query(query)
            
            result = {
                'test': 'basic_query',
                'query': query,
                'provider_used': response.provider_used,
                'success': response.confidence > 0.5,
                'processing_time': response.processing_time,
                'tokens_consumed': response.tokens_consumed,
                'timestamp': datetime.now().isoformat()
            }
            self.test_results.append(result)
            
            print(f"✅ プロバイダー: {response.provider_used}")
            print(f"⏱️ 処理時間: {response.processing_time:.2f}秒")
            print(f"🔢 トークン数: {response.tokens_consumed}")
            print(f"📊 信頼度: {response.confidence:.2f}")
            print(f"📝 応答: {response.content[:200]}...")
            
        except Exception as e:
            print(f"❌ 基本クエリテストエラー: {e}")
            result = {
                'test': 'basic_query',
                'query': query,
                'error': str(e),
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
            self.test_results.append(result)
    
    async def test_statistical_analysis(self):
        """統計分析テスト"""
        print("\n🔍 統計分析テスト開始...")
        
        # テストデータ作成
        np.random.seed(42)
        data = pd.DataFrame({
            'group_a': np.random.normal(100, 15, 30),
            'group_b': np.random.normal(105, 15, 30),
            'age': np.random.randint(20, 60, 30),
            'score': np.random.normal(75, 10, 30)
        })
        
        queries = [
            "このデータでt検定を実行してください",
            "相関分析を行ってください",
            "回帰分析の方法を教えてください"
        ]
        
        for query in queries:
            try:
                response = await self.orchestrator.analyze_query(query, data)
                
                result = {
                    'test': 'statistical_analysis',
                    'query': query,
                    'provider_used': response.provider_used,
                    'success': response.confidence > 0.5,
                    'processing_time': response.processing_time,
                    'timestamp': datetime.now().isoformat()
                }
                self.test_results.append(result)
                
                print(f"✅ {query}")
                print(f"   プロバイダー: {response.provider_used}")
                print(f"   処理時間: {response.processing_time:.2f}秒")
                
            except Exception as e:
                print(f"❌ 統計分析テストエラー: {e}")
                result = {
                    'test': 'statistical_analysis',
                    'query': query,
                    'error': str(e),
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                }
                self.test_results.append(result)
    
    async def test_provider_fallback(self):
        """プロバイダーフォールバックテスト"""
        print("\n🔍 プロバイダーフォールバックテスト開始...")
        
        # 各プロバイダーでテスト
        providers = self.orchestrator.get_available_providers()
        
        for provider in providers:
            try:
                response = await self.orchestrator.analyze_query(
                    "簡単な統計の説明をしてください",
                    preferred_provider=provider
                )
                
                result = {
                    'test': 'provider_fallback',
                    'preferred_provider': provider,
                    'actual_provider': response.provider_used,
                    'success': response.confidence > 0.5,
                    'processing_time': response.processing_time,
                    'timestamp': datetime.now().isoformat()
                }
                self.test_results.append(result)
                
                print(f"✅ {provider} -> {response.provider_used}")
                
            except Exception as e:
                print(f"❌ {provider} フォールバックテストエラー: {e}")
                result = {
                    'test': 'provider_fallback',
                    'preferred_provider': provider,
                    'error': str(e),
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                }
                self.test_results.append(result)
    
    def test_gguf_model_manager(self):
        """GGUFモデル管理テスト"""
        print("\n🔍 GGUFモデル管理テスト開始...")
        
        # 利用可能モデル確認
        available_models = self.model_manager.get_available_models()
        print(f"利用可能モデル数: {len(available_models)}")
        
        # ダウンロード済みモデル確認
        downloaded_models = self.model_manager.list_downloaded_models()
        print(f"ダウンロード済みモデル数: {len(downloaded_models)}")
        
        # 統計分析推奨モデル
        recommended = self.model_manager.get_recommended_models("statistics")
        print(f"統計分析推奨モデル: {recommended}")
        
        # モデル統計
        stats = self.model_manager.get_model_stats()
        print(f"総ダウンロードサイズ: {stats['total_size_gb']}GB")
        
        result = {
            'test': 'gguf_model_manager',
            'available_models': len(available_models),
            'downloaded_models': len(downloaded_models),
            'total_size_gb': stats['total_size_gb'],
            'recommended_models': recommended,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
    
    def test_environment_configuration(self):
        """環境設定テスト"""
        print("\n🔍 環境設定テスト開始...")
        
        # 環境変数確認
        env_vars = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
            'OLLAMA_BASE_URL': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            'LMSTUDIO_BASE_URL': os.getenv('LMSTUDIO_BASE_URL', 'http://localhost:1234')
        }
        
        print("環境変数設定状況:")
        for var, value in env_vars.items():
            if value:
                print(f"✅ {var}: 設定済み")
            else:
                print(f"❌ {var}: 未設定")
        
        result = {
            'test': 'environment_configuration',
            'env_vars': {k: bool(v) for k, v in env_vars.items()},
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
    
    def generate_test_report(self):
        """テストレポート生成"""
        print("\n📊 テストレポート生成...")
        
        # 成功/失敗統計
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result.get('success', False))
        failed_tests = total_tests - successful_tests
        
        print(f"総テスト数: {total_tests}")
        print(f"成功: {successful_tests}")
        print(f"失敗: {failed_tests}")
        print(f"成功率: {(successful_tests/total_tests*100):.1f}%")
        
        # テスト結果をJSONファイルに保存
        report_file = Path("test_results") / f"latest_ai_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        report_data = {
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate': successful_tests/total_tests*100 if total_tests > 0 else 0
            },
            'test_results': self.test_results,
            'timestamp': datetime.now().isoformat()
        }
        
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"📄 レポート保存: {report_file}")
        
        return report_data
    
    async def run_all_tests(self):
        """全テスト実行"""
        print("🚀 最新AI統合システムテスト開始")
        print("=" * 50)
        
        # テスト実行
        await self.test_provider_availability()
        await self.test_basic_query()
        await self.test_statistical_analysis()
        await self.test_provider_fallback()
        self.test_gguf_model_manager()
        self.test_environment_configuration()
        
        # レポート生成
        report = self.generate_test_report()
        
        print("\n" + "=" * 50)
        print("🎉 テスト完了！")
        
        return report

async def main():
    """メイン関数"""
    tester = LatestAIIntegrationTester()
    report = await tester.run_all_tests()
    
    # 結果サマリー
    summary = report['test_summary']
    print(f"\n📈 テスト結果サマリー:")
    print(f"成功率: {summary['success_rate']:.1f}%")
    print(f"成功: {summary['successful_tests']}/{summary['total_tests']}")

if __name__ == "__main__":
    asyncio.run(main()) 