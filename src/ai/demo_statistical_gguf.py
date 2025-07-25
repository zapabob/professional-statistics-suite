#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GGUF統計解析デモンストレーション
ローカルGGUFモデルを使った統計解析AIシステムのデモ
"""

import sys
import time

# 設定とライセンス
try:
    from config import check_feature_permission
    if not check_feature_permission('advanced_ai'):
        print("❌ Advanced AI機能にはProfessional版以上が必要です")
        sys.exit(1)
except ImportError:
    print("⚠️ ライセンス確認をスキップします（開発モード）".encode('utf-8').decode(sys.stdout.encoding, 'ignore'))

# AI統合モジュール
from src.ai.ai_integration import AIStatisticalAnalyzer

# スタブ関数（gguf_test_helperの代替）
def setup_gguf_test_environment():
    return {
        'skip_tests': True,
        'error': 'GGUF test helper not available',
        'model_path': None
    }

def create_test_queries():
    return ["基本的な統計質問"]

def create_statistical_context():
    return {
        'dataset_info': 'サンプルデータセット',
        'analysis_goal': '探索的データ解析'
    }

def validate_gguf_response(response):
    return {'score': 50}

def print_test_summary(env):
    print("GGUF test environment summary")

def demonstrate_gguf_integration():
    """GGUF統合機能のデモンストレーション"""
    print("🚀 GGUF統計解析AIシステム デモンストレーション")
    print("=" * 60)
    
    # 1. 環境セットアップ
    print("\n📋 Step 1: 環境セットアップ")
    test_env = setup_gguf_test_environment()
    
    if test_env['skip_tests']:
        print(f"❌ {test_env['error']}")
        print("\n💡 解決方法:")
        print("   1. GGUFモデルファイルを ./models/ に配置")
        print("   2. 環境変数 GGUF_MODEL_PATH を設定")
        print("   3. llama-cpp-python をインストール: pip install llama-cpp-python")
        return False
    
    print_test_summary(test_env)
    
    # 2. AIStatisticalAnalyzer初期化
    print("\n🧠 Step 2: AI統計解析器の初期化")
    try:
        analyzer = AIStatisticalAnalyzer(gguf_model_path=test_env['model_path'])
        
        if 'gguf' in analyzer.providers:
            print("✅ GGUFプロバイダー初期化成功")
            print(f"📊 現在のプロバイダー: {analyzer.current_provider}")
            
            # モデル情報表示
            gguf_provider = analyzer.providers['gguf']
            model_info = gguf_provider.get_model_info()
            print(f"🔧 GPU有効: {model_info['gpu_enabled']}")
            print(f"📏 コンテキストサイズ: {model_info['context_size']}")
        else:
            print("❌ GGUFプロバイダー初期化失敗")
            return False
            
    except Exception as e:
        print(f"❌ AI統計解析器初期化エラー: {e}")
        return False
    
    # 3. 基本的な統計質問
    print("\n📊 Step 3: 基本的な統計質問")
    basic_questions = [
        "平均と中央値の違いを教えてください。",
        "t検定とは何ですか？",
        "相関係数の解釈方法を説明してください。"
    ]
    
    for i, question in enumerate(basic_questions, 1):
        print(f"\n🔍 質問 {i}: {question}")
        
        start_time = time.time()
        result = analyzer.analyze_statistical_question(question)
        processing_time = time.time() - start_time
        
        if result['success']:
            print("✅ 回答生成成功")
            print(f"⏱️  処理時間: {processing_time:.2f}秒")
            print(f"🔢 トークン消費: {result['tokens_consumed']}")
            print(f"📝 回答:\n{result['response'][:200]}...")
            
            # 応答品質評価
            validation = validate_gguf_response(result['response'])
            print(f"📊 品質スコア: {validation['score']}/100")
            
        else:
            print(f"❌ 回答生成失敗: {result['error']}")
    
    # 4. コンテキスト付き解析
    print("\n🎯 Step 4: コンテキスト付き統計解析")
    context = create_statistical_context()
    contextual_question = "このデータセットに適した統計手法を推奨してください。"
    
    print(f"📋 データセット情報: {context['dataset_info']}")
    print(f"🎯 解析目標: {context['analysis_goal']}")
    print(f"🔍 質問: {contextual_question}")
    
    start_time = time.time()
    result = analyzer.analyze_statistical_question(contextual_question, context)
    processing_time = time.time() - start_time
    
    if result['success']:
        print("✅ コンテキスト解析成功")
        print(f"⏱️  処理時間: {processing_time:.2f}秒")
        print(f"📝 推奨手法:\n{result['response'][:300]}...")
    else:
        print(f"❌ コンテキスト解析失敗: {result['error']}")
    
    # 5. 性能ベンチマーク
    print("\n⚡ Step 5: 性能ベンチマーク")
    test_queries = create_test_queries()
    
    # 最初の5つのクエリでベンチマーク
    benchmark_queries = dict(list(test_queries.items())[:5])
    
    total_time = 0
    total_tokens = 0
    successful_queries = 0
    
    for query_name, query_text in benchmark_queries.items():
        print(f"🔄 実行中: {query_name}")
        
        start_time = time.time()
        result = analyzer.analyze_statistical_question(query_text)
        query_time = time.time() - start_time
        
        if result['success']:
            successful_queries += 1
            total_time += query_time
            total_tokens += result['tokens_consumed']
            print(f"   ✅ 成功 ({query_time:.2f}秒)")
        else:
            print(f"   ❌ 失敗: {result['error']}")
    
    if successful_queries > 0:
        avg_time = total_time / successful_queries
        avg_tokens = total_tokens / successful_queries
        
        print("\n📈 ベンチマーク結果:")
        print(f"   成功率: {successful_queries}/{len(benchmark_queries)} ({successful_queries/len(benchmark_queries)*100:.1f}%)")
        print(f"   平均処理時間: {avg_time:.2f}秒")
        print(f"   平均トークン数: {avg_tokens:.0f}")
        print(f"   総処理時間: {total_time:.2f}秒")
    
    # 6. 高度な統計解析シナリオ
    print("\n🎓 Step 6: 高度な統計解析シナリオ")
    advanced_scenarios = [
        {
            "scenario": "実験計画法",
            "question": "2要因の分散分析を実行する際の前提条件と解釈方法を教えてください。",
            "context": {
                "analysis_type": "experimental_design",
                "factors": 2,
                "dependent_variable": "continuous"
            }
        },
        {
            "scenario": "機械学習統計",
            "question": "回帰分析における多重共線性の問題と対処法を説明してください。",
            "context": {
                "analysis_type": "regression",
                "problem": "multicollinearity",
                "data_type": "observational"
            }
        }
    ]
    
    for scenario in advanced_scenarios:
        print(f"\n🔬 シナリオ: {scenario['scenario']}")
        print(f"❓ 質問: {scenario['question']}")
        
        start_time = time.time()
        result = analyzer.analyze_statistical_question(
            scenario['question'], 
            scenario['context']
        )
        processing_time = time.time() - start_time
        
        if result['success']:
            print("✅ 高度解析成功")
            print(f"⏱️  処理時間: {processing_time:.2f}秒")
            print(f"📝 専門的回答:\n{result['response'][:250]}...")
            
            # 専門性評価
            validation = validate_gguf_response(result['response'])
            statistical_terms = validation['details'].get('statistical_terms', [])
            print(f"🎯 統計用語数: {len(statistical_terms)}")
            print(f"📊 専門性スコア: {validation['score']}/100")
        else:
            print(f"❌ 高度解析失敗: {result['error']}")
    
    print("\n🎉 デモンストレーション完了！")
    print("=" * 60)
    
    return True

def interactive_gguf_session():
    """インタラクティブなGGUF統計解析セッション"""
    print("\n🎮 インタラクティブモード開始")
    print("統計学に関する質問を入力してください（'quit'で終了）")
    print("-" * 50)
    
    # 環境セットアップ
    test_env = setup_gguf_test_environment()
    if test_env['skip_tests']:
        print(f"❌ {test_env['error']}")
        return
    
    # AI解析器初期化
    try:
        analyzer = AIStatisticalAnalyzer(gguf_model_path=test_env['model_path'])
        if 'gguf' not in analyzer.providers:
            print("❌ GGUFプロバイダーが利用できません")
            return
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        return
    
    session_count = 0
    
    while True:
        try:
            # ユーザー入力
            question = input(f"\n[{session_count + 1}] 統計質問> ").strip()
            
            if question.lower() in ['quit', 'exit', '終了', 'q']:
                print("👋 セッション終了")
                break
            
            if not question:
                continue
            
            # 質問処理
            print("🤔 考え中...")
            start_time = time.time()
            
            result = analyzer.analyze_statistical_question(question)
            processing_time = time.time() - start_time
            
            if result['success']:
                print(f"\n📝 回答 ({processing_time:.2f}秒):")
                print("-" * 40)
                print(result['response'])
                print("-" * 40)
                print(f"🔢 トークン: {result['tokens_consumed']}")
                
                # 品質評価
                validation = validate_gguf_response(result['response'])
                if validation['score'] < 50:
                    print("⚠️ 回答品質が低い可能性があります")
                
            else:
                print(f"❌ エラー: {result['error']}")
            
            session_count += 1
            
        except KeyboardInterrupt:
            print("\n👋 セッション中断")
            break
        except Exception as e:
            print(f"❌ セッションエラー: {e}")

def main():
    """メイン実行関数"""
    print("🧪 Professional Statistics Suite - GGUF Integration Demo")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--interactive':
            interactive_gguf_session()
            return
        elif sys.argv[1] == '--help':
            print("使用方法:")
            print("  py -3 demo_statistical_gguf.py           # デモンストレーション実行")
            print("  py -3 demo_statistical_gguf.py --interactive  # インタラクティブモード")
            print("  py -3 demo_statistical_gguf.py --help         # ヘルプ表示")
            return
    
    # デモンストレーション実行
    success = demonstrate_gguf_integration()
    
    if success:
        print("\n💡 次のステップ:")
        print("   - インタラクティブモード: py -3 demo_statistical_gguf.py --interactive")
        print("   - テスト実行: py -3 test_gguf_integration.py")
        print("   - モックテスト: py -3 test_gguf_mock.py")
    else:
        print("\n🔧 トラブルシューティング:")
        print("   1. llama-cpp-python インストール: pip install llama-cpp-python")
        print("   2. GGUFモデル配置: ./models/your-model.gguf")
        print("   3. 環境変数設定: set GGUF_MODEL_PATH=path/to/model.gguf")

if __name__ == '__main__':
    main()