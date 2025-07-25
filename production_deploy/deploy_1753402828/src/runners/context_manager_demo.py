#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Context Manager Demo
コンテキスト管理システムのデモンストレーション
"""

import asyncio
from ai_integration import ContextManager

async def demo_context_management():
    """コンテキスト管理システムのデモ"""
    print("🤖 コンテキスト管理システム デモンストレーション")
    print("=" * 50)
    
    # ContextManagerを初期化
    context_manager = ContextManager()
    
    # 新しいユーザーセッションを作成
    print("\n1. 新しいユーザーセッションを作成")
    context = context_manager.get_or_create_context(
        user_id="demo_user",
        session_id="demo_session_001",
        data_fingerprint="sales_data_2024"
    )
    
    print(f"   ユーザーID: {context.user_id}")
    print(f"   セッションID: {context.session_id}")
    print(f"   専門レベル: {context.user_expertise_level}")
    print(f"   開始時刻: {context.timestamp}")
    
    # ユーザー設定を更新
    print("\n2. ユーザー設定を更新")
    new_preferences = {
        "preferred_visualization": "matplotlib",
        "explanation_style": "detailed",
        "language": "ja"
    }
    context_manager.update_user_preferences(context, new_preferences)
    print(f"   更新された設定: {context.user_preferences}")
    
    # 分析履歴を追加
    print("\n3. 分析履歴を追加")
    analysis_results = [
        {
            'type': 'descriptive',
            'method': 'basic_stats',
            'success': True,
            'query': '売上データの基本統計を計算してください',
            'provider': 'openai',
            'processing_time': 1.2,
            'tokens_consumed': 120
        },
        {
            'type': 'inferential',
            'method': 't_test',
            'success': True,
            'query': '地域間の売上に差があるか検定してください',
            'provider': 'anthropic',
            'processing_time': 2.1,
            'tokens_consumed': 180
        },
        {
            'type': 'predictive',
            'method': 'machine_learning',
            'success': True,
            'query': '来月の売上を予測してください',
            'provider': 'google',
            'processing_time': 5.3,
            'tokens_consumed': 350
        }
    ]
    
    for result in analysis_results:
        context_manager.update_context(context, result)
        print(f"   追加: {result['method']} ({result['type']})")
    
    # 専門レベルの評価
    print("\n4. 専門レベルの動的評価")
    expertise_level = context_manager.get_user_expertise_level(context)
    print(f"   現在の専門レベル: {expertise_level}")
    
    # 分析パターンの取得
    print("\n5. 分析パターンの分析")
    patterns = context_manager.get_analysis_patterns(context)
    print(f"   総分析数: {patterns['total_analyses']}")
    print(f"   成功率: {patterns['success_rate']:.2%}")
    print(f"   平均処理時間: {patterns['average_processing_time']:.2f}秒")
    print(f"   よく使用する分析タイプ: {patterns['most_used_analysis_types']}")
    print(f"   好みのプロバイダー: {patterns['preferred_providers']}")
    
    # コンテキストタグを追加
    print("\n6. コンテキストタグの管理")
    tags = ["sales_analysis", "quarterly_report", "business_intelligence"]
    for tag in tags:
        context_manager.add_context_tag(context, tag)
    print(f"   追加されたタグ: {context.context_tags}")
    
    # 学習進捗を更新
    print("\n7. 学習進捗の追跡")
    learning_concepts = {
        "t_test": 0.8,
        "anova": 0.6,
        "regression": 0.9,
        "machine_learning": 0.7
    }
    for concept, progress in learning_concepts.items():
        context_manager.update_learning_progress(context, concept, progress)
    print(f"   学習進捗: {context.learning_progress}")
    
    # お気に入り手法を追加
    print("\n8. お気に入り手法の管理")
    favorite_methods = ["t_test", "regression", "correlation", "anova"]
    for method in favorite_methods:
        context_manager.add_favorite_method(context, method)
    print(f"   お気に入り手法: {context.favorite_methods}")
    
    # 最近のクエリを追加
    print("\n9. 最近のクエリ履歴")
    recent_queries = [
        "売上データの相関分析を実行してください",
        "季節性を考慮した時系列分析をお願いします",
        "顧客セグメンテーションを行ってください"
    ]
    for query in recent_queries:
        context_manager.add_recent_query(context, query)
    print(f"   最近のクエリ: {context.recent_queries[:3]}")
    
    # コンテキストに基づく推奨事項
    print("\n10. コンテキスト推奨事項")
    recommendations = context_manager.get_contextual_recommendations(context)
    print(f"   推奨手法: {recommendations['suggested_methods']}")
    print(f"   学習機会: {recommendations['learning_opportunities']}")
    print(f"   ワークフロー改善: {recommendations['workflow_improvements']}")
    
    # コンテキスト対応応答の生成
    print("\n11. コンテキスト対応応答の生成")
    base_response = "分析が完了しました。統計的に有意な結果が得られています。"
    
    # 詳細説明スタイル
    context.user_preferences['explanation_style'] = 'detailed'
    detailed_response = context_manager.generate_context_aware_response(context, base_response)
    print(f"   詳細応答: {detailed_response}")
    
    # 簡潔スタイル
    context.user_preferences['explanation_style'] = 'concise'
    context.user_expertise_level = 'expert'
    concise_response = context_manager.generate_context_aware_response(context, base_response)
    print(f"   簡潔応答: {concise_response}")
    
    # セッション要約
    print("\n12. セッション要約")
    summary = context_manager.get_session_summary(context)
    print(f"   セッション開始: {summary['session_start']}")
    print(f"   現在の専門レベル: {summary['current_expertise_level']}")
    print(f"   プライバシー設定: {summary['privacy_settings']}")
    print(f"   ユーザー設定: {summary['user_preferences']}")
    
    # コンテキストの永続化確認
    print("\n13. コンテキストの永続化確認")
    context_manager._save_context_to_disk(context)
    
    # 新しいContextManagerで読み込み
    new_manager = ContextManager()
    loaded_context = new_manager.get_or_create_context(
        "demo_user", "demo_session_001", "sales_data_2024"
    )
    
    print(f"   永続化確認: 履歴数 {len(loaded_context.analysis_history)}")
    print(f"   永続化確認: 専門レベル {loaded_context.user_expertise_level}")
    print(f"   永続化確認: タグ数 {len(loaded_context.context_tags)}")
    
    print("\n✅ コンテキスト管理システムのデモが完了しました！")

if __name__ == "__main__":
    asyncio.run(demo_context_management())