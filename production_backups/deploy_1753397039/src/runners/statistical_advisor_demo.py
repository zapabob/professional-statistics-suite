#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistical Method Advisor Demo
統計手法アドバイザーのデモンストレーション
"""

import numpy as np
import pandas as pd
from ai_integration import StatisticalMethodAdvisor

def create_demo_datasets():
    """デモ用データセットを作成"""
    np.random.seed(42)
    
    datasets = {}
    
    # 1. 正規分布データ（大サンプル）
    datasets['large_normal'] = pd.DataFrame({
        'treatment_group': np.random.choice(['A', 'B', 'C'], 300),
        'score_before': np.random.normal(50, 10, 300),
        'score_after': np.random.normal(55, 12, 300),
        'age': np.random.randint(20, 70, 300),
        'gender': np.random.choice(['Male', 'Female'], 300)
    })
    
    # 2. 小サンプルデータ
    datasets['small_sample'] = pd.DataFrame({
        'group': np.random.choice(['Control', 'Treatment'], 12),
        'measurement': np.random.normal(25, 5, 12),
        'baseline': np.random.normal(20, 4, 12)
    })
    
    # 3. カテゴリカルデータ
    datasets['categorical'] = pd.DataFrame({
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 200),
        'job_satisfaction': np.random.choice(['Low', 'Medium', 'High'], 200),
        'department': np.random.choice(['Sales', 'Engineering', 'Marketing', 'HR'], 200),
        'performance_rating': np.random.choice(['Poor', 'Average', 'Good', 'Excellent'], 200)
    })
    
    # 4. 欠損値を含むデータ
    missing_data = pd.DataFrame({
        'variable1': np.random.normal(0, 1, 100),
        'variable2': np.random.normal(5, 2, 100),
        'variable3': np.random.normal(-2, 1.5, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    # 意図的に欠損値を作成
    missing_indices = np.random.choice(100, 20, replace=False)
    missing_data.loc[missing_indices, 'variable1'] = np.nan
    missing_indices = np.random.choice(100, 15, replace=False)
    missing_data.loc[missing_indices, 'variable2'] = np.nan
    datasets['missing_data'] = missing_data
    
    # 5. 時系列風データ
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    datasets['time_series'] = pd.DataFrame({
        'date': dates,
        'sales': np.random.normal(1000, 200, 100) + np.sin(np.arange(100) * 0.1) * 100,
        'temperature': np.random.normal(20, 5, 100),
        'promotion': np.random.choice([0, 1], 100, p=[0.8, 0.2])
    })
    
    return datasets

def demo_data_characteristics_analysis():
    """データ特性分析のデモ"""
    print("🔍 データ特性分析デモ")
    print("=" * 50)
    
    advisor = StatisticalMethodAdvisor()
    datasets = create_demo_datasets()
    
    for name, data in datasets.items():
        print(f"\n📊 データセット: {name}")
        print(f"   サイズ: {data.shape[0]}行 × {data.shape[1]}列")
        
        characteristics = advisor.analyze_data_characteristics(data)
        
        print(f"   データ品質スコア: {characteristics.data_quality_score:.3f}")
        print(f"   列タイプ: {characteristics.column_types}")
        
        # 欠損データ情報
        missing_info = {col: f"{ratio:.1%}" for col, ratio in characteristics.missing_data_pattern.items() if ratio > 0}
        if missing_info:
            print(f"   欠損データ: {missing_info}")
        
        # 分布特性（数値列のみ）
        numeric_cols = [col for col, dtype in characteristics.column_types.items() if dtype == 'numeric']
        if numeric_cols and characteristics.distribution_characteristics:
            print("   分布特性:")
            for col in numeric_cols[:3]:  # 最初の3列のみ表示
                if col in characteristics.distribution_characteristics:
                    dist_info = characteristics.distribution_characteristics[col]
                    is_normal = dist_info.get('normality_test', {}).get('is_normal', False)
                    skewness = dist_info.get('skewness', 0)
                    print(f"     {col}: 正規性={is_normal}, 歪度={skewness:.2f}")

def demo_method_suggestions():
    """手法推奨のデモ"""
    print("\n\n🎯 統計手法推奨デモ")
    print("=" * 50)
    
    advisor = StatisticalMethodAdvisor()
    datasets = create_demo_datasets()
    
    # 様々な研究質問でのデモ
    research_scenarios = [
        {
            'dataset': 'large_normal',
            'question': 'グループ間に有意な差があるか検定したい',
            'expertise': 'intermediate',
            'description': '大サンプル・推測統計'
        },
        {
            'dataset': 'small_sample',
            'question': '平均値を比較したい',
            'expertise': 'intermediate',
            'description': '小サンプル・比較分析'
        },
        {
            'dataset': 'categorical',
            'question': 'カテゴリ間の関連性を調べたい',
            'expertise': 'beginner',
            'description': 'カテゴリカルデータ・関連性分析'
        },
        {
            'dataset': 'large_normal',
            'question': '将来の値を予測したい',
            'expertise': 'advanced',
            'description': '予測分析'
        },
        {
            'dataset': 'large_normal',
            'question': 'データの基本的な統計量を知りたい',
            'expertise': 'beginner',
            'description': '記述統計'
        }
    ]
    
    for scenario in research_scenarios:
        print(f"\n📋 シナリオ: {scenario['description']}")
        print(f"   データセット: {scenario['dataset']}")
        print(f"   研究質問: {scenario['question']}")
        print(f"   ユーザー専門レベル: {scenario['expertise']}")
        
        data = datasets[scenario['dataset']]
        characteristics = advisor.analyze_data_characteristics(data)
        
        suggestions = advisor.suggest_methods(
            characteristics,
            scenario['question'],
            scenario['expertise']
        )
        
        print(f"\n   💡 推奨手法 (上位{len(suggestions)}つ):")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion.method_name}")
            print(f"      信頼度: {suggestion.confidence_score:.3f}")
            print(f"      根拠: {suggestion.rationale}")
            print(f"      仮定: {', '.join(suggestion.assumptions) if suggestion.assumptions else 'なし'}")
            print(f"      推定計算時間: {suggestion.estimated_computation_time:.1f}秒")
            
            if scenario['expertise'] == 'beginner' and suggestion.educational_content:
                print(f"      教育的説明: {suggestion.educational_content[:100]}...")
            
            if suggestion.alternative_methods:
                print(f"      代替手法: {', '.join(suggestion.alternative_methods[:2])}")
            print()

def demo_expertise_level_adaptation():
    """専門レベル適応のデモ"""
    print("\n\n🎓 専門レベル適応デモ")
    print("=" * 50)
    
    advisor = StatisticalMethodAdvisor()
    data = create_demo_datasets()['large_normal']
    characteristics = advisor.analyze_data_characteristics(data)
    
    question = "グループ間の差を検定したい"
    expertise_levels = ['beginner', 'intermediate', 'advanced']
    
    print(f"研究質問: {question}")
    print(f"データサイズ: {data.shape}")
    
    for expertise in expertise_levels:
        print(f"\n👤 専門レベル: {expertise}")
        suggestions = advisor.suggest_methods(characteristics, question, expertise)
        
        if suggestions:
            top_suggestion = suggestions[0]
            print(f"   推奨手法: {top_suggestion.method_name}")
            print(f"   信頼度: {top_suggestion.confidence_score:.3f}")
            print("   教育的コンテンツ:")
            print(f"   {top_suggestion.educational_content}")
            
            if len(suggestions) > 1:
                print(f"   その他の推奨: {', '.join([s.method_name for s in suggestions[1:3]])}")

def demo_data_quality_impact():
    """データ品質の影響デモ"""
    print("\n\n📈 データ品質の影響デモ")
    print("=" * 50)
    
    advisor = StatisticalMethodAdvisor()
    datasets = create_demo_datasets()
    
    # 高品質データ vs 低品質データ
    high_quality_data = datasets['large_normal']
    low_quality_data = datasets['missing_data']
    
    question = "変数間の関係を調べたい"
    expertise = 'intermediate'
    
    for data_name, data in [('高品質データ', high_quality_data), ('低品質データ', low_quality_data)]:
        print(f"\n📊 {data_name}")
        characteristics = advisor.analyze_data_characteristics(data)
        
        print(f"   データ品質スコア: {characteristics.data_quality_score:.3f}")
        print(f"   欠損データ率: {np.mean(list(characteristics.missing_data_pattern.values())):.1%}")
        
        suggestions = advisor.suggest_methods(characteristics, question, expertise)
        
        if suggestions:
            print(f"   推奨手法数: {len(suggestions)}")
            print(f"   最高信頼度: {suggestions[0].confidence_score:.3f}")
            print(f"   トップ推奨: {suggestions[0].method_name}")
        else:
            print("   推奨手法なし")

def demo_computation_time_estimation():
    """計算時間推定のデモ"""
    print("\n\n⏱️ 計算時間推定デモ")
    print("=" * 50)
    
    advisor = StatisticalMethodAdvisor()
    datasets = create_demo_datasets()
    
    # 異なるサイズのデータセットで計算時間を比較
    small_data = datasets['small_sample']
    large_data = datasets['large_normal']
    
    question = "データを分析したい"
    expertise = 'intermediate'
    
    for data_name, data in [('小データ', small_data), ('大データ', large_data)]:
        print(f"\n📊 {data_name} ({data.shape[0]}行)")
        characteristics = advisor.analyze_data_characteristics(data)
        suggestions = advisor.suggest_methods(characteristics, question, expertise)
        
        if suggestions:
            print("   推定計算時間:")
            for suggestion in suggestions[:3]:
                print(f"   {suggestion.method_name}: {suggestion.estimated_computation_time:.1f}秒")

def main():
    """メインデモ実行"""
    print("🤖 統計手法アドバイザー デモンストレーション")
    print("=" * 60)
    
    try:
        # 各デモを実行
        demo_data_characteristics_analysis()
        demo_method_suggestions()
        demo_expertise_level_adaptation()
        demo_data_quality_impact()
        demo_computation_time_estimation()
        
        print("\n\n✅ 全てのデモが完了しました！")
        print("\n📝 統計手法アドバイザーの主な機能:")
        print("   • データ特性の自動分析")
        print("   • 研究質問に基づく手法推奨")
        print("   • ユーザー専門レベルに応じた適応")
        print("   • データ品質を考慮した推奨")
        print("   • 計算時間の推定")
        print("   • 教育的コンテンツの提供")
        print("   • 代替手法の提案")
        
    except Exception as e:
        print(f"\n❌ デモ実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()