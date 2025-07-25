#!/bin/bash
echo "🤖 AI Orchestrator - Gemini Integration Test"
echo "=========================================="

# 環境変数の確認
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "⚠️ GOOGLE_API_KEY環境変数が設定されていません"
    echo "Google AI Studio (https://aistudio.google.com/) でAPIキーを取得してください"
    echo ""
    echo "設定方法:"
    echo "export GOOGLE_API_KEY=your_api_key_here"
    echo "または .env ファイルに記載してください"
    echo ""
    exit 1
fi

echo "✅ Google API Key: 設定済み"
echo ""

# Pythonバージョン確認
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3が見つかりません"
    echo "Python 3をインストールしてください"
    exit 1
fi

echo ""
echo "🚀 テスト実行中..."
echo ""

# テスト実行
python3 test_ai_orchestrator.py

echo ""
echo "📊 テスト完了"