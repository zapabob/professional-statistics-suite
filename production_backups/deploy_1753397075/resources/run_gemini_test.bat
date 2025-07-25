@echo off
echo 🤖 AI Orchestrator - Gemini Integration Test
echo ==========================================

REM 環境変数の確認
if "%GOOGLE_API_KEY%"=="" (
    echo ⚠️ GOOGLE_API_KEY環境変数が設定されていません
    echo Google AI Studio (https://aistudio.google.com/) でAPIキーを取得してください
    echo.
    echo 設定方法:
    echo set GOOGLE_API_KEY=your_api_key_here
    echo または .env ファイルに記載してください
    echo.
    pause
    exit /b 1
)

echo ✅ Google API Key: 設定済み
echo.

REM Pythonバージョン確認
py -3 --version
if errorlevel 1 (
    echo ❌ Python 3が見つかりません
    echo Python 3をインストールしてください
    pause
    exit /b 1
)

echo.
echo 🚀 テスト実行中...
echo.

REM テスト実行
py -3 test_ai_orchestrator.py

echo.
echo 📊 テスト完了
pause