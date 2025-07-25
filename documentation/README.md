# 🔬 Professional Statistics Suite - AI Edition

**The Future of Data Analysis is Conversational. Just talk to your data.**  
**未来のデータ分析は「対話」から。AIに話しかけるだけで、高度な統計分析を。**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Mac%20%7C%20Linux-lightgrey.svg)]()
[![Local LLMs](https://img.shields.io/badge/Local%20LLMs-Ollama%20%7C%20LM%20Studio%20%7C%20Kobold.cpp-blueviolet.svg)]()
[![Cloud LLMs](https://img.shields.io/badge/Cloud%20LLMs-OpenAI%20%7C%20Anthropic%20%7C%20Google-brightgreen.svg)]()

---

## 🚀 Overview | 概要

**Professional Statistics Suite (PSS)** は、従来の統計ソフトウェアのパラダイムを覆す、**対話型AI分析プラットフォーム**です。メニューをクリックしたり、複雑なスクリプトを書いたりする必要はもうありません。分析したいことを自然言語でAIに伝えるだけで、データの読み込みから高度な統計分析、可視化まで、すべてが自動で実行されます。

SPSSが提供するような伝統的な分析機能のすべてを、より直感的かつ強力な形で提供します。

- **English**: PSS is a conversational AI analysis platform that revolutionizes the traditional statistics software paradigm. No more clicking through menus or writing complex scripts. Simply tell the AI what you want to analyze in natural language, and it will handle everything from data loading to advanced statistical analysis and visualization.
- **日本語**: PSSは、従来の統計ソフトウェアの常識を覆す、対話型のAI分析プラットフォームです。メニューをクリックしたり、複雑なスクリプトを書いたりする必要はもうありません。分析したいことを自然言語でAIに伝えるだけで、データの読み込みから高度な統計分析、可視化まで、すべてが自動で実行されます。

---

## ✨ Key Features | AIが実現する主要機能

### 💬 **1. Conversational Analysis | 対話型分析**
チャット形式のUIでAIに話しかけるだけで分析が進みます。「A列とB列の相関を調べて」「このデータをワークロードタイプ別に可視化して」のように、あなたの言葉がそのまま分析に変わります。

### 🧠 **2. Self-Correcting Code Engine | 自己修正コードエンジン**
AIが生成した分析コードにエラーがあっても、心配は無用です。システムがエラーを自動で検知し、AI自身が問題点を分析してコードを修正します。ユーザーは最終的な正しい結果だけを受け取ることができます。

### 📚 **3. Knowledge-Augmented Analysis (RAG) | 知識拡張（RAG）分析**
プロジェクト内の過去の分析ログ（`_docs`フォルダ）をAIが自動で学習します。「前のRTX3080の分析と同じようにやって」といった曖昧な指示でも、AIが文脈を理解して適切な分析を実行します。

### 🔌 **4. Bring Your Own LLM | 選べるLLMバックエンド**
クラウドからローカルまで、お好みのLLMを自由に選択できます。
- **Cloud APIs**: OpenAI (GPT-4o), Anthropic (Claude 3.5), Google (Gemini 1.5)
- **Local LLMs**: **Ollama**, **LM Studio**, **Kobold.cpp** に完全対応。APIキー不要で、オフラインかつセキュアな環境でGGUFモデルなどを利用できます。

### 🛠️ **5. Comprehensive Statistical Powerhouse | 堅牢な統計分析能力**
バックエンドは`pandas`, `scikit-learn`, `statsmodels`などの強力なライブラリで構築されており、記述統計から多変量解析、機械学習まで、SPSSが提供する分析機能を網羅しています。

### 🖥️ **6. GUIアプリケーション |  Integrated GUI Application**
**NEW!** 既存のGUIアプリを統合した新しいデスクトップアプリケーションが利用可能です。タブ形式のインターフェースで、AI分析、統計手法アドバイザー、仮説検証、可視化、レポート生成を統合的に利用できます。

---

## 🚀 Quick Start | クイックスタート

わずか3ステップで、次世代の対話型分析を始められます。

### 1. Clone Repository | リポジトリのクローン
```bash
git clone https://github.com/zapabob/professional-statistics-suite.git
cd professional-statistics-suite/professional-statistics-suite
```

### 2. Install Dependencies | 依存関係のインストール
```bash
# 仮想環境の作成を推奨
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 3. Launch the App | アプリケーションの起動

#### 🖥️ **Kiro統合GUIアプリ（推奨）**
```bash
# Windows
py -3 run_kiro_gui.py

# macOS/Linux
python3 run_kiro_gui.py
```

#### 🌐 **Streamlit Webアプリ**
```bash
streamlit run interactive_analysis_app.py
```

---

## 📖 How to Use | 基本的な使い方

### 🖥️ 統合GUIアプリの使用方法

1. **アプリケーション起動**: `run_kiro_gui.py`を実行
2. **データ読み込み**: 「データ管理」タブでCSVファイルを読み込み
3. **AI分析**: 「AI分析」タブで自然言語クエリを入力
4. **統計手法推奨**: 「統計手法アドバイザー」タブでデータ特性分析
5. **仮説検証**: 「仮説検証」タブで統計的仮説を検証
6. **可視化**: 「可視化」タブでグラフを作成
7. **レポート生成**: 「レポート生成」タブで分析レポートを作成

### 🌐 Streamlit Webアプリの使用方法

1.  **データをアップロード:** サイドバーから分析したいCSVファイルをアップロードします。
2.  **LLMを選択:** サイドバーで使用したいLLMプロバイダー（例: `ollama`, `lmstudio`, `koboldcpp`）とモデル名を選択します。
    - *ローカルLLMを使用する場合は、事前に各ツールのサーバーを起動しておいてください。*
3.  **AIに話しかける:** 画面下のチャットボックスに、分析したい内容を自然言語で入力します。

これだけで、AIとの対話を通じたデータ分析が始まります。

---

## 📁 Project Architecture | プロジェクト構成

```
professional-statistics-suite/
├── 🚀 run_kiro_gui.py              # Kiro統合GUIアプリ起動スクリプト
├── 🖥️ kiro_integrated_gui.py       # Kiro統合GUIアプリケーション (Tkinter)
├── 🌐 interactive_analysis_app.py   # Webアプリケーション (Streamlit)
├── 🤖 ai_integration.py            # AI連携と分析実行のコアエンジン
├── 📊 statistical_method_advisor.py # 統計手法アドバイザー
├── 🔍 assumption_validator.py      # 仮説検証エンジン
├── 📋 professional_reports.py       # レポート生成エンジン
│
├── 📄 requirements.txt             # 依存ライブラリ
├── 📂 _docs/                        # AIが学習するナレッジベース (Markdown)
└── 📂 (その他、テストや設定ファイル)
```

---

## 🎯 Kiro統合GUIアプリの主要機能

### 📊 データ管理タブ
- CSVファイルの読み込み・保存
- データ表示と情報表示
- データクリア機能

### 🤖 AI分析タブ
- 自然言語クエリ入力
- AIOrchestratorによる分析実行
- 非同期処理によるUI応答性確保

### 📈 統計手法アドバイザータブ
- データ特性分析
- 統計手法推奨
- StatisticalMethodAdvisorとの統合

### 🔍 仮説検証タブ
- 正規性検定
- 等分散性検定
- 独立性検定
- AssumptionValidatorとの統合

### 📊 可視化タブ
- ヒストグラム、散布図、箱ひげ図、相関行列
- matplotlib/seabornによる可視化
- 動的な可視化生成

### 📋 レポート生成タブ
- 包括的レポート生成
- AI分析レポート
- 統計手法レポート
- ReportGeneratorとの統合

### 📝 ログ表示タブ
- 実装ログの表示
- アプリケーションログの記録
- ログ更新機能

---

## ⚖️ License | ライセンス

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルをご覧ください。