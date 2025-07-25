'''
# Interactive AI Analysis App (Streamlit)
# 対話型AI分析アプリ

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from ai_integration import analyze_with_ai, get_available_models

st.set_page_config(page_title="対話型AI統計分析", layout="wide")

st.title("🤖 対話型AI統計分析")
st.caption("アップロードしたデータについて、自然言語でAIに分析を依頼できます。")

# --- サイドバー --- #
st.sidebar.header("設定")

uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.sidebar.success("ファイルを読み込みました！")
        st.sidebar.dataframe(df.head())
    except Exception as e:
        st.sidebar.error(f"ファイル読み込みエラー: {e}")

# 利用可能なモデルを取得
available_models = get_available_models()

provider = st.sidebar.selectbox(
    "LLMプロバイダー",
    options=list(available_models.keys()),
    index=0
)

model = st.sidebar.selectbox(
    "モデル",
    options=available_models.get(provider, []),
    index=0
)

enable_rag = st.sidebar.toggle("RAG (知識ベース) を有効化", value=True)
enable_correction = st.sidebar.toggle("自己修正を有効化", value=True)

# --- メイン画面 --- #

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "code" in message:
            st.code(message["code"], language="python")
        if "output" in message:
            st.text(message["output"])
        if "fig" in message:
            st.pyplot(message["fig"])

if prompt := st.chat_input("分析したい内容をどうぞ (例: A列とB列の相関を調べて)"):
    if 'df' not in st.session_state:
        st.error("まずCSVファイルをアップロードしてください。")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AIが分析中..."):
                result = analyze_with_ai(
                    query=prompt,
                    data=st.session_state.df,
                    provider=provider,
                    model=model,
                    enable_rag=enable_rag,
                    enable_correction=enable_correction
                )

            if result["success"]:
                st.success("分析が完了しました！")
                response_content = f"分析が完了しました。プロバイダー: `{result.get('provider', 'N/A')}`, モデル: `{result.get('model', 'N/A')}`"
                st.markdown(response_content)
                
                message = {"role": "assistant", "content": response_content}

                if result.get("extracted_code"):
                    st.code(result["extracted_code"], language="python")
                    message["code"] = result["extracted_code"]
                
                exec_res = result.get("execution_result", {})
                if exec_res.get("output"):
                    st.text(exec_res["output"])
                    message["output"] = exec_res["output"]
                
                # Streamlitではmatplotlibのfigureを直接扱う
                # グローバルなpyplotの状態からfigureを取得して表示
                figs = [plt.figure(n) for n in plt.get_fignums()]
                for i, fig in enumerate(figs):
                    st.pyplot(fig)
                    # セッションステートに保存するためにfigureをシリアライズできないので、
                    # ここでは単純に表示するだけにする。
                    # 必要であればBytesIOに保存するなどの工夫が必要。
                
'''
