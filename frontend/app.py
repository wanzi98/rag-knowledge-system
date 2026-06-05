"""
Streamlit 前端 v2：展示语义检索 + 相似度分数 + 来源追踪
"""

import os
import json
import requests
import streamlit as st

# ============================================================
# 配置
# ============================================================

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="智能RAG知识库问答系统 v2",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# 辅助函数
# ============================================================

def upload_file(file):
    files = {"file": (file.name, file.getvalue())}
    resp = requests.post(f"{API_BASE}/upload", files=files)
    return resp.json()


def ask_question(question: str, history: list):
    resp = requests.post(f"{API_BASE}/ask", json={
        "question": question,
        "history": history
    })
    return resp.json()


def get_files():
    resp = requests.get(f"{API_BASE}/files")
    return resp.json().get("files", [])


def delete_file(filename: str):
    requests.delete(f"{API_BASE}/files/{filename}")


def get_stats():
    resp = requests.get(f"{API_BASE}/stats")
    return resp.json()


# ============================================================
# 页面布局
# ============================================================

# 侧边栏
with st.sidebar:
    st.title("📚 知识库")

    # 引擎状态
    try:
        stats = get_stats()
        st.caption(f"🔧 {stats.get('embedding_type', '?')}")
        st.caption(f"📦 {stats.get('total_chunks', 0)} 个文本块")
        st.caption(f"📄 {len(stats.get('files', []))} 个文档")
    except:
        st.caption("⚠️ 后端未连接")

    st.markdown("---")

    # 文件上传
    uploaded_file = st.file_uploader(
        "上传文档",
        type=["txt", "pdf", "docx"],
        help="支持 TXT、PDF、DOCX 格式"
    )

    if uploaded_file is not None:
        with st.spinner("正在处理文档..."):
            try:
                result = upload_file(uploaded_file)
                st.success(f"✅ {result['message']} （{result['chunks']} 个文本块）")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 上传失败: {e}")

    st.markdown("---")
    st.subheader("已上传的文档")

    files = get_files()
    if not files:
        st.info("还没有上传任何文档")
    else:
        for fname in files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"📄 {fname}")
            with col2:
                if st.button("删除", key=f"del_{fname}"):
                    delete_file(fname)
                    st.rerun()


# 主区域
st.title("💬 智能知识库问答")
st.caption("基于语义检索（ChromaDB + n-gram Embedding）+ DeepSeek API")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 如果有来源信息，展示
        if "sources" in msg and msg["sources"]:
            st.caption(f"📎 来源: {', '.join(msg['sources'])}")

# 提问输入框
if prompt := st.chat_input("输入你的问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    # 获取回答
    with st.chat_message("assistant"):
        with st.spinner("🔍 检索中..."):
            result = ask_question(prompt, history)

        answer = result.get("answer", "出错了")
        sources = result.get("sources", [])
        scores = result.get("scores", [])

        st.markdown(answer)

        # 展示来源和相似度
        if sources:
            source_info = []
            for s, score in zip(sources, scores):
                source_info.append(f"{s} (相似度:{score:.2f})")
            st.caption(f"📎 来源: {', '.join(source_info)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "scores": scores
    })


# 底部说明
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🧠 语义检索: 同义词/近义词也能匹配")
with col2:
    st.caption("💾 向量库: ChromaDB")
with col3:
    st.caption("🔗 来源追踪: 显示答案出处")
