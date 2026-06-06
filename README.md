<img src="https://img.shields.io/badge/Python-3.7%2B-blue" /> <img src="https://img.shields.io/badge/FastAPI-0.110%2B-green" /> <img src="https://img.shields.io/badge/ChromaDB-0.5%2B-orange" /> <img src="https://img.shields.io/badge/Streamlit-1.35%2B-red" /> <img src="https://img.shields.io/badge/DeepSeek-API-brightgreen" />

# 📚 智能 RAG 知识库问答系统 v2

> **上传文档 → 自动分块 → 语义检索 → AI 基于文档内容回答**
>
> 一个完整可运行的 RAG（检索增强生成）系统，从底层 Embedding 到前端展示全部手写实现。
>
> 📖 **[免费部署指南](DEPLOY_GUIDE.md)** — 把项目部署到 Render + Streamlit Cloud

---

## ✨ 功能一览

| 功能 | 说明 |
|------|------|
| 📄 **多格式上传** | 支持 TXT / PDF / DOCX 文档上传 |
| 🔍 **语义检索** | 基于 ChromaDB 向量库，理解同义词/近义词，而非字面匹配 |
| 📊 **来源追踪** | 每条回答附带文档来源和相似度分数，让 AI 输出可验证 |
| 🧠 **本地 Embedding** | 自研字符 n-gram 哈希嵌入引擎，不依赖外部模型 |
| 💬 **多轮对话** | 带对话记忆，上下文连贯 |
| ⚡ **流式输出** | 打字机效果，实时展示回答过程 |
| 🎛️ **系统监控** | 内置 `/stats` 接口，查看 Embedding 类型、Chunk 数量等 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         用户界面 (Streamlit)                         │
│             上传文档 / 提问 / 查看来源和相似度分数                     │
└─────────────────────────┬───────────────────────────────────────────┘
                          │ HTTP / REST
┌─────────────────────────▼───────────────────────────────────────────┐
│                      后端 API (FastAPI)                              │
│        POST /upload → 接收文档 │ POST /ask → 回答问题               │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────┐
│                        RAG 引擎 (rag_engine.py)                      │
│                                                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │ DocumentProcessor│ → │  ChromaDB       │ → │  LLM (DeepSeek) │  │
│  │  智能分块+重叠   │    │  向量存储+ANN搜索│   │  基于资料回答    │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│                                ↑                                     │
│                         ┌─────────────────┐                          │
│                         │  LocalEmbedding │                          │
│                         │  n-gram哈希嵌入  │                          │
│                         │  (纯numpy实现)  │                          │
│                         └─────────────────┘                          │
└──────────────────────────────────────────────────────────────────────┘
```

### RAG 全流程

```
Step 1: 文档加载 → 智能分块(300字符+50字符重叠) → 防止切碎语义
Step 2: 每个文本块 → n-gram哈希嵌入(256维向量) → 存入ChromaDB
Step 3: 用户提问 → 同样方法嵌入 → ChromaDB ANN近似最近邻搜索
Step 4: 检索到的 top-3 文本块 + 问题 → 拼接 Prompt → 调用 DeepSeek
Step 5: 返回回答 + 来源文件 + 相似度分数
```

---

## 🚀 快速开始

### 环境要求

- Python 3.7+
- DeepSeek API Key（[点此申请](https://platform.deepseek.com/)）

### 安装

```bash
# 1. 克隆（或进入项目目录）
cd rag-project

# 2. 安装依赖
pip install fastapi uvicorn chromadb numpy openai streamlit requests PyPDF2 python-docx

# 3. 设置 API Key
# Windows (PowerShell):
$env:DEEPSEEK_API_KEY="sk-your-key-here"
# 或直接修改 backend/main.py 中的配置
```

### 运行

**终端 1 — 启动后端：**

```bash
cd rag-project
python backend/main.py
# 后端运行在 http://localhost:8000
# Swagger API 文档在 http://localhost:8000/docs
```

**终端 2 — 启动前端：**

```bash
cd rag-project
streamlit run frontend/app.py --server.port 8501
# 界面在 http://localhost:8501
```

打开浏览器访问 `http://localhost:8501`，上传文档开始提问 🎉

---

## 📸 页面预览（运行后截图贴这里）

```
+--------------------------------------------------+
|  📚 知识库           |  💬 智能知识库问答           |
|  ─────────           |  ────────────────           |
|  🔧 n-gram hash     |                              |
|  📦 42 个文本块      |  [用户] 什么是RAG？         |
|  📄 2 个文档         |                              |
|  ─────────           |  [AI] RAG是检索增强生       |
|  [上传文档]          |  成的缩写（Retrieval-       |
|  ─────────           |  Augmented Generation）...  |
|  已上传的文档:       |                              |
|  📄 AI.txt    [删除] |  📎 来源: AI.txt (相似度    |
|  📄 Python.txt [删除]|  :0.87)                     |
+--------------------------------------------------+
```

---

## 🧠 核心技术详解（面试向）

### 1. 自研本地 Embedding 引擎

因网络限制无法下载预训练模型，自行实现了基于字符 n-gram 的哈希嵌入：

```
文本 "AI" → 字符级n-gram → ["A", "I", "AI"]
         → MD5哈希 → 映射到256维向量 → L2归一化
```

**技术权衡：**
- ✅ 不依赖外部模型、纯 numpy、速度快、适合中文
- ❌ 质量不如 BERT/sentence-transformers
- 🔧 设计了可插拔接口，替换只需实现 `embed_one(text) → vector`

### 2. 智能分块策略（v1 → v2 关键升级）

```
v1: 简单按空行切分 → 长段落溢出token限制 / 短段落语义不完整
v2: 段落→句子二级切分 + 前后块50字符重叠 → 防切碎语义
```

### 3. ChromaDB 向量检索

传统 `LIKE '%关键词%'` 只能精确匹配，ChromaDB 做 ANN（近似最近邻）搜索，支持同义词匹配。内部使用 HNSW 图索引，比暴力搜索快几十倍。

---

## 📋 API 文档

| 方法 | 路径 | 说明 | 请求体/参数 |
|------|------|------|-------------|
| GET | `/` | 系统信息 | - |
| POST | `/upload` | 上传文档并导入 | `file` (multipart) |
| POST | `/ask` | 非流式问答 | `{question, history?}` |
| GET | `/ask` | 流式问答 | `q` (query), `history` (JSON) |
| GET | `/files` | 文件列表 | - |
| DELETE | `/files/{name}` | 删除文件 | - |
| GET | `/stats` | 引擎状态 | - |

---

## 📈 v1 → v2 升级对比

| 维度 | v1（TF-IDF） | v2（ChromaDB） |
|------|-------------|----------------|
| 检索原理 | 词频统计，字面匹配 | 向量语义，近似匹配 |
| 同义词识别 | ❌ | ✅ |
| 分块策略 | 简单按空行切 | 段落→句子+重叠 |
| 结果展示 | 仅文字回答 | 回答+来源+相似度 |
| 配置管理 | 硬编码 | Config 类集中管理 |

---

## 🔜 升级路线

- [ ] 替换 Embedding → sentence-transformers / BGE（精度大幅提升）
- [ ] 添加 Reranker → 检索后重排序，提高 top 结果质量
- [ ] 部署到云端 → Render + Streamlit Cloud 免费部署
- [ ] 添加 Graph RAG → 处理多文档间关系
- [ ] 支持更多格式 → Markdown、HTML、图片OCR

---

## 📝 License

MIT
