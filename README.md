# RAG 知识库问答系统

FastAPI + DeepSeek + Streamlit 构建的 RAG (检索增强生成) 问答系统。

## 功能

- 📄 支持上传 TXT / PDF / DOCX 文档
- 🔍 自动检索最相关的段落
- 🤖 基于 DeepSeek API 智能回答
- 💬 多轮对话记忆
- ⚡ 流式输出（打字机效果）

## 项目结构

```
rag-project/
├── backend/
│   ├── main.py          # FastAPI 后端服务
│   ├── rag_engine.py    # RAG 核心引擎（TF-IDF 检索 + LLM 问答）
│   └── requirements.txt
├── frontend/
│   └── app.py           # Streamlit 前端界面
├── data/                # 文档存储目录
└── README.md
```

## 运行方式

### 1. 激活环境

```bash
conda activate ai_app
```

### 2. 启动后端（终端 1）

```bash
cd rag-project
python backend/main.py
# 后端运行在 http://localhost:8000
# API 文档在 http://localhost:8000/docs
```

### 3. 启动前端（终端 2）

```bash
cd rag-project
streamlit run frontend/app.py --server.port 8501
# 界面在 http://localhost:8501
```

### 4. 使用

1. 打开浏览器访问 http://localhost:8501
2. 左侧上传文档（支持 TXT/PDF/DOCX）
3. 在聊天框提问

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/upload` | 上传文档 |
| POST | `/ask` | 问答 |
| GET | `/ask?q=问题` | 流式问答 |
| GET | `/files` | 文件列表 |
| DELETE | `/files/{name}` | 删除文件 |

## 核心原理

```
上传文档 → 分段 → TF-IDF 向量化 → 入库
              ↓
用户提问 → 检索相关段落 → 拼入 Prompt → 调用 DeepSeek → 回答
```

## 升级路线

- 替换 `TfidfVectorStore` → ChromaDB + BGE Embedding（语义搜索）
- 加 Rerank 重排序 → 提高检索精度
- 加 Graph RAG → 处理多文档关系
- 加 Agent → 让 AI 能主动查询、分析
