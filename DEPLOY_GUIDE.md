# 🚀 免费部署指南

> 把你的 RAG 知识库项目部署到云端，面试官可以直接点开链接试用。
>
> **总费用：$0**（全部使用免费额度）

---

## 架构

```
用户浏览器                     云端                          DeepSeek
    │                           │                              │
    ▼                           ▼                              ▼
┌─────────────┐    API请求    ┌──────────────┐    API调用    ┌────────┐
│  Streamlit  │ ───────────→  │  FastAPI     │ ───────────→  │LLM API │
│  Cloud 前端 │ ←───────────  │  Render 后端 │ ←───────────  │        │
│  (免费)     │   JSON响应    │  (免费)      │              └────────┘
└─────────────┘               └──────────────┘
                                   │
                                   ▼
                              ┌──────────┐
                              │ ChromaDB │
                              │ (持久化)  │
                              └──────────┘
```

---

## 第一步：推送代码到 GitHub

如果你还没推送，先创建 GitHub 仓库：

```bash
# 在 rag-project 目录下（已完成 git init）
git add .
git commit -m "v2 RAG knowledge system with ChromaDB"

# 在 GitHub 新建仓库（不要勾选 README/license）
git remote add origin https://github.com/wanzi98/rag-knowledge-system.git
git push -u origin main
```

⚠️ **重要：不要把 API Key 提交到 GitHub！**

确保项目中的 `DEEPSEEK_API_KEY` 读取的是环境变量，而不是硬编码：
```python
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-your-key-here")
```
这样在 GitHub 上是占位符，真正的 Key 在部署平台的环境变量中设置。

---

## 第二步：部署后端（FastAPI → Render）

[Render](https://render.com) 提供免费的后端托管，支持自动从 GitHub 部署。

### 注册 & 连接 GitHub

1. 打开 [render.com](https://render.com) → 用 GitHub 账号注册
2. 点 **New +** → **Web Service**
3. 选择你的仓库 `rag-knowledge-system`
4. 填写配置：

| 字段 | 值 |
|------|-----|
| **Name** | `rag-knowledge-system` |
| **Environment** | `Python 3` |
| **Build Command** | `pip install -r backend/requirements.txt` |
| **Start Command** | `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Plan** | **Free** |

5. 点 **Advanced** → **Add Environment Variable**：

| Key | Value |
|-----|-------|
| `DEEPSEEK_API_KEY` | 你的真实 DeepSeek API Key |

6. 点 **Create Web Service**

### 等几分钟

部署成功后你会得到一个 URL，像这样：
```
https://rag-knowledge-system.onrender.com
```

访问这个 URL，应该看到：
```json
{"name":"智能RAG知识库问答系统","version":"2.0","status":"running"}
```

### 额外：设置 requirements.txt

确保 `backend/requirements.txt` 包含：
```
fastapi==0.104.1
uvicorn==0.24.0
openai>=1.30
chromadb>=0.5.0
numpy>=1.21.0
pypdf2==3.0.1
python-docx==0.8.11
python-multipart==0.0.6
```

---

## 第三步：部署前端（Streamlit → Streamlit Cloud）

[Streamlit Cloud](https://streamlit.io/cloud) 提供免费的前端托管。

### 部署

1. 打开 [share.streamlit.io](https://share.streamlit.io) → 用 GitHub 登录
2. 点 **New app**
3. 选择仓库 `rag-knowledge-system`
4. 填写配置：

| 字段 | 值 |
|------|-----|
| **Branch** | `main` |
| **Main file path** | `frontend/app.py` |
| **Python version** | `3.11` |

5. **Advanced settings...** → **Secrets** 中填入：

```toml
API_BASE = "https://rag-knowledge-system.onrender.com"
```

> 注意：这里的 `API_BASE` 要改成你自己的 Render 后端 URL。

6. 点 **Deploy**

### 等几分钟

部署成功后你会得到一个 URL：
```
https://rag-knowledge-system.streamlit.app/
```

这就是你可以放进简历的 Demo 链接！

---

## 第四步：验证

1. 打开 Streamlit 前端 URL
2. 上传一个文档（比如 `data/knowledge.txt`）
3. 提问一个和文档相关的问题
4. 确认能得到回答，且显示来源和相似度分数

---

## 常见问题

### Q: Render 免费版有什么限制？
- 15 分钟不访问会进入休眠（冷启动约 30-60 秒）
- 每月 750 小时免费时长（够了）
- 512 MB 内存（够用）

### Q: 数据会丢吗？
- Render 免费版是临时文件系统，重启后数据会丢失
- **每次重启后需要重新上传文档**
- 如果需要持久化，可以加 SQLite 或 PostgreSQL（后续再说）

### Q: Streamlit Cloud 有什么限制？
- 每个账号 1 个免费 app
- 公开可见（没关系，正好展示）
- 每月 1GB 网络带宽

### Q: 如何更新代码？
- 推送到 GitHub 的 main 分支
- Render 和 Streamlit Cloud 都会自动重新部署
- 等待 2-3 分钟生效

---

## 效果（放进简历）

```
📎 在线 Demo：
- 前端：https://rag-knowledge-system.streamlit.app
- 后端 API：https://rag-knowledge-system.onrender.com
- 源码：https://github.com/wanzi98/rag-knowledge-system
```

---

## 如果你还想部署 AI Agent 项目

AI Agent 只需要前端（Streamlit），部署更简单：

1. 推送 `ai-agent` 项目到独立的 GitHub 仓库
2. 在 Streamlit Cloud 部署 `app.py`
3. 在 Secrets 中设置 `DEEPSEEK_API_KEY`

> AI Agent 项目不需要后端，Streamlit 可以直接调 DeepSeek API。
