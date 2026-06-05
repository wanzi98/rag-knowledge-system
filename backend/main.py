"""
FastAPI 后端 v2：适配 ChromaDB + Embedding 新引擎

v2 新增：
  - GET /stats       引擎状态（查看 embedding 类型、chunk 数量等）
  - 回答附带来源和相似度分数
  - 更完善的错误处理
"""

import os
import sys
import json
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(__file__))
from rag_engine import RAGEngine, Config

# ============================================================
# 初始化
# ============================================================

app = FastAPI(title="智能RAG知识库问答系统 v2")
engine = RAGEngine()

DATA_DIR = Config.DATA_DIR
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# 数据模型
# ============================================================

class AskRequest(BaseModel):
    question: str
    history: Optional[list] = []


# ============================================================
# API 路由
# ============================================================

@app.get("/")
async def root():
    """首页"""
    return {
        "name": "智能RAG知识库问答系统",
        "version": "2.0",
        "status": "running",
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文档并自动导入向量库"""
    filepath = os.path.join(DATA_DIR, file.filename)
    try:
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)

        chunk_count = engine.ingest(filepath)
        stats = engine.get_stats()

        return {
            "message": f"文件 '{file.filename}' 上传并导入成功",
            "chunks": chunk_count,
            "filename": file.filename,
            "total_chunks": stats["total_chunks"],
        }
    except Exception as e:
        # 清理失败的文件
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ask")
async def ask(req: AskRequest):
    """非流式问答（返回结果+来源）"""
    result = engine.query(req.question, req.history)
    return {
        "answer": result["answer"],
        "question": req.question,
        "sources": result["sources"],
        "scores": result["scores"],
    }


@app.get("/ask")
async def ask_stream(q: str, history: Optional[str] = None):
    """流式问答"""
    hist = json.loads(history) if history else []

    return StreamingResponse(
        engine.query_stream(q, hist),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/files")
async def list_files():
    """列出所有已入库的文档"""
    files = engine.vector_store.list_files()
    return {"files": files}


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """删除文档"""
    engine.vector_store.delete_file(filename)

    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    return {"message": f"文件 '{filename}' 已删除"}


@app.get("/stats")
async def get_stats():
    """查看引擎状态（面试展示点：能说出当前用了什么 embedding、多少 chunk）"""
    return engine.get_stats()


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    print(f"RAG 引擎 v2 启动中...")
    print(f"   Embedding: {Config.EMBED_DIM}维 n-gram 哈希")
    print(f"   分块大小: {Config.CHUNK_SIZE}字符 (重叠{Config.CHUNK_OVERLAP})")
    print(f"   向量数据库: ChromaDB")
    print(f"   LLM: DeepSeek Chat")
    uvicorn.run(app, host="0.0.0.0", port=8000)
