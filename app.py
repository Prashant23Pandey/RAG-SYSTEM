import os
import sys
import uuid
import json
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))

from workflow import compile_workflow
from vector_store import get_vectorstore, CHROMA_DIR
from ingest import ingest_file


# ---------------------------------------------------------------------------
# Lifespan: compile workflow once at startup
# ---------------------------------------------------------------------------
app_workflow = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_workflow
    app_workflow = compile_workflow()
    yield


app = FastAPI(
    title="Agentic RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User question")
    session_id: Optional[str] = Field(
        default=None, description="Optional session identifier"
    )


class StepEvent(BaseModel):
    node: str
    detail: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    steps: list[StepEvent]
    web_search_used: bool


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /api/chat – streams intermediate steps then final answer
# ---------------------------------------------------------------------------
@app.post("/api/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or uuid.uuid4().hex

    async def event_generator() -> AsyncGenerator[str, None]:
        steps: list[StepEvent] = []
        final_answer = ""
        web_search_used = False

        try:
            initial_state = {
                "question": req.prompt,
                "documents": [],
                "generation": "",
                "web_search_needed": False,
                "loop_count": 0,
                "guard_flag": "",
            }

            for step_event in app_workflow.stream(
                initial_state, stream_mode="updates"
            ):
                for node_name, update in step_event.items():
                    if node_name == "__end__":
                        continue

                    detail = _summarise_node(node_name, update)
                    evt = StepEvent(node=node_name, detail=detail)
                    steps.append(evt)
                    yield f"data: {evt.model_dump_json()}\n\n"

                    if node_name == "web_search":
                        web_search_used = True
                    if node_name == "generate":
                        final_answer = update.get("generation", "")

            if not final_answer:
                final_answer = (
                    "I could not generate an answer. Please try rephrasing."
                )

            payload = ChatResponse(
                session_id=session_id,
                answer=final_answer,
                steps=steps,
                web_search_used=web_search_used,
            )
            yield f"data: {payload.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as exc:
            error_payload = {"error": str(exc)}
            yield f"data: {json.dumps(error_payload)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# POST /api/upload – add documents to ChromaDB (Phase 7: multi-format ETL)
# ---------------------------------------------------------------------------
@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file_path: str, tenant_id: str = "default"):
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=400, detail="File not found")

    ext = os.path.splitext(file_path)[1].lower()
    supported = {".txt", ".pdf", ".md", ".html", ".htm", ".docx"}
    if ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(supported)}",
        )

    try:
        result = ingest_file(file_path, persist_dir=CHROMA_DIR, tenant_id=tenant_id)
        return UploadResponse(
            filename=result["filename"], chunks_added=result["chunks_added"]
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _summarise_node(node_name: str, update: dict) -> str:
    if node_name == "input_guard":
        flag = update.get("guard_flag", "")
        if flag:
            return f"BLOCKED: {flag}"
        return "Input cleared – PII masked if present"
    if node_name == "retrieve":
        docs = update.get("documents", [])
        return f"Hybrid search returned {len(docs)} reranked chunk(s)"
    if node_name == "grade_documents":
        needed = update.get("web_search_needed", False)
        if needed:
            return "No relevant documents found – web search flagged"
        docs = update.get("documents", [])
        return f"{len(docs)} relevant document(s) graded as useful"
    if node_name == "transform_query":
        q = update.get("question", "")
        return f"Query rewritten: {q[:120]}"
    if node_name == "web_search":
        docs = update.get("documents", [])
        return f"Web search returned {len(docs)} result(s)"
    if node_name == "generate":
        gen = update.get("generation", "")
        return f"Generation complete ({len(gen)} chars)"
    return str(update)[:200]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
