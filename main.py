from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx, os, json
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST  = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_KEY   = os.getenv("OLLAMA_API_KEY")

app = FastAPI(title="Gemma4 Chat Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    with httpx.Client(timeout=120) as client:
        r = client.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": req.message}],
                "stream": False,
            },
            headers={"Authorization": f"Bearer {OLLAMA_KEY}"},
        )
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)

        lines  = [l for l in r.text.strip().splitlines() if l.strip()]
        parsed = [json.loads(l) for l in lines]
        content = "".join(p.get("message", {}).get("content", "") for p in parsed)
        return {"response": content}

@app.get("/health")
def health():
    return {"status": "ok", "model": OLLAMA_MODEL}

@app.get("/")
def root():
    return {"service": "Gemma4 Chat Service", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
