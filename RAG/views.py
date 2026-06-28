from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import re

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from .utils import RAG

# =====================================================
# SESSION RAGS
# =====================================================
SESSION_RAGS = {}


def normalize_session_id(name):
    if not name:
        return None
    safe = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return safe or "session"


def get_or_create_rag(session_id):
    if not session_id:
        return None
    if session_id not in SESSION_RAGS:
        SESSION_RAGS[session_id] = RAG(session_id=session_id)
    return SESSION_RAGS[session_id]


# =====================================================
# LOCAL LLM
# =====================================================
class LocalLLM:

    def __init__(self):

        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        print("Loading LLM...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )

    def generate(self, prompt):

        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.1,
            do_sample=False
        )

        # ONLY return new tokens (not prompt)
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()


llm = LocalLLM()


# =====================================================
# UPLOAD
# =====================================================
def upload_pdf(request):

    if request.method == "POST":

        if not request.FILES:
            return render(request, "RAG/upload.html", {"msg": "❌ No file"})

        file = next(iter(request.FILES.values()))
        session_id = request.POST.get("session_id")

        if not session_id:
            session_id = normalize_session_id(file.name)

        rag = get_or_create_rag(session_id)
        rag.ingest(file, filename=file.name)

        return render(
            request,
            "RAG/upload.html",
            {"msg": f"✅ {file.name} indexed"}
        )

    return render(request, "RAG/upload.html")


@csrf_exempt
def upload_document(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    if not request.FILES:
        return JsonResponse({"error": "No file provided."}, status=400)

    file = next(iter(request.FILES.values()))
    session_id = request.POST.get("session_id")

    if not session_id:
        session_id = normalize_session_id(file.name)

    rag = get_or_create_rag(session_id)
    if rag.index.ntotal == 0:
        rag.ingest(file, filename=file.name)

    return JsonResponse({
        "msg": f"✅ {file.name} indexed",
        "session_id": session_id,
        "file_name": file.name
    })


# =====================================================
# CHAT
# =====================================================
@csrf_exempt
def chat(request):

    if request.method == "GET":
        return render(request, "RAG/chat.html")

    query = request.POST.get("question", "").strip()
    session_id = request.POST.get("session_id", "").strip()

    if not query:
        return JsonResponse({"error": "empty query"})

    if not session_id:
        return JsonResponse({"error": "missing session_id"})

    rag = get_or_create_rag(session_id)
    if not rag or rag.index.ntotal == 0:
        return JsonResponse({"error": "document not indexed for this session"}, status=400)

    result = rag.query(query)
    context = result["context"][:2000]

    prompt = f"""
### SYSTEM
You are a strict QA system.

RULES:
- Use ONLY context
- Output ONLY final answer
- No explanations
- If not found: I don't know

### CONTEXT
{context}

### QUESTION
{query}

### ANSWER
"""

    try:
        answer = llm.generate(prompt)
        print(answer)
    except Exception as e:
        answer = str(e)

    return JsonResponse({
        "answer": answer,
        "sources": result["docs"]
    })


# =====================================================
# DEBUG
# =====================================================
@csrf_exempt
def rag_tool(request):

    if request.method != "POST":
        return JsonResponse({"error": "POST only"})

    body = json.loads(request.body)
    query = body.get("query", "")
    session_id = body.get("session_id", "").strip()

    if not session_id:
        return JsonResponse({"error": "missing session_id"}, status=400)

    rag = get_or_create_rag(session_id)
    if not rag or rag.index.ntotal == 0:
        return JsonResponse({"error": "document not indexed for this session"}, status=400)

    return JsonResponse(rag.query(query))