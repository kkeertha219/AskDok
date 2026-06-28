import io
import os
import re
import pickle
import logging
import requests
import numpy as np
import pandas as pd
import faiss

from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from bs4 import BeautifulSoup

from sentence_transformers import SentenceTransformer
from pathlib import Path

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("RAG")


# =====================================================
# EXTERNAL BINARIES (UPDATE THESE PATHS)
# =====================================================
BASE_DIR = str(Path(__file__).resolve().parent.parent.parent)

POPPLER_PATH = BASE_DIR + r"\external_tools\poppler\Library\bin"

TESSERACT_PATH = BASE_DIR + r"\external_tools\tessract\tesseract.exe"
print(TESSERACT_PATH)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# =====================================================
# CONFIG
# =====================================================
class RAGConfig:
    def __init__(self):
        self.embed_model = "BAAI/bge-base-en-v1.5"

        self.chunk_size = 300
        self.chunk_overlap = 50

        self.use_ocr = True

        # FAISS persistence
        self.faiss_index_file = "rag_index.faiss"
        self.metadata_file = "rag_metadata.pkl"


# =====================================================
# RAG ENGINE
# =====================================================
class RAG:
    def __init__(
        self,
        cfg=RAGConfig(),
        session_id=None,
        index_file=None,
        metadata_file=None
    ):
        self.cfg = cfg
        self.session_id = session_id

        log.info("Loading embedding model...")
        self.embedder = SentenceTransformer(cfg.embed_model)

        self.dim = self.embedder.get_embedding_dimension()

        if session_id and not index_file:
            safe_session_id = self.sanitize_session_id(session_id)
            index_file = f"rag_index_{safe_session_id}.faiss"
            metadata_file = f"rag_metadata_{safe_session_id}.pkl"

        self.index_file = self.resolve_path(
            index_file or self.cfg.faiss_index_file
        )
        self.metadata_file = self.resolve_path(
            metadata_file or self.cfg.metadata_file
        )

        log.info("Initializing FAISS...")

        if os.path.exists(self.index_file):

            self.index = faiss.read_index(self.index_file)

            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, "rb") as f:
                    self.metadata = pickle.load(f)
            else:
                self.metadata = []

            log.info(
                f"Loaded existing FAISS index at {self.index_file} "
                f"with {self.index.ntotal} vectors"
            )

        else:
            # Cosine similarity via normalized vectors
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadata = []

        self.next_id = len(self.metadata)

    def resolve_path(self, path):
        if not path:
            return None
        return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)

    def sanitize_session_id(self, session_id):
        return re.sub(r"[^a-z0-9_-]+", "_", session_id.lower()).strip("_")

    # =====================================================
    # SAVE INDEX
    # =====================================================
    def save_index(self):
        faiss.write_index(self.index, self.index_file)

        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)

    # =====================================================
    # CLEAN TEXT
    # =====================================================
    def clean(self, text):
        return re.sub(r"\s+", " ", text).strip()

    # =====================================================
    # CHUNKING
    # =====================================================
    def chunk(self, text):
        words = text.split()

        step = self.cfg.chunk_size - self.cfg.chunk_overlap

        return [
            " ".join(words[i:i + self.cfg.chunk_size])
            for i in range(0, len(words), step)
        ]

    # =====================================================
    # TYPE DETECTION
    # =====================================================
    def detect_type(self, filename):
        filename = filename.lower()

        if filename.endswith(".pdf"):
            return "pdf"

        if filename.endswith(".txt"):
            return "txt"

        if filename.endswith(".html"):
            return "html"

        if filename.endswith(".csv"):
            return "csv"

        return "txt"

    # =====================================================
    # OCR PDF
    # =====================================================
    def extract_pdf_ocr(self, file_bytes):

        images = convert_from_bytes(
            file_bytes,
            poppler_path=POPPLER_PATH
        )

        text = []

        for img in images:
            page_text = pytesseract.image_to_string(img)
            text.append(page_text)

        return self.clean(" ".join(text))

    # =====================================================
    # EXTRACT PDF
    # =====================================================
    def extract_pdf(self, file_bytes):

        reader = PdfReader(io.BytesIO(file_bytes))

        text = " ".join(
            page.extract_text() or ""
            for page in reader.pages
        )

        return self.clean(text)

    # =====================================================
    # EXTRACT TXT
    # =====================================================
    def extract_txt(self, file_bytes):
        return self.clean(
            file_bytes.decode(
                "utf-8",
                errors="ignore"
            )
        )

    # =====================================================
    # EXTRACT HTML
    # =====================================================
    def extract_html(self, file_bytes):

        soup = BeautifulSoup(
            file_bytes,
            "html.parser"
        )

        for tag in soup(["script", "style"]):
            tag.decompose()

        return self.clean(
            soup.get_text(" ")
        )

    # =====================================================
    # EXTRACT CSV
    # =====================================================
    def extract_csv(self, file_bytes):

        df = pd.read_csv(
            io.BytesIO(file_bytes)
        )

        rows = []

        for _, row in df.iterrows():

            rows.append(
                " | ".join(
                    f"{c}: {row[c]}"
                    for c in df.columns
                )
            )

        return self.clean(
            " ".join(rows)
        )

    # =====================================================
    # INGEST
    # =====================================================
    def ingest(self, file, filename=None):

        ftype = self.detect_type(
            filename or ""
        )

        log.info(
            f"Ingesting {filename} as {ftype}"
        )

        data = file.read()

        file.seek(0)

        if ftype == "pdf":

            text = self.extract_pdf(data)

            # OCR fallback for scanned PDFs
            if (
                self.cfg.use_ocr
                and len(text.split()) < 20
            ):
                log.info(
                    "PDF appears scanned. Using OCR..."
                )

                text = self.extract_pdf_ocr(data)

        elif ftype == "html":
            text = self.extract_html(data)

        elif ftype == "csv":
            text = self.extract_csv(data)

        else:
            text = self.extract_txt(data)

        if not text.strip():
            log.warning(
                "No text extracted from document"
            )
            return

        chunks = self.chunk(text)

        if not chunks:
            log.warning(
                "No chunks generated"
            )
            return

        log.info(
            f"Generating embeddings "
            f"for {len(chunks)} chunks..."
        )

        embeddings = self.embedder.encode(
            chunks,
            show_progress_bar=False
        )

        vectors = []

        for chunk, emb in zip(
            chunks,
            embeddings
        ):

            if len(chunk.strip()) < 20:
                continue

            emb = np.asarray(
                emb,
                dtype=np.float32
            )

            norm = np.linalg.norm(emb)

            if norm > 0:
                emb = emb / norm

            vectors.append(emb)

            self.metadata.append({
                "id": self.next_id,
                "text": chunk
            })

            self.next_id += 1

        if vectors:

            vectors_np = np.vstack(
                vectors
            ).astype(np.float32)

            self.index.add(vectors_np)

            self.save_index()

        log.info(
            f"Indexed {len(vectors)} chunks"
        )

    # =====================================================
    # RETRIEVE
    # =====================================================
    def retrieve(self, query, top_k=5):

        if self.index.ntotal == 0:
            return []

        qvec = self.embedder.encode(query)

        qvec = np.asarray(
            qvec,
            dtype=np.float32
        )

        norm = np.linalg.norm(qvec)

        if norm > 0:
            qvec = qvec / norm

        qvec = np.expand_dims(
            qvec,
            axis=0
        )

        scores, indices = self.index.search(
            qvec,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx < 0:
                continue

            results.append({
                "text": self.metadata[idx]["text"],
                "score": float(score)
            })

        return results

    # =====================================================
    # QUERY
    # =====================================================
    def query(self, question):

        docs = self.retrieve(question)

        context = "\n\n".join(
            d["text"]
            for d in docs
        )

        return {
            "context": context,
            "docs": docs
        }

    # =====================================================
    # LLM HELPER
    # =====================================================
    def llm(
        self,
        prompt,
        url="http://127.0.0.1:1234/v1/chat/completions"
    ):

        try:

            res = requests.post(
                url,
                json={
                    "model": "qwen",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.3
                },
                timeout=60
            )

            return res.json()["choices"][0]["message"]["content"]

        except Exception as e:
            return f"LLM error: {str(e)}"