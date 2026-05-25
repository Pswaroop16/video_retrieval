# 🎬 Video Retrieval System

A production-grade multimodal video retrieval system that finds the most visually similar video from a database — using either an **image** or a **video clip** as the query. Powered by OpenCLIP (ViT-B/32) for visual embeddings and FAISS for fast nearest-neighbour search.

---

## 📌 Overview

Videos in the database are pre-processed into mean-pooled CLIP embeddings and stored in a FAISS index. At query time, your image or video is encoded the same way and matched against the index using cosine similarity. The system returns ranked results in milliseconds — even for large databases.

---

## 🗂️ Project Structure

```
video_retrieval/
│
├── main.py                    # CLI entry point — wires all services together
│
├── config/
│   ├── __init__.py
│   └── settings.py            # All paths, model names, and constants
│
├── core/
│   ├── __init__.py
│   ├── embedder.py            # CLIPEmbedder — L2-normalised image embeddings
│   ├── video_processor.py     # VideoProcessor — extracts frames from video files
│   └── index_manager.py       # IndexManager — FAISS index build / save / load / search
│
├── retrieval/
│   ├── __init__.py
│   ├── indexer.py             # DatabaseIndexer — scans DB, encodes videos, builds index
│   └── searcher.py            # VideoSearcher — image-to-video & video-to-video queries
│
├── utils/
│   ├── __init__.py
│   ├── file_utils.py          # File/directory validation and video file iteration
│   └── logger.py              # Centralised rotating-file + console logger
│
├── database/                  # 📂 Place your database videos here (.mp4, .avi, .mov …)
├── query_images/              # 📂 Place query images here
├── query_videos/              # 📂 Place query video clips here
├── artifacts/                 # Auto-generated: faiss.index + video_paths.npy (saved index)
├── logs/                      # Auto-generated: retrieval.log
│
├── tests/
│   ├── __init__.py
│   └── test_core.py           # Unit tests for core services
│
├── requirements.txt
└── .gitignore
```

---

## 🧠 Architecture

```
Query (image / video)
        │
        ▼
 VideoProcessor          ← extracts frames (video queries only)
        │
        ▼
  CLIPEmbedder           ← encodes frames → L2-normalised vectors
        │                   (mean-pools for video queries)
        ▼
  IndexManager           ← FAISS IndexFlatIP cosine similarity search
        │
        ▼
 RetrievalResult[]       ← ranked list: rank, video path, score
```

**Why FAISS `IndexFlatIP`?**  
Embeddings are L2-normalised, so inner product equals cosine similarity — exactly the right metric for CLIP features.

---

## ⚙️ Configuration

All settings live in `config/settings.py`. Edit once, applies everywhere:

```python
VIDEO_DB_DIR     = BASE_DIR / "database"          # video database folder
QUERY_IMAGE_DIR  = BASE_DIR / "query_images"      # image query folder
QUERY_VIDEO_DIR  = BASE_DIR / "query_videos"      # video query folder

CLIP_MODEL_NAME  = "ViT-B-32"                     # OpenCLIP architecture
CLIP_PRETRAINED  = "openai"                        # pretrained weights
EMBEDDING_DIM    = 512                             # ViT-B-32 output size

FRAME_SAMPLES    = 15                              # frames sampled per video
TOP_K            = 1                               # results returned per query
DEVICE           = "cuda" / "cpu"                  # auto-detected
```

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/video-retrieval-system.git
cd video-retrieval-system
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** swap `faiss-cpu` for `faiss-gpu` in `requirements.txt` for faster search.

### 4. Add your videos

```
database/         ← add .mp4 / .avi / .mov / .mkv / .webm files here
query_images/     ← add .jpg / .png query images here
query_videos/     ← add query video clips here
```

---

## ▶️ Usage

### Interactive mode (no arguments)

```bash
python main.py
```
```
┌─────────────────────────────────────┐
│   Multimodal Video Retrieval System │
└─────────────────────────────────────┘

Choose Query Type:
  1 - Image query
  2 - Video query

Enter choice [1/2]:
```

### CLI mode

```bash
# Build index and query with an image
python main.py --query-type image --query-file sunset.jpg

# Query with a video clip (reuse saved index)
python main.py --query-type video --query-file clip.mp4 --load-index

# Build index only, no query
python main.py --build-only

# Return top-3 results
python main.py --query-type image --query-file frame.png --top-k 3
```

### Example output

```
─────────────────────────────────────────────
  Top 1 Result(s)
─────────────────────────────────────────────
  [Rank 1] yoga.mp4  (score=0.9124)
─────────────────────────────────────────────
```

---

## 📦 Requirements

| Package | Purpose |
|---------|---------|
| `torch >= 2.1.0` | Deep learning backend |
| `torchvision >= 0.16.0` | Required by OpenCLIP internally |
| `open-clip-torch >= 2.24.0` | CLIP model (ViT-B-32) |
| `opencv-python >= 4.9.0` | Video frame extraction |
| `faiss-cpu >= 1.7.4` | Nearest-neighbour index |
| `numpy >= 1.26.0` | Numerical operations |
| `Pillow >= 10.2.0` | Image loading |
| `tqdm >= 4.66.0` | Progress bars |

---

## 🧪 Running Tests

```bash
pytest tests/test_core.py -v
```

---

## 🛠️ How It Works

**Indexing (one-time)**
1. `DatabaseIndexer` scans `database/` for supported video files.
2. `VideoProcessor` samples 15 frames evenly across each video using `np.linspace`.
3. `CLIPEmbedder` encodes all frames and mean-pools them into a single L2-normalised vector per video.
4. `IndexManager` builds a FAISS `IndexFlatIP` and saves it to `artifacts/`.

**Querying (fast, reusable)**
1. Load or reuse the saved FAISS index.
2. Encode the query (single image → 1 embedding; video → mean-pooled frames).
3. Run inner-product search → ranked `RetrievalResult` list.

---

## 💡 Notes

- GPU (CUDA) is detected and used automatically; falls back to CPU.
- The FAISS index is saved to `artifacts/` after building — use `--load-index` to skip re-indexing on subsequent runs.
- Logs are written to `logs/retrieval.log` with rotation (5 MB × 3 backups).
- Supported video formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`.

---

## 🔮 Future Extensions

The codebase is structured to make these easy to add:

- **Text-to-video search** — CLIP text encoder already pairs with the same embedding space
- **REST API** — `fastapi` is listed as an optional dependency
- **Streamlit UI** — `streamlit` is listed as an optional dependency
- **Larger FAISS indexes** — swap `IndexFlatIP` for `IndexIVFFlat` for million-scale databases

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
