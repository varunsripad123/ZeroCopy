# Zero-Copy AI

Zero-Copy AI is a reference implementation of a queryable video compression
pipeline. It showcases the major architectural components described in the
product requirements document: video chunking, latent embedding generation,
vector search, and an HTTP API for compression, semantic query, and
chunk-level decoding.

> **Note**
> The encoder and text embedding components in this repository provide a
> deterministic placeholder implementation based on hashing. They are designed
> so that transformer-based models can be integrated without changing the
> surrounding infrastructure.

## Features

- Video chunking using FFmpeg with configurable segment length.
- Latent embedding generation with deterministic hashing placeholder.
- JSONL manifest for persisted chunk metadata and embedding file references.
- In-memory cosine similarity vector index with deterministic text encoder.
- FastAPI service exposing `/compress`, `/compress/upload`, `/query`, `/decode`, and `/chunks/{chunk_id}` endpoints.
- Configurable storage directories, API host, and index rebuild behaviour.
- Tailwind + React operations console for uploading videos, running queries, and previewing decoded chunks.

## Getting Started

### Prerequisites

- Python 3.10+
- FFmpeg installed and accessible on the system path.
- (Optional) Virtual environment tooling such as `venv` or `conda`.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

### Running the API

```bash
uvicorn zerocopy.api.main:app --host 0.0.0.0 --port 8080
```

### Example Usage

```bash
# Compress a video from an existing server path
curl -X POST http://localhost:8080/compress \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/data/videos/sample.mp4", "segment_length": 3}'

# Compress by uploading a video file directly
curl -X POST http://localhost:8080/compress/upload \
  -F "file=@/path/to/local/video.mp4" \
  -F "segment_length=4" \
  -F 'metadata={"site":"warehouse-a"}'

# Query the compressed archive
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "red car entering gate", "top_k": 5}'

# Decode a specific chunk
curl -X POST http://localhost:8080/decode \
  -H "Content-Type: application/json" \
  -d '{"chunk_id": "<chunk-id-from-query>"}'

# Stream a decoded chunk directly
curl -L http://localhost:8080/chunks/<chunk-id-from-query> --output preview.mp4
```

### Configuration

Environment variables can be used to control runtime behaviour:

| Variable | Description | Default |
| --- | --- | --- |
| `ZEROCOPY_DATA_DIR` | Root directory for manifests and embeddings | `data` |
| `ZEROCOPY_MANIFEST` | Manifest filename | `manifest.jsonl` |
| `ZEROCOPY_CHUNK_DIR` | Directory for chunked video files | `chunks` |
| `ZEROCOPY_UPLOAD_DIR` | Directory for temporarily storing uploaded originals | `uploads` |
| `ZEROCOPY_SIMILARITY` | Similarity metric for the index | `cosine` |
| `ZEROCOPY_REBUILD_INDEX` | Rebuild index from manifest on startup | `false` |
| `ZEROCOPY_API_HOST` | API host binding | `0.0.0.0` |
| `ZEROCOPY_API_PORT` | API port | `8080` |
| `ZEROCOPY_CORS_ALLOW_ORIGIN` | Optional single CORS origin | _unset_ |
| `ZEROCOPY_CORS_ALLOW_ORIGINS` | Comma-separated list of CORS origins | _unset_ |
| `ZEROCOPY_VIDEOMAE_ENABLED` | Enable downloading and using the real VideoMAE encoder weights | `true` |

Setting `ZEROCOPY_VIDEOMAE_ENABLED=false` keeps the API responsive in environments without
access to the Hugging Face model hub by falling back to a deterministic stub encoder that
maintains a compatible embedding dimensionality for the FAISS index.

### Frontend Console

The repository ships with a production-ready React console that sits alongside the API. It is built with Vite, Tailwind, and Axios.

```bash
cd frontend
npm install
# Point the UI at your API (defaults to http://localhost:8080)
echo "VITE_API_BASE_URL=http://localhost:8080" > .env.local
npm run dev        # Start development server at http://localhost:5173
npm run build      # Produce a production build in frontend/dist
```

You can reverse proxy the compiled assets via FastAPI, nginx, or a static file host to expose the control plane in production.

### Testing

```bash
pytest
```

## Roadmap

- Replace placeholder encoders with VideoMAE/VQ-VAE and CLIP/VideoCLIP models.
- Add Milvus/FAISS integration for large-scale vector storage.
- Introduce background workers for ingesting high-throughput streams.
- Implement authentication/authorization for the API.
- Provide dashboard UI for monitoring compression statistics and search results.
