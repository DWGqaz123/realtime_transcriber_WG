# Realtime Transcriber

> A macOS app that turns voice conversations into structured knowledge — real-time transcription, auto-generated summaries, and semantic search across your sessions.

**[Download Latest Release](https://github.com/DWGqaz123/realtime_transcriber_WG/releases/latest)**

---

## What it does

1. **Transcribes** live audio via ElevenLabs Scribe with <100ms latency. Choose between Lecture or Discussion mode.
2. **Summarizes** automatically every ~2 minutes using GPT-4o, with context carried between summaries.
3. **Indexes** completed sessions locally (FAISS + embeddings) for semantic search across your history.

---

## Getting Started

1. Download and open the `.dmg` from [Releases](https://github.com/DWGqaz123/realtime_transcriber_WG/releases)
2. Drag the app to Applications
3. On first launch, go to **Settings (⌘,)** and enter your API keys:
   - **OpenAI API Key** — for summarization
   - **ElevenLabs API Key** — for transcription
4. Click **Save & Restart Backend**, then start a session

> First launch may show a Gatekeeper warning. Go to System Settings → Privacy & Security → click "Open Anyway".

---

## Requirements

- macOS 12.0 (Monterey) or later
- Apple Silicon or Intel
- Internet connection (for transcription and summarization APIs)

---

## Features

- **Live subtitles** — real-time transcription displayed as you speak
- **Auto summaries** — bullet-point summaries generated at natural pauses, no manual intervention
- **Session management** — organize sessions into projects, add names and notes
- **Semantic search** — query across all past sessions by meaning, not just keywords
- **Language setting** — configure summary output language in Settings
- **Local storage** — all transcripts and embeddings stored in `~/Library/Application Support/RealtimeTranscriber/`

---

## Engineering Highlights

### Double-Container Buffering

The core challenge of real-time summarization is that LLM inference (~2–5s) overlaps with a continuous audio stream. A naive single-buffer approach either blocks incoming transcription or corrupts the snapshot mid-inference.

The solution uses two decoupled containers:

```
Incoming transcription  →  [Ingestion Buffer]  →  atomic snapshot  →  [Processing Snapshot]  →  OpenAI
                                ↑                                                                    ↓
                           always writable                                                    context cache
```

When summarization triggers, `ingestion_buffer` is copied and immediately cleared — the stream continues writing to the now-empty buffer without any lock. The snapshot is passed to the LLM independently. The last N sentences are retained in a `context_cache` and injected as background context into the next prompt, ensuring coherence across summary windows.

### Connection Pooling for OpenAI API

`SummaryService` holds a single `httpx.AsyncClient` instance for its lifetime instead of creating one per API call. Each new client requires a full TCP + TLS handshake (~100–400ms for overseas endpoints). With a persistent connection, subsequent calls reuse the established session — directly observable as a speedup after the first summary in a session.

### Adaptive Summary Triggering

Summaries are not triggered on a fixed timer. The trigger logic uses a three-condition hybrid:

1. **Time elapsed** ≥ `SUMMARY_INTERVAL_SECONDS` (primary gate)
2. **Min sentence count** in the buffer (prevents summaries on near-empty content)
3. **Semantic integrity check** — waits for the last buffered sentence to end with a sentence-ender (`.`, `?`, `!`); skips this check and fires anyway if elapsed time exceeds `LOOSE_MODE_THRESHOLD`

This avoids cutting mid-sentence, producing more coherent summaries without complex NLP.

### Asynchronous FAISS Indexing

After a session ends, embedding and indexing happen in a background `asyncio` task, fully non-blocking to the WebSocket handler. Vectors are stored in a per-project `IndexFlatIP` (inner product, equivalent to cosine similarity on normalized vectors). The FAISS manager returns assigned vector IDs directly from `add_vectors`, eliminating the reverse-mapping lookup that would otherwise require iterating the full index.

---

## Tech Stack

| Layer | Stack |
|-------|-------|
| Frontend | Swift / SwiftUI |
| Backend | Python / FastAPI |
| Transcription | ElevenLabs Scribe v2 |
| Summarization | OpenAI GPT-4o |
| Search | FAISS + SentenceTransformers |
| Database | SQLite via SQLAlchemy |

---

## Contact

**Winston (Wenguang) Dong** — MISM @ CMU
[LinkedIn](https://www.linkedin.com/in/wenguang-qaz1105/) · wenguand@andrew.cmu.edu
