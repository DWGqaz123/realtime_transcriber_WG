# 🧠 AI-Powered Real-Time Memory Assistant

> **A macOS-native productivity tool that turns voice conversations into structured knowledge.** featuring real-time transcription, intelligent context-aware summarization, and local semantic search.

---

## 📖 Overview

This project is a full-stack AI application designed to close the loop between **capturing information** and **retrieving knowledge**. Unlike traditional recorders that leave you with hours of raw audio, this assistant acts as a "second brain":

1. **Listens** using low-latency transcription (ElevenLabs Scribe). Chose between 'Lecture' and 'Discussion' mode.
2. **Thinks** by generating structured, context-aware summaries every ~120 seconds (GPT-4).
3. **Remembers** by indexing content locally for semantic retrieval (FAISS + SentenceTransformers).

It creates a seamless pipeline from **Speech**  **Text**  **Insight**  **Long-term Memory**.

---

## 📸 Screenshots

| Real-time Transcription | Transcripts Record | Semantic Search |
| --- | --- | --- |
|<img width="1399" height="762" alt="Screenshot 2026-01-15 at 20 30 06" src="https://github.com/user-attachments/assets/f13af411-323d-48ca-9af8-9a30342bf584" /> | <img width="800" height="760" alt="Screenshot 2026-01-15 at 20 30 51" src="https://github.com/user-attachments/assets/eb194900-b9ec-4e08-9602-080f07712eac" /> <img width="800" height="758" alt="Screenshot 2026-01-15 at 20 31 38" src="https://github.com/user-attachments/assets/eeb0e9bd-830f-4e41-95e7-16fda5e6b700" /> | <img width="1399" height="758" alt="Screenshot 2026-01-15 at 20 31 38" src="https://github.com/user-attachments/assets/d86ac4b5-8711-4587-83c4-fc38d5d5fc4b" /> |

---

# Realtime Transcriber User Handbook
## 📥 Download

**[Latest Release (v1.0.1)](https://github.com/DWGqaz123/realtime_transcriber_WG/releases/latest)**

## 🚀 Quick Start

1. Download and install the app
2. Configure API keys in Settings
3. Start recording and transcribing!

[Full documentation →](docs/)

## 🛠️ Development

### Requirements

- macOS 12.0+
- Xcode 14.0+
- Python 3.10+
- Conda

### Setup
```bash
# Clone repository
git clone https://github.com/你的用户名/realtime_transcriber_WG.git
cd realtime_transcriber_WG

# Setup backend
cd backend
conda create -n realtime_transcriber_env python=3.10
conda activate realtime_transcriber_env
pip install -r requirements.txt

# Open frontend in Xcode
cd ../frontend_mac/RealtimeTranscriberMac
open RealtimeTranscriberMac.xcodeproj
```

## ✨ Key Features

### ⚡️ Real-Time Intelligence

* **Live Transcription:** Integrated with ElevenLabs Scribe v2 for <100ms latency speech-to-text.
* **Smart Ticker:** Generates structured summaries (bullet points & action items) in real-time without interrupting the transcription flow.
* **Adaptive Triggering:** Uses a hybrid algorithm (Time + Sentence Count + Semantic Integrity) to ensure summaries are generated at natural pauses, not arbitrary time cuts.

### 🧠 Long-Term Memory (RAG)

* **Local Vectorization:** Uses `paraphrase-multilingual-MiniLM-L12-v2` to embed text locally on the CPU (~6 texts/sec). **Zero API cost for storage.**
* **Semantic Search:** Built on **FAISS (Facebook AI Similarity Search)**. Find content by meaning (e.g., query "Budget issues" to find segments about "financial constraints").
* **Asynchronous Indexing:** Indexing happens in the background via non-blocking tasks, ensuring the UI never freezes.

### 🛡️ Architecture Highlights

* **Double-Container Buffering:** Solves the data-loss problem common in streaming applications.
* *Ingestion Buffer:* Always open for incoming audio text.
* *Processing Snapshot:* Atomically isolated for LLM inference.


* **Privacy First:** Vector embeddings and databases (SQLite + Chroma/FAISS) are stored locally on the user's machine.

---

## 🏗 System Architecture

### 1. The "Double-Container" Pipeline

One of the core engineering challenges was handling the race condition between the continuous WebSocket stream and the latent LLM inference.

```mermaid
graph TD
    WS[WebSocket Stream] -->|Push Text| IB[📥 Ingestion Buffer]
    
    subgraph "Atomic Transfer"
        IB -->|Cut & Move| PS[📸 Processing Snapshot]
    end
    
    PS -->|Prompt Construction| GPT[OpenAI GPT-4]
    GPT -->|Summary| DB[(SQLite)]
    GPT -->|Broadcast| UI[SwiftUI Frontend]
    
    subgraph "Context Management"
        PS -->|Extract Last 3 Sentences| Cache[Context Cache]
        Cache -->|Inject into Next Prompt| GPT
    end

```

### 2. The Local RAG Engine

A privacy-focused implementation of Retrieval-Augmented Generation.

* **Ingest:** Finished sessions are chunked.
* **Embed:** `SentenceTransformers` (Local CPU) converts chunks to 384-dimensional vectors.
* **Index:** Vectors are stored in a `FAISS IndexFlatIP` structure for efficient cosine similarity search.
* **Retrieve:** User queries are vectorized and matched against the index to return the Top-K relevant segments.

---

## 🛠 Tech Stack

### Backend (Python 3.10+)

* **Framework:** FastAPI (Async Web Server)
* **Real-time:** WebSockets
* **Database:** SQLAlchemy + SQLite
* **AI & ML:**
* OpenAI API (Summarization)
* ElevenLabs API (Scribe v2 STT)
* SentenceTransformers (Local Embedding)
* FAISS (Vector Indexing)



### Frontend (macOS)

* **Language:** Swift 5
* **UI Framework:** SwiftUI
* **Architecture:** MVVM + Combine
* **Audio:** AVFoundation (High-performance capture)

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📧 Contact

**Winston (Wenguang) Dong** Master of Information Systems Management @ CMU

[https://www.linkedin.com/in/wenguang-qaz1105/] | [wenguand@andrew.cmu.edu]

---

*Built with ❤️ at Carnegie Mellon University*
