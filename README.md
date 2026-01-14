# 🧠 AI-Powered Real-Time Memory Assistant

> **A macOS-native productivity tool that turns voice conversations into structured knowledge.** featuring real-time transcription, intelligent context-aware summarization, and local semantic search.

---

## 📖 Overview

This project is a full-stack AI application designed to close the loop between **capturing information** and **retrieving knowledge**. Unlike traditional recorders that leave you with hours of raw audio, this assistant acts as a "second brain":

1. **Listens** using low-latency transcription (ElevenLabs Scribe).
2. **Thinks** by generating structured, context-aware summaries every ~30 seconds (GPT-4).
3. **Remembers** by indexing content locally for semantic retrieval (FAISS + SentenceTransformers).

It creates a seamless pipeline from **Speech**  **Text**  **Insight**  **Long-term Memory**.


# Realtime Transcriber 用户指南

## 快速开始

### 1. 安装

1. 双击 `RealtimeTranscriber_v1.0.0.dmg`
2. 拖动 App 到 Applications 文件夹
3. 首次打开时：
   - 右键点击 App → "打开"
   - 或在安全设置中允许

### 2. 配置 API Keys

1. 启动 App
2. 菜单栏 → RealtimeTranscriberMac → Settings (⌘,)
3. 输入 API Keys：
   - **OpenAI**: 从 https://platform.openai.com/api-keys 获取
   - **ElevenLabs**: 从 https://elevenlabs.io 获取
4. 点击 "Save & Restart Backend"

### 3. 开始使用

1. 创建项目
2. 选择模式（讲座/对话）
3. 点击 "开始录音"
4. 说话即可实时转录
5. 系统会自动生成摘要

## 常见问题

### Q: 无法打开 App
A: 右键点击 → "打开"，或在安全设置中允许

### Q: 后端无法启动
A: Settings → "Save & Restart Backend"

### Q: 转录不准确
A: 检查麦克风权限，确保 ElevenLabs API Key 正确

## 技术支持

邮箱: your-email@example.com

---

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

## 🚀 Getting Started

### Prerequisites

* macOS 13.0+ (Ventura or later)
* Python 3.10+
* Xcode 14+
* API Keys for OpenAI and ElevenLabs.

### 1. Backend Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/project-name.git
cd project-name/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "OPENAI_API_KEY=your_key" > .env
echo "ELEVENLABS_API_KEY=your_key" >> .env

# Run the server
uvicorn main:app --reload

```

### 2. Frontend Setup

1. Open `frontend/YourApp.xcodeproj` in Xcode.
2. Ensure the backend is running on `http://localhost:8000`.
3. Build and Run (Cmd + R).

---

## 📸 Screenshots

| Real-time Transcription | Semantic Search |
| --- | --- |
| *(Place screenshot of the main transcription view here)* | *(Place screenshot of the search results view here)* |

---

## 🔮 Future Roadmap

* [ ] **GPU Acceleration:** Move local embedding to Metal (MPS) for faster processing.
* [ ] **Graph View:** Visualize connections between different meetings/sessions.
* [ ] **Multi-modal:** Support indexing of shared images or screen captures during meetings.
* [ ] **Export:** Export summaries to Notion/Obsidian.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📧 Contact

**Winston (Wenguang) Dong** Master of Information Systems Management @ CMU

[https://www.linkedin.com/in/wenguang-qaz1105/] | [wenguand@andrew.cmu.edu]

---

*Built with ❤️ at Carnegie Mellon University*