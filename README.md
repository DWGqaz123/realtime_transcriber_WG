# realtime_transcriber_WG
A speech to text tool using api from Scribe v2 Realtime of ElevenLabs

# Tech Stack
Backend: FastAPI（Python 3.11+）
Realtime STT: Scribe v2 Realtime of ElevenLabs
Desktop Frontend: SwiftUI + AppKit
Audio capture: AVAudioEngine

# Experiment
## Experiment A

Behavior was measured at CHUNK_MS = 100 / 200 / 300ms using two 30 second test audios (From real class' recording on zoom):

Key findings:
1. first subtitle (partial) delay stabilizes at ~2 sec.
→ determined by the model's "first 2 seconds startup", not strongly correlated with chunk size. 
2. final paragraph (committed) appears at the end of the whole paragraph.
→ Default VAD treats long continuous speech as one paragraph, suitable for short conversations but not for long classroom statements.
3. chunk mainly affects sending frequency, not critical delay
→ 200ms is the best compromise between smoothness and load.

All events and summaries during the experiment are automatically disked to runs/YYYY-MM-DD_xxxxxxx/.

## Experiment B

We compared the two official segmentation modes:

VAD Mode (Voice Activity Detection)
	•	Suitable for conversations, interviews, meetings
	•	Fastest “finalized” text and most responsive UI
	•	But breaks long sentences → not ideal for lectures

Manual Mode (commit every 35 seconds)
	•	Predictable, clean paragraph chunks
	•	Much more natural for continuous speech
	•	Ideal for:
	•	Classes & lectures
	•	Long explanations
	•	Notes summarization


```

    ---


    # 数据存储位置

所有用户数据存储在：
```
~/Library/Application Support/RealtimeTranscriber/
├── transcripts.db                   # 数据库
└── transcripts/                     # 转录文件
    └── {项目名}/
        └── {时间戳}_{模式}.txt
```

## 卸载

如需完全删除应用数据：
```bash
rm -rf ~/Library/Application\ Support/RealtimeTranscriber/
```