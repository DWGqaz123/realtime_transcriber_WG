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
	•	~10 committed segments for a 30-second sample
	•	Commit every 2–3 seconds
	•	Suitable for conversations, interviews, meetings
	•	Fastest “finalized” text and most responsive UI
	•	But breaks long sentences → not ideal for lectures

Manual Mode (commit every 15 seconds)
	•	Exactly 2 segments for the 30-second sample
	•	Predictable, clean paragraph chunks
	•	Much more natural for continuous speech
	•	Ideal for:
	•	Classes & lectures
	•	Long explanations
	•	Notes summarization

## Experiment C
### 1. Commit interval = **8 seconds**
- Produces short, sentence-level segments  
- Highly responsive, good for dictation or short-turn speech  
- Too fragmented for long explanations  
→ Best for **Dictation Mode** or fast-paced Q&A

### 2. Commit interval = **12 seconds**
- Produces natural, readable segments (1–2 sentences each)  
- Excellent balance between semantic completeness and real-time updates  
- Matches the rhythm of student presentations or structured speaking  
→ Test1 shows **12s is the optimal value** for presentations  
→ Recommended default for **Presentation Mode / Medium-length speech**

### 3. Commit interval = **20 seconds**
- Forms longer, more coherent paragraph-like segments  
- Matches continuous lecture-style speech  
- Less real-time feedback, but best semantic structure  
→ Test2 shows **20s performs best** for continuous professor lectures  
→ Recommended default for **Lecture Mode (long-form speech)**

## Experiment D

More VAD parameters adjuestment  TBD


