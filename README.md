# realtime_transcriber_WG
A speech to text tool using api from Scribe v2 Realtime of ElevenLabs

# Tech Stack
Backend: FastAPI（Python 3.11+）
Realtime STT: Scribe v2 Realtime of ElevenLabs
Desktop Frontend: SwiftUI + AppKit
Audio capture: AVAudioEngine

<img width="699" height="595" alt="Screenshot 2025-12-09 at 16 56 48" src="https://github.com/user-attachments/assets/91959f92-f692-4080-9010-f229fc3229a8" />
<img width="699" height="595" alt="Screenshot 2025-12-09 at 16 57 57" src="https://github.com/user-attachments/assets/90a0ec32-5951-4864-9d19-115f346f06b9" />
<img width="699" height="595" alt="Screenshot 2025-12-09 at 16 58 26" src="https://github.com/user-attachments/assets/13c7851a-a35d-4f07-88b9-932fcbf9db66" />
<img width="699" height="595" alt="Screenshot 2025-12-09 at 16 58 32" src="https://github.com/user-attachments/assets/08990919-bc2a-4ac5-ae2e-15ea01b08fb5" />

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
