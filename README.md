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

<<<<<<< HEAD
# Tech detail 
### (Atomic Transfer)
```
时间轴：
T=5:00.000  触发条件满足
            ↓
T=5:00.001  【原子操作开始】
            ├─ Step 1: 复制
            │  processing_snapshot = ingestion_buffer.copy()
            │
            ├─ Step 2: 清空
            │  ingestion_buffer.clear()
            │
            ├─ Step 3: 设置锁
            │  is_generating = True
            │
            └─ Step 4: 记录时间
               last_summary_time = now
            【原子操作结束 - 总耗时 <1ms】
            ↓
T=5:00.002  接收缓冲区已就绪，可接收新数据
            处理快照开始异步处理（不影响接收）
            ↓
T=5:00.100  新的 [final] 到达
            → 直接写入 ingestion_buffer
            → processing_snapshot 不受影响
            ↓
T=5:15.000  LLM 返回摘要（耗时 15 秒）
            ↓
            保存、推送、更新上下文
            ↓
            is_generating = False
```

**关键点**：
1. ⚡ **微秒级操作**：复制和清空在 <1ms 内完成
2. 🔒 **立即加锁**：防止重复触发
3. 🎯 **数据隔离**：新旧数据完全分离
4. ⏱️ **立即重置计时器**：防止时间判断误差

---

### 上下文桥接机制 (Context Bridging)

#### 上下文管理策略
```
生命周期：

录音开始
  ↓
context_cache = []  (空)
  ↓
第一次摘要
  ├─ Input:  context_cache (空) + processing_snapshot (句1-10)
  ├─ Output: Summary 1
  └─ Update: context_cache = [句8, 句9, 句10]  (最后3句)
  ↓
第二次摘要 (5分钟后)
  ├─ Input:  context_cache (句8-10) + processing_snapshot (句11-20)
  ├─ Output: Summary 2
  └─ Update: context_cache = [句18, 句19, 句20]
  ↓
第三次摘要
  ├─ Input:  context_cache (句18-20) + processing_snapshot (句21-30)
