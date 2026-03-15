//
//  TranscribeViewModel.swift
//  RealtimeTranscriberMac
//
//  Created by 董文光 on 2025/11/14.
//

import Foundation
import Combine

private struct WebSocketEvent: Codable {
    let type: String
    let payload: Payload

    struct Payload: Codable {
        let id: Int?
        let content: String?
        let created_at: String?
        let text: String?
        let message: String?
        let status: String?
        let project_id: Int?
        let mode: String?
        let is_final: Bool?
        let sentence_count: Int?
        let session_id: Int?
        let duration: Int?
        let sentences: Int?
        let characters: Int?
        let summaries: Int?
        let indexed_count: Int?
    }
}

private struct SummaryTimingConfig: Decodable {
    let summary_interval_seconds: Int
}

enum RecordingMode: String, CaseIterable, Identifiable {
    case lecture
    case discussion

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .lecture: return "Lecture"
        case .discussion: return "Discussion"
        }
    }
}

@MainActor
final class TranscribeViewModel: ObservableObject {
    // MARK: - Published Properties
    
    // 录音状态
    @Published var isRecording: Bool = false
    @Published var permissionStatus: String = "Unknown"
    @Published var showPermissionAlert: Bool = false
    
    // 转录内容
    @Published var currentSubtitle: String = ""
    @Published var fullTranscript: String = ""
    
    // 模式和项目
    @Published var mode: RecordingMode = .lecture
    @Published var currentProjectId: Int?  // 🔧 项目关联
    
    // 音频可视化
    @Published var audioLevel: Float = 0.0
    @Published var isDetectingSound: Bool = false
    @Published var recordingDuration: TimeInterval = 0.0
    
    // 摘要列表
    @Published var summaries: [Summary] = []
    
    // 🔧 新增：摘要生成状态
    @Published var isGeneratingSummary: Bool = false
    @Published var nextSummaryCountdown: Int = 0  // 倒计时秒数
    @Published var lastSummaryTimestamp: Date? = nil
    @Published var summaryGenerationProgress: String = ""  // 进度文本

    // 🔧 新增：计时器
    private var summaryCountdownTimer: Timer?

    // 🔧 新增：配置常量
    @Published private(set) var summaryIntervalSeconds: Int = 30
    
    // MARK: - Private Properties
    
    private let client = TranscriptionClient()
    private let audioCapture = AudioCaptureService()
    private let apiClient = APIClient()
    private var recordingTimer: Timer?
    private let summaryDecoder = JSONDecoder()
 
    
    // MARK: - Initialization
    
    init() {
        configureDecoders()
        setupTranscriptionClient()
        setupAudioCapture()
        checkMicrophonePermission()
        Task {
            await loadSummaryTimingConfig()
        }
    }
    
    deinit {
        recordingTimer?.invalidate()
        summaryCountdownTimer?.invalidate()
    }
    
    // MARK: - Setup Methods

    private func configureDecoders() {
        summaryDecoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let dateString = try container.decode(String.self)

            let formatter1 = DateFormatter()
            formatter1.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
            formatter1.locale = Locale(identifier: "en_US_POSIX")
            formatter1.timeZone = TimeZone(secondsFromGMT: 0)

            if let date = formatter1.date(from: dateString) {
                return date
            }

            let formatter2 = DateFormatter()
            formatter2.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
            formatter2.locale = Locale(identifier: "en_US_POSIX")
            formatter2.timeZone = TimeZone(secondsFromGMT: 0)

            if let date = formatter2.date(from: dateString) {
                return date
            }

            let iso8601Formatter = ISO8601DateFormatter()
            if let date = iso8601Formatter.date(from: dateString) {
                return date
            }

            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Cannot decode date: \(dateString)"
            )
        }
    }
    
    private func setupTranscriptionClient() {
        // 处理来自后端的消息
        client.onMessage = { [weak self] text in
            guard let self = self else { return }
            Task { @MainActor in
                self.handleTranscriptMessage(text)
            }
        }
    }
    
    private func setupAudioCapture() {
        // 发送音频数据
        audioCapture.onAudioData = { [weak self] data in
            self?.client.sendAudio(data)
        }
        
        // 监听音频电平
        audioCapture.onAudioLevel = { [weak self] level in
            guard let self = self else { return }
            Task { @MainActor in
                self.audioLevel = level
                self.isDetectingSound = level > 0.01
            }
        }
    }
    
    // MARK: - Microphone Permission
    
    private func checkMicrophonePermission() {
        audioCapture.requestPermission { [weak self] granted in
            guard let self = self else { return }
            Task { @MainActor in
                if granted {
                    self.permissionStatus = "Granted ✅"
                } else {
                    self.permissionStatus = "Denied ❌"
                    self.showPermissionAlert = true
                }
            }
        }
    }
    
    // MARK: - Recording Control
    
    func startRecording() {
        // 先请求麦克风权限
        audioCapture.requestPermission { [weak self] granted in
            guard let self = self else { return }
            
            Task { @MainActor in
                await self.loadSummaryTimingConfig()
                // 检查权限
                if !granted {
                    self.permissionStatus = "Denied - Check System Settings ❌"
                    self.showPermissionAlert = true
                    return
                }
                //remore all possible remains summary
                self.summaries.removeAll()
                
                // 🔧 检查项目 ID
                guard let projectId = self.currentProjectId else {
                    self.currentSubtitle = "Please select a project first"
                    self.permissionStatus = "No project selected ⚠️"
                    return
                }
                
                // 设置录音状态
                self.isRecording = true
                self.currentSubtitle = ""
                // 🔧 只在已保存后才清空 fullTranscript
                // 如果 fullTranscript 不为空，说明上次录音未保存，继续累积
                if self.fullTranscript.isEmpty {
                    // 新会话，无需操作
                } else {
                    // 继续使用现有 fullTranscript，不清空
                }
                
                
                self.permissionStatus = "Recording... 🎤"
                self.recordingDuration = 0.0
                self.audioLevel = 0.0
                self.isDetectingSound = false
                
                // 启动录音计时器
                self.recordingTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
                    guard let self = self else { return }
                    Task { @MainActor in
                        self.recordingDuration += 0.1
                    }
                }
                
                // 1. 连接 WebSocket
                self.client.connect()
                try? await Task.sleep(nanoseconds: 100_000_000)  // 0.1秒
                // 2. 🔧 发送项目 ID
                self.client.send(text: "PROJECT_ID:\(projectId)")
                
                // 3. 发送模式配置
                let modeString = self.mode.rawValue
                self.client.send(text: "MODE:\(modeString)")
                
                // 4. 开始音频采集
                do {
                    try self.audioCapture.start()
                    // 🔧 启动摘要倒计时
                    self.startSummaryCountdown()
                } catch {
                    self.currentSubtitle = "Error: \(error.localizedDescription)"
                    self.isRecording = false
                    self.recordingTimer?.invalidate()
                    self.recordingTimer = nil
                }
            }
        }
    }
    
    func stopRecording() {
        guard isRecording else { return }
        
        // 🔧 在停止前，将 currentSubtitle 添加到 fullTranscript
        if !currentSubtitle.isEmpty {
            
            if !fullTranscript.isEmpty {
                fullTranscript += "\n"
            }
            fullTranscript += currentSubtitle
            
        }
        
        isRecording = false
        currentSubtitle = ""  // 清空字幕
        permissionStatus = "Recording stopped - Ready to save 💾"
        
        // 停止倒计时
        stopSummaryCountdown()
        
        // 停止计时器
        recordingTimer?.invalidate()
        recordingTimer = nil
        
        // 停止音频采集
        audioCapture.stop()
        
        // 发送停止信号（但不断开 WebSocket）
        client.send(text: "STOP")
        
        // 重置音频状态
        audioLevel = 0
        isDetectingSound = false
        
    }
    
    func saveSession() {
        guard !fullTranscript.isEmpty else {
            currentSubtitle = "No transcript to save"
            return
        }
        
        currentSubtitle = "Saving session..."
        
        // send save command
        client.send(text: "SAVE")
    }
    func clearTranscript() {
        fullTranscript = ""
        currentSubtitle = ""
    }
    
    
    // MARK: - Transcript Handling
    
    private func handleTranscriptMessage(_ text: String) {
        guard let data = text.data(using: .utf8),
              let event = try? JSONDecoder().decode(WebSocketEvent.self, from: data) else {
            self.currentSubtitle = text
            return
        }

        switch event.type {
        case "config", "status", "echo":
            break
        case "partial":
            currentSubtitle = event.payload.text ?? ""
        case "final":
            let content = event.payload.text ?? ""
            currentSubtitle = ""
            if !fullTranscript.isEmpty {
                fullTranscript += "\n"
            }
            fullTranscript += content
        case "summary_started":
            isGeneratingSummary = true
            if event.payload.is_final == true {
                summaryGenerationProgress = "🏁 Generating final summary of entire recording..."
            } else {
                summaryGenerationProgress = "🤖 Generating summary..."
            }
        case "summary":
            handleSummaryEvent(from: data)
        case "save_complete":
            currentSubtitle = "✅ Session saved successfully"
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                if self.currentSubtitle.contains("saved successfully") {
                    self.currentSubtitle = ""
                }
            }
            client.disconnect()
            fullTranscript = ""
            summaries.removeAll()
            permissionStatus = "Ready to record 🎤"
        case "save_status":
            handleSaveStatus(event.payload)
        case "indexing_start":
            break
        case "indexing_complete":
            break
        case "indexing_error":
            errorMessageIfPresent(event.payload.message ?? "Indexing failed")
        case "error":
            errorMessageIfPresent(event.payload.message ?? "Unknown error")
        default:
            if let message = event.payload.message ?? event.payload.text {
                currentSubtitle = message
            }
        }
    }

    private func handleSummaryEvent(from data: Data) {
        do {
            let event = try summaryDecoder.decode(WebSocketEvent.self, from: data)
            let payloadData = try JSONEncoder().encode(event.payload)
            let summary = try JSONDecoder().decode(Summary.self, from: payloadData)

            isGeneratingSummary = false
            summaries.insert(summary, at: 0)
            resetSummaryCountdown()

            currentSubtitle = summary.isFinal ? "🏁 Final summary generated!" : "✨ New summary generated!"
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                if self.currentSubtitle.contains("summary generated") {
                    self.currentSubtitle = ""
                }
            }
        } catch {
            isGeneratingSummary = false
        }
    }

    private func handleSaveStatus(_ payload: WebSocketEvent.Payload) {
        switch payload.status {
        case "error":
            currentSubtitle = "❌ \(payload.message ?? "Save failed")"
        default:
            if let message = payload.message {
                currentSubtitle = message
            }
        }
    }

    private func errorMessageIfPresent(_ message: String) {
        currentSubtitle = "❌ \(message)"
        isGeneratingSummary = false
    }
    
    func clearSummaries() {
        summaries.removeAll()
    }
    
    // MARK: - Computed Properties
    
    var formattedDuration: String {
        let minutes = Int(recordingDuration) / 60
        let seconds = Int(recordingDuration) % 60
        let milliseconds = Int((recordingDuration.truncatingRemainder(dividingBy: 1)) * 10)
        return String(format: "%02d:%02d.%d", minutes, seconds, milliseconds)
    }
    
    var sentenceCount: Int {
        guard !fullTranscript.isEmpty else { return 0 }
        return fullTranscript.components(separatedBy: "\n").filter { !$0.isEmpty }.count
    }
    
    // MARK: - Summary Progress Management

    /// 启动摘要倒计时
    private func startSummaryCountdown() {
        // 重置状态
        lastSummaryTimestamp = Date()
        nextSummaryCountdown = summaryIntervalSeconds
        
        // 启动计时器
        summaryCountdownTimer?.invalidate()
        summaryCountdownTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            
            Task { @MainActor in
                self.updateSummaryCountdown()
            }
        }
        
    }

    /// 更新倒计时
    private func updateSummaryCountdown() {
        guard let lastTimestamp = lastSummaryTimestamp else { return }
        
        let elapsed = Int(Date().timeIntervalSince(lastTimestamp))
        let remaining = max(0, summaryIntervalSeconds - elapsed)
        
        nextSummaryCountdown = remaining
        
        // 更新进度文本
        if isGeneratingSummary {
            summaryGenerationProgress = "🤖 Generating summary..."
        } else if remaining > 0 {
            summaryGenerationProgress = "⏳ Next summary in: \(remaining)s"
        } else {
            summaryGenerationProgress = "⏳ Waiting for complete sentence..."
        }
    }

    /// 停止倒计时
    private func stopSummaryCountdown() {
        summaryCountdownTimer?.invalidate()
        summaryCountdownTimer = nil
        nextSummaryCountdown = 0
        summaryGenerationProgress = ""
    }

    /// 重置倒计时（摘要生成后）
    private func resetSummaryCountdown() {
        lastSummaryTimestamp = Date()
        nextSummaryCountdown = summaryIntervalSeconds
    }

    private func loadSummaryTimingConfig() async {
        do {
            let config: SummaryTimingConfig = try await apiClient.get("api/config/summary")
            if config.summary_interval_seconds > 0 {
                summaryIntervalSeconds = config.summary_interval_seconds
                if nextSummaryCountdown > summaryIntervalSeconds {
                    nextSummaryCountdown = summaryIntervalSeconds
                }
            }
        } catch {
            // Keep the current local value as fallback if backend is unavailable.
        }
    }
}
