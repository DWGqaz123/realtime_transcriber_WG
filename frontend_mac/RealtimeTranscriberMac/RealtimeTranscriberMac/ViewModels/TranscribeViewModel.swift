//
//  TranscribeViewModel.swift
//  RealtimeTranscriberMac
//
//  Created by 董文光 on 2025/11/14.
//

import Foundation
import Combine

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
    
    // 静音检测和流量统计
    @Published var isSilent: Bool = false
    @Published var totalChunks: Int = 0
    @Published var sentChunks: Int = 0
    @Published var skippedChunks: Int = 0
    @Published var trafficSavedPercent: Int = 0
    // 摘要列表
    @Published var summaries: [Summary] = []
    
    // MARK: - Private Properties
    
    private let client = TranscriptionClient()
    private let audioCapture = AudioCaptureService()
    private var recordingTimer: Timer?
    private var recordingStartTime: Date?  // 🔧 添加录音开始时间
 
    
    // MARK: - Initialization
    
    init() {
        setupTranscriptionClient()
        setupAudioCapture()
        checkMicrophonePermission()
    }
    
    // MARK: - Setup Methods
    
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
        
        // 监听静音状态变化
        audioCapture.onSilenceStateChanged = { [weak self] isSilent in
            guard let self = self else { return }
            Task { @MainActor in
                self.isSilent = isSilent
            }
        }
        
        // 监听统计信息更新
        audioCapture.onStatisticsUpdated = { [weak self] total, sent, skipped in
            guard let self = self else { return }
            Task { @MainActor in
                self.totalChunks = total
                self.sentChunks = sent
                self.skippedChunks = skipped
                self.trafficSavedPercent = total > 0 ? Int(Double(skipped) / Double(total) * 100) : 0
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
                // 检查权限
                if !granted {
                    print("❌ Cannot start recording: Microphone permission denied")
                    self.permissionStatus = "Denied - Check System Settings ❌"
                    self.showPermissionAlert = true
                    return
                }
                //remore all possible remains summary
                self.summaries.removeAll()
                
                // 🔧 检查项目 ID
                guard let projectId = self.currentProjectId else {
                    print("❌ No project selected")
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
                    print("📝 Continuing previous session, transcript preserved")
                }
                
                
                
                self.permissionStatus = "Recording... 🎤"
                self.recordingDuration = 0.0
                self.audioLevel = 0.0
                self.isDetectingSound = false
                self.isSilent = false
                self.recordingStartTime = Date()
                
                // 重置统计
                self.totalChunks = 0
                self.sentChunks = 0
                self.skippedChunks = 0
                self.trafficSavedPercent = 0
                
                // 启动录音计时器
                self.recordingTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
                    guard let self = self else { return }
                    Task { @MainActor in
                        self.recordingDuration += 0.1
                    }
                }
                
                // 1. 连接 WebSocket
                self.client.connect()
                
                // 2. 🔧 发送项目 ID（新增）
                self.client.send(text: "PROJECT_ID:\(projectId)")
                print("📁 Sent project ID: \(projectId)")
                
                // 3. 发送模式配置
                let modeString = self.mode.rawValue
                self.client.send(text: "MODE:\(modeString)")
                print("📡 Sent mode: \(modeString)")
                
                // 4. 开始音频采集
                do {
                    try self.audioCapture.start()
                    print("✅ Recording started successfully")
                } catch {
                    print("❌ Failed to start audio capture: \(error)")
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
            print("📝 Committing current subtitle to transcript before stop")
            print("   Current subtitle: \(currentSubtitle)")
            
            if !fullTranscript.isEmpty {
                fullTranscript += "\n"
            }
            fullTranscript += currentSubtitle
            
            print("✅ Committed, clearing currentSubtitle")
        }
        
        isRecording = false
        currentSubtitle = ""  // 清空字幕
        permissionStatus = "Recording stopped - Ready to save 💾"
        
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
        isSilent = false
        
        print("🛑 Recording stopped (Session kept alive for saving)")
    }
    
    func saveSession() {
        guard !fullTranscript.isEmpty else {
            print("⚠️ No transcript to save")
            currentSubtitle = "No transcript to save"
            return
        }
        
        print("💾 Saving session...")
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
        if text.hasPrefix("[config]") {
            print("Config: \(text)")
            
        } else if text.hasPrefix("[partial]") {
            let content = text.replacingOccurrences(of: "[partial] ", with: "")
            self.currentSubtitle = content
            
        } else if text.hasPrefix("[final]") {
            let content = text.replacingOccurrences(of: "[final] ", with: "")
            self.currentSubtitle = ""
            
            if !self.fullTranscript.isEmpty {
                self.fullTranscript += "\n"
            }
            self.fullTranscript += content
            
        }else if text.hasPrefix("[save]") {
            // save
            let content = text.replacingOccurrences(of: "[save] ", with: "")
            print("💾 Save response: \(content)")
            
            if content.contains("successfully") {
                self.currentSubtitle = "✅ Session saved successfully"
                
                // clear info after 3 s
                DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                    if self.currentSubtitle.contains("saved successfully") {
                        self.currentSubtitle = ""
                    }
                }
                
                // clear
                self.client.disconnect()
                self.fullTranscript = ""
                self.summaries.removeAll()
                self.permissionStatus = "Ready to record 🎤"
                
            } else if content.contains("ERROR") {
                self.currentSubtitle = "❌ \(content)"
            } else if content.contains("already saved") {
                self.currentSubtitle = "⚠️ Session already saved"
                self.summaries.removeAll()
            }
            
        } else if text.hasPrefix("[summary]") {
            // 🔧 新增：处理摘要消息
            let jsonString = text.replacingOccurrences(of: "[summary] ", with: "")
            print("📝 Received summary: \(jsonString.prefix(100))...")
            do {
                let jsonData = jsonString.data(using: .utf8)!
                let decoder = JSONDecoder()
                
                // 🔧 修改：使用自定义日期解码策略
                // 🔧 自定义日期解码策略
                decoder.dateDecodingStrategy = .custom { decoder in
                    let container = try decoder.singleValueContainer()
                    let dateString = try container.decode(String.self)
                    
                    // 尝试多种格式
                    // 格式 1: 2025-12-08T23:19:23.703617
                    let formatter1 = DateFormatter()
                    formatter1.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
                    formatter1.locale = Locale(identifier: "en_US_POSIX")
                    formatter1.timeZone = TimeZone(secondsFromGMT: 0)
                    
                    if let date = formatter1.date(from: dateString) {
                        return date
                    }
                    
                    // 格式 2: 2025-12-08T23:19:23
                    let formatter2 = DateFormatter()
                    formatter2.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
                    formatter2.locale = Locale(identifier: "en_US_POSIX")
                    formatter2.timeZone = TimeZone(secondsFromGMT: 0)
                    
                    if let date = formatter2.date(from: dateString) {
                        return date
                    }
                    
                    // 格式 3: 标准 ISO8601
                    let iso8601Formatter = ISO8601DateFormatter()
                    if let date = iso8601Formatter.date(from: dateString) {
                        return date
                    }
                    
                    throw DecodingError.dataCorruptedError(
                        in: container,
                        debugDescription: "Cannot decode date: \(dateString)"
                    )
                }
                
                let summary = try decoder.decode(Summary.self, from: jsonData)
                
                // 添加到列表
                self.summaries.insert(summary, at: 0)
                
                print("✅ Summary added successfully!")
                print("   ID: \(summary.id)")
                print("   Content length: \(summary.content.count) chars")
                print("   Total summaries: \(self.summaries.count)")
                
                // 显示提示
                self.currentSubtitle = "✨ New summary generated"
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                    if self.currentSubtitle == "✨ New summary generated" {
                        self.currentSubtitle = ""
                    }
                }
                
            } catch DecodingError.dataCorrupted(let context) {
                print("❌ Decoding error - data corrupted:")
                print("   \(context.debugDescription)")
                print("   JSON: \(jsonString)")
            } catch DecodingError.keyNotFound(let key, let context) {
                print("❌ Decoding error - key not found:")
                print("   Key: \(key)")
                print("   \(context.debugDescription)")
            } catch {
                print("❌ Failed to decode summary: \(error)")
                print("   JSON: \(jsonString)")
            }
            
        } else {
            self.currentSubtitle = text
        }
    }
    
    func clearSummaries() {
        summaries.removeAll()
        print("🗑️ Summaries cleared")
    }
    // 🔧 添加辅助方法（在类外部或作为扩展）
    private func parseCustomISO8601(_ dateString: String) -> Date? {
        // 格式: 2025-12-08T23:19:23.703617
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        
        if let date = formatter.date(from: dateString) {
            return date
        }
        
        // 尝试不带微秒的格式
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        return formatter.date(from: dateString)
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
    
    var formattedTrafficSaved: String {
        let savedKB = (skippedChunks * 3200) / 1024  // 假设每块 3200 bytes
        return "\(savedKB) KB"
    }
}
