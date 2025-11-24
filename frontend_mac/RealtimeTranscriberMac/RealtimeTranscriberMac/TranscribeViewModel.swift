//
//  TranscribeViewModel.swift
//  RealtimeTranscriberMac
//
//  Created by 董文光 on 2025/11/14.
//
//
//  TranscribeViewModel.swift (With Audio Level)
//  RealtimeTranscriberMac
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

final class TranscribeViewModel: ObservableObject {
    @Published var isRecording: Bool = false
    @Published var currentSubtitle: String = ""
    @Published var fullTranscript: String = ""
    @Published var mode: RecordingMode = .lecture
    @Published var permissionStatus: String = "Unknown"
    @Published var showPermissionAlert: Bool = false
    
    // 音频电平相关
    @Published var audioLevel: Float = 0.0
    @Published var isDetectingSound: Bool = false
    @Published var recordingDuration: TimeInterval = 0.0
    
    // 静音检测和流量统计
    @Published var isSilent: Bool = false                   // 当前是否静音
    @Published var totalChunks: Int = 0                     // 总块数
    @Published var sentChunks: Int = 0                      // 发送块数
    @Published var skippedChunks: Int = 0                   // 跳过块数
    @Published var trafficSavedPercent: Int = 0             // 节省流量百分比

    private let client = TranscriptionClient()
    private let audioCapture = AudioCaptureService()
    private var recordingTimer: Timer?

    init() {
        // 处理来自后端的消息
        client.onMessage = { [weak self] text in
            DispatchQueue.main.async {
                self?.handleTranscriptMessage(text)
            }
        }

        // 发送音频数据
        audioCapture.onAudioData = { [weak self] data in
            self?.client.sendAudio(data)
        }
        
        // 监听音频电平
        audioCapture.onAudioLevel = { [weak self] level in
            DispatchQueue.main.async {
                self?.audioLevel = level
                self?.isDetectingSound = level > 0.01
            }
        }
        
        // 监听静音状态变化
        audioCapture.onSilenceStateChanged = { [weak self] isSilent in
            DispatchQueue.main.async {
                self?.isSilent = isSilent
            }
        }
        
        // 监听统计信息更新
        audioCapture.onStatisticsUpdated = { [weak self] total, sent, skipped in
            DispatchQueue.main.async {
                self?.totalChunks = total
                self?.sentChunks = sent
                self?.skippedChunks = skipped
                self?.trafficSavedPercent = total > 0 ? Int(Double(skipped) / Double(total) * 100) : 0
            }
        }
        
        checkMicrophonePermission()
    }
    
    /// 处理转录消息
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
            
        } else {
            self.currentSubtitle = text
        }
    }
    
    private func checkMicrophonePermission() {
        audioCapture.requestPermission { [weak self] granted in
            DispatchQueue.main.async {
                if granted {
                    self?.permissionStatus = "Granted ✅"
                } else {
                    self?.permissionStatus = "Denied ❌"
                    self?.showPermissionAlert = true
                }
            }
        }
    }

    func startRecording() {
        audioCapture.requestPermission { [weak self] granted in
            guard let self = self else { return }
            
            DispatchQueue.main.async {
                if !granted {
                    print("❌ Cannot start recording: Microphone permission denied")
                    self.permissionStatus = "Denied - Check System Settings ❌"
                    self.showPermissionAlert = true
                    return
                }
                
                // 开始录音
                self.isRecording = true
                self.currentSubtitle = ""
                self.fullTranscript = ""
                self.permissionStatus = "Recording... 🎤"
                self.recordingDuration = 0.0
                self.audioLevel = 0.0
                self.isDetectingSound = false
                self.isSilent = false
                
                // 重置统计
                self.totalChunks = 0
                self.sentChunks = 0
                self.skippedChunks = 0
                self.trafficSavedPercent = 0

                // 启动录音计时器
                self.recordingTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
                    self?.recordingDuration += 0.1
                }

                // 1. Connect WebSocket
                self.client.connect()

                // 2. Send mode configuration
                let modeString = self.mode.rawValue
                self.client.send(text: "MODE:\(modeString)")

                // 3. Start audio capture
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
        isRecording = false
        permissionStatus = "Stopped"
        
        // 停止计时器
        recordingTimer?.invalidate()
        recordingTimer = nil

        // Stop audio capture
        audioCapture.stop()

        // Send stop signal
        client.send(text: "STOP")

        // Close WebSocket
        client.disconnect()
        
        // 重置状态
        audioLevel = 0.0
        isDetectingSound = false
        isSilent = false
        currentSubtitle = ""
        
        print("🛑 Recording stopped (Duration: \(String(format: "%.1f", recordingDuration))s)")
    }
    
    /// 清空转录记录
    func clearTranscript() {
        fullTranscript = ""
        currentSubtitle = ""
    }
    
    /// 格式化录音时长
    var formattedDuration: String {
        let minutes = Int(recordingDuration) / 60
        let seconds = Int(recordingDuration) % 60
        let milliseconds = Int((recordingDuration.truncatingRemainder(dividingBy: 1)) * 10)
        return String(format: "%02d:%02d.%d", minutes, seconds, milliseconds)
    }
    
    /// 统计句子数量
    var sentenceCount: Int {
        guard !fullTranscript.isEmpty else { return 0 }
        return fullTranscript.components(separatedBy: "\n").filter { !$0.isEmpty }.count
    }
    
    /// 格式化流量节省
    var formattedTrafficSaved: String {
        let savedKB = (skippedChunks * 3200) / 1024  // 假设每块 3200 bytes
        return "\(savedKB) KB"
    }
}


