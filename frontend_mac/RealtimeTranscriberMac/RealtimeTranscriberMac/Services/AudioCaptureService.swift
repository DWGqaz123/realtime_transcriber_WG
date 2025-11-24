//
//  AudioCaptureService.swift
//  RealtimeTranscriberMac
//
//  Created by 董文光 on 2025/11/15.
//
//
//
//  AudioCaptureService.swift (Debug Enhanced Version)
//  验证音频转换是否真的成功
//
import Foundation
import AVFoundation

final class AudioCaptureService {
    
    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private var chunkCount = 0
    
    // 静音检测配置
    private let silenceThreshold: Float = 0.5          // 静音阈值
    private let silenceFramesThreshold: Int = 500       // 连续静音帧数阈值
    private var silentFramesCount: Int = 0             // 当前连续静音帧计数
    private var isSilent: Bool = false                  // 是否处于静音状态
    
    // 统计信息
    private var totalChunks: Int = 0                    // 总音频块数
    private var sentChunks: Int = 0                     // 实际发送的音频块数
    private var skippedChunks: Int = 0                  // 跳过的静音块数
    
    private lazy var targetFormat: AVAudioFormat = {
        return AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        )!
    }()
    
    var onAudioData: ((Data) -> Void)?
    var onAudioLevel: ((Float) -> Void)?
    var onSilenceStateChanged: ((Bool) -> Void)?        // 新增：静音状态变化回调
    var onStatisticsUpdated: ((Int, Int, Int) -> Void)? // 新增：统计信息回调 (total, sent, skipped)
    
    /// Request microphone permission
    func requestPermission(completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            print("🎤 Microphone: Already authorized")
            completion(true)
            
        case .notDetermined:
            print("🎤 Microphone: Requesting permission...")
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                DispatchQueue.main.async {
                    if granted {
                        print("✅ Microphone: Permission granted")
                    } else {
                        print("❌ Microphone: Permission denied")
                    }
                    completion(granted)
                }
            }
            
        case .denied, .restricted:
            print("❌ Microphone: Permission denied or restricted")
            completion(false)
            
        @unknown default:
            completion(false)
        }
    }
    
    /// Start capturing audio
    func start() throws {
        let inputNode = engine.inputNode
        let inputFormat = inputNode.inputFormat(forBus: 0)
        
        print("\n" + String(repeating: "=", count: 50))
        print("🎤 Audio Capture Starting (with Silence Detection)")
        print(String(repeating: "=", count: 50))
        print("Input: \(inputFormat.sampleRate)Hz, \(inputFormat.channelCount)ch, \(inputFormat.commonFormat)")
        print("Target: \(targetFormat.sampleRate)Hz, \(targetFormat.channelCount)ch, \(targetFormat.commonFormat)")
        print("Silence Threshold: \(silenceThreshold)")
        print("Silence Frames: \(silenceFramesThreshold) frames (~\(Double(silenceFramesThreshold) * 0.021)s)")
        
        // Create converter if needed
        if inputFormat.sampleRate != targetFormat.sampleRate ||
           inputFormat.channelCount != targetFormat.channelCount ||
           inputFormat.commonFormat != targetFormat.commonFormat {
            
            guard let audioConverter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
                throw NSError(
                    domain: "AudioCaptureService",
                    code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to create audio converter"]
                )
            }
            self.converter = audioConverter
            print("✅ Audio converter created")
        } else {
            self.converter = nil
            print("✅ No conversion needed")
        }
        
        // Reset statistics
        totalChunks = 0
        sentChunks = 0
        skippedChunks = 0
        silentFramesCount = 0
        isSilent = false
        
        let bufferSize: AVAudioFrameCount = 1024
        
        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) { [weak self] buffer, _ in
            guard let self = self else { return }
            
            self.chunkCount += 1
            self.totalChunks += 1
            
            // 计算音频电平
            let audioLevel = self.calculateAudioLevel(buffer: buffer)
            self.onAudioLevel?(audioLevel)
//            
//            // 静音检测
//            let isCurrentlySilent = audioLevel < self.silenceThreshold
//            
//            if isCurrentlySilent {
//                self.silentFramesCount += 1
//            } else {
//                self.silentFramesCount = 0
//            }
//            
//            // 检查是否进入/退出静音状态
//            let wasInSilence = self.isSilent
//            self.isSilent = self.silentFramesCount >= self.silenceFramesThreshold
//            
//            // 状态变化时通知
//            if wasInSilence != self.isSilent {
//                DispatchQueue.main.async {
//                    self.onSilenceStateChanged?(self.isSilent)
//                }
//                
//                if self.isSilent {
//                    print("🔇 Entered silence state (no audio will be sent)")
//                } else {
//                    print("🔊 Exited silence state (resuming audio transmission)")
//                }
//            }
//            
//            // 如果处于静音状态，跳过发送
//            if self.isSilent {
//                self.skippedChunks += 1
//                
//                // 每 50 个跳过的块打印一次统计
//                if self.skippedChunks % 50 == 0 {
//                    let savedPercent = Int(Double(self.skippedChunks) / Double(self.totalChunks) * 100)
//                    print("📊 Traffic saved: \(self.skippedChunks)/\(self.totalChunks) chunks (\(savedPercent)%)")
//                }
//                
//                // 更新统计
//                DispatchQueue.main.async {
//                    self.onStatisticsUpdated?(self.totalChunks, self.sentChunks, self.skippedChunks)
//                }
//                
////                return  // 跳过发送
//            }
            
            // 前 3 个块打印调试信息
            if self.chunkCount <= 3 {
                print("✅ Chunk #\(self.chunkCount): Level=\(String(format: "%.3f", audioLevel)) → Sending")
            }
            
            // Convert if needed
            let processedBuffer: AVAudioPCMBuffer
            if let converter = self.converter {
                if let converted = self.convertBuffer(buffer, using: converter) {
                    processedBuffer = converted
                } else {
                    return
                }
            } else {
                processedBuffer = buffer
            }
            
            // Convert to Data and send
            if let data = Self.bufferToData(buffer: processedBuffer) {
                self.sentChunks += 1
                self.onAudioData?(data)
                
                // 更新统计
                DispatchQueue.main.async {
                    self.onStatisticsUpdated?(self.totalChunks, self.sentChunks, self.skippedChunks)
                }
            }
        }
        
        try engine.start()
        print("✅ Audio engine started")
        print(String(repeating: "=", count: 50) + "\n")
    }
    
    /// Stop capturing audio
    func stop() {
        let inputNode = engine.inputNode
        inputNode.removeTap(onBus: 0)
        engine.stop()
        converter = nil
        chunkCount = 0
        
        // 打印最终统计
        if totalChunks > 0 {
            let savedPercent = Int(Double(skippedChunks) / Double(totalChunks) * 100)
            let savedData = (skippedChunks * 3200) / 1024  // 假设每块 3200 bytes
            
            print("\n" + String(repeating: "=", count: 50))
            print("📊 Final Statistics")
            print(String(repeating: "=", count: 50))
            print("Total chunks: \(totalChunks)")
            print("Sent chunks: \(sentChunks)")
            print("Skipped chunks: \(skippedChunks) (\(savedPercent)%)")
            print("Data saved: ~\(savedData) KB")
            print(String(repeating: "=", count: 50))
        }
        
        print("🛑 Audio capture stopped")
    }
    
    /// Calculate audio level (RMS)
    private func calculateAudioLevel(buffer: AVAudioPCMBuffer) -> Float {
        // Float32 format
        if buffer.format.commonFormat == .pcmFormatFloat32,
           let channelData = buffer.floatChannelData?[0] {
            
            var sum: Float = 0
            let frameCount = Int(buffer.frameLength)
            
            for i in 0..<frameCount {
                let sample = channelData[i]
                sum += sample * sample
            }
            
            let rms = sqrt(sum / Float(frameCount))
            return rms
        }
        
        // Int16 format
        if buffer.format.commonFormat == .pcmFormatInt16,
           let channelData = buffer.int16ChannelData?[0] {
            
            var sum: Int64 = 0
            let frameCount = Int(buffer.frameLength)
            
            for i in 0..<frameCount {
                let sample = Int64(channelData[i])
                sum += sample * sample
            }
            
            let rms = sqrt(Double(sum) / Double(frameCount))
            return Float(rms / 32767.0)
        }
        
        return 0.0
    }
    
    /// Convert audio buffer to target format
    private func convertBuffer(_ buffer: AVAudioPCMBuffer, using converter: AVAudioConverter) -> AVAudioPCMBuffer? {
        let ratio = targetFormat.sampleRate / buffer.format.sampleRate
        let outputFrameCapacity = AVAudioFrameCount(ceil(Double(buffer.frameLength) * ratio))
        
        guard let convertedBuffer = AVAudioPCMBuffer(
            pcmFormat: targetFormat,
            frameCapacity: outputFrameCapacity
        ) else {
            return nil
        }
        
        var error: NSError?
        var inputBufferConsumed = false
        
        let inputBlock: AVAudioConverterInputBlock = { inNumPackets, outStatus in
            if inputBufferConsumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            
            inputBufferConsumed = true
            outStatus.pointee = .haveData
            return buffer
        }
        
        _ = converter.convert(to: convertedBuffer, error: &error, withInputFrom: inputBlock)
        
        if let error = error {
            print("❌ Conversion error: \(error)")
            return nil
        }
        
        return convertedBuffer.frameLength > 0 ? convertedBuffer : nil
    }
    
    /// Convert AVAudioPCMBuffer to Data
    private static func bufferToData(buffer: AVAudioPCMBuffer) -> Data? {
        let audioBufferList = buffer.audioBufferList.pointee
        var audioBuffer = audioBufferList.mBuffers
        
        guard let mData = audioBuffer.mData else {
            return nil
        }
        
        let dataSize = Int(audioBuffer.mDataByteSize)
        return dataSize > 0 ? Data(bytes: mData, count: dataSize) : nil
    }
    
    /// Get traffic statistics
    func getStatistics() -> (total: Int, sent: Int, skipped: Int, savedPercent: Int) {
        let savedPercent = totalChunks > 0 ? Int(Double(skippedChunks) / Double(totalChunks) * 100) : 0
        return (totalChunks, sentChunks, skippedChunks, savedPercent)
    }
}

