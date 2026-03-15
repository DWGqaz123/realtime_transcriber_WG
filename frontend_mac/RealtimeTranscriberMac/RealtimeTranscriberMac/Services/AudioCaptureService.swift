//
//  AudioCaptureService.swift
//  RealtimeTranscriberMac
//
//  Created by 董文光 on 2025/11/15.
import Foundation
import AVFoundation

final class AudioCaptureService {
    
    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    
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
    
    /// Request microphone permission
    func requestPermission(completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            completion(true)
            
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                DispatchQueue.main.async {
                    if granted {
                    } else {
                    }
                    completion(granted)
                }
            }
            
        case .denied, .restricted:
            completion(false)
            
        @unknown default:
            completion(false)
        }
    }
    
    /// Start capturing audio
    func start() throws {
        let inputNode = engine.inputNode
        let inputFormat = inputNode.inputFormat(forBus: 0)
        
        
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
        } else {
            self.converter = nil
        }
        
        let bufferSize: AVAudioFrameCount = 1024
        
        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) { [weak self] buffer, _ in
            guard let self = self else { return }

            // 计算音频电平
            let audioLevel = self.calculateAudioLevel(buffer: buffer)
            self.onAudioLevel?(audioLevel)

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
                self.onAudioData?(data)
            }
        }
        
        try engine.start()
    }
    
    /// Stop capturing audio
    func stop() {
        let inputNode = engine.inputNode
        inputNode.removeTap(onBus: 0)
        engine.stop()
        converter = nil
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
}
