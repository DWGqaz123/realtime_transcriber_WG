//
//  TranscripterionClient.swift
//  RealtimeTranscriberMac
//
//  Created by 董文光 on 2025/11/14.
//

import Foundation

final class TranscriptionClient: NSObject {
    private var webSocketTask: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    private var serverURL: URL
    private var isConnected: Bool = false          // 当前 WS 是否已经完成握手并且打开
    private var shouldReconnect: Bool = false      // 是否需要自动重连
    private var currentMode: String = ""           // 保存当前模式 (lecture/discussion)
    private var reconnectAttempts: Int = 0         // 重连次数
    private let maxReconnectAttempts: Int = 5      // 最大重连次数
    
    /// 文本消息回调（服务端推来的字幕 / 提示）
    var onMessage: ((String) -> Void)?
    
    init(serverURL: URL = URL(string: "ws://127.0.0.1:8000/ws/transcribe")!) {
        self.serverURL = serverURL
        super.init()
        
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 60.0
        configuration.timeoutIntervalForResource = 300.0
        
        self.urlSession = URLSession(
            configuration: configuration,
            delegate: self,
            delegateQueue: OperationQueue()
        )
    }
    
    /// 建立 WebSocket 连接
    func connect() {
        guard !isConnected else {
            print("[TranscriptionClient] Already connected")
            return
        }
        
        shouldReconnect = true
        reconnectAttempts = 0
        
        print("[TranscriptionClient] Connecting to \(serverURL.absoluteString)")
        
        guard let session = urlSession else {
            print("[TranscriptionClient] ❌ URLSession not initialized")
            return
        }
        
        webSocketTask = session.webSocketTask(with: serverURL)
        webSocketTask?.resume()
        
        print("[TranscriptionClient] WebSocket: connect() called, task resumed")
        // 注意：此时握手还没完成，isConnected 仍然是 false
        startReceiving()  // 等待 didOpenWithProtocol 回调来设置 isConnected = true
    }
    
    /// 自动重连逻辑（带指数退避）
    private func reconnect() {
        guard shouldReconnect else {
            print("[TranscriptionClient] Reconnect skipped (shouldReconnect = false)")
            return
        }
        
        guard reconnectAttempts < maxReconnectAttempts else {
            print("[TranscriptionClient] ❌ Max reconnect attempts reached (\(maxReconnectAttempts))")
            return
        }
        
        reconnectAttempts += 1
        let delay = min(Double(reconnectAttempts) * 2.0, 10.0)  // 指数退避，最多 10 秒
        
        print("[TranscriptionClient] 🔄 Reconnecting in \(delay)s (attempt \(reconnectAttempts)/\(maxReconnectAttempts))...")
        
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self = self else { return }
            
            // 关闭旧连接
            self.webSocketTask?.cancel(with: .goingAway, reason: nil)
            self.webSocketTask = nil
            self.isConnected = false
            
            guard let session = self.urlSession else { return }
            
            // 建立新连接
            self.webSocketTask = session.webSocketTask(with: self.serverURL)
            self.webSocketTask?.resume()
            
            print("[TranscriptionClient] ✅ Reconnected to WebSocket (handshake pending)")
            
            // 重新开始接收消息
            self.startReceiving()
            
            // ⚠️ 不要在这里把 isConnected 设为 true，
            // 只在 didOpenWithProtocol 里设，代表握手真正成功。
            
            // 重新发送模式配置（如果已经有）
            if !self.currentMode.isEmpty {
                print("[TranscriptionClient] Resending mode after reconnect: \(self.currentMode)")
                self.send(text: "MODE:\(self.currentMode)")
            }
        }
    }
    
    /// 主动断开连接
    func disconnect() {
        shouldReconnect = false
        isConnected = false
        
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
        
        print("[TranscriptionClient] Disconnected")
    }
    
    /// 发送文本消息（比如 MODE:lecture / STOP 等）
    func send(text: String) {
        // ✅ 只要任务存在就允许尝试发送，不强依赖 isConnected，
        // 因为 URLSession 会在握手 OK 后真正发出去；握手失败则在回调里出错。
        guard let task = webSocketTask else {
            print("[TranscriptionClient] ❌ Cannot send text, webSocketTask is nil")
            return
        }
        
        // 保存模式信息（用于断线重连后恢复）
        if text.uppercased().hasPrefix("MODE:") {
            currentMode = text.split(separator: ":")[1].trimmingCharacters(in: .whitespaces)
        }
        
        let message = URLSessionWebSocketTask.Message.string(text)
        task.send(message) { [weak self] error in
            guard let self = self else { return }
            
            if let error = error {
                print("[TranscriptionClient] ❌ Failed to send text: \(error.localizedDescription)")
                self.isConnected = false
                if self.shouldReconnect {
                    self.reconnect()
                }
            } else {
                print("[TranscriptionClient] Text sent: \(text)")
            }
        }
    }
    
    /// 发送音频二进制数据
    func sendAudio(_ data: Data) {
        guard let task = webSocketTask else {
            print("[TranscriptionClient] ❌ Cannot send audio, webSocketTask is nil")
            return
        }
        
        let message = URLSessionWebSocketTask.Message.data(data)
        task.send(message) { [weak self] error in
            guard let self = self else { return }
            
            if let error = error {
                print("[TranscriptionClient] ❌ Failed to send audio chunk: \(error.localizedDescription)")
                self.isConnected = false
                if self.shouldReconnect {
                    self.reconnect()
                }
            }
        }
    }
    
    /// 循环接收来自服务器的消息
    private func startReceiving() {
        // 如果当前没有 task，就不再递归
        guard let task = webSocketTask else {
            print("[TranscriptionClient] ⚠️ startReceiving called but webSocketTask is nil")
            return
        }
        
        task.receive { [weak self] result in
            guard let self = self else { return }
            
            switch result {
            case .success(let message):
                switch message {
                case .string(let text):
                    print("[TranscriptionClient] WebSocket received string: \(text)")
                    DispatchQueue.main.async {
                        self.onMessage?(text)
                    }
                    
                case .data(let data):
                    print("[TranscriptionClient] WebSocket received data: \(data.count) bytes")
                    
                @unknown default:
                    print("[TranscriptionClient] ⚠️ Unknown message type")
                }
                
                // ✅ 无论 isConnected 与否，只要 task 还在，就继续接收下一条。
                self.startReceiving()
                
            case .failure(let error):
                print("[TranscriptionClient] ❌ WebSocket receive error: \(error.localizedDescription)")
                self.isConnected = false
                
                if self.shouldReconnect {
                    self.reconnect()
                }
            }
        }
    }
}

// MARK: - URLSessionWebSocketDelegate

extension TranscriptionClient: URLSessionWebSocketDelegate {
    func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didOpenWithProtocol `protocol`: String?
    ) {
        print("[TranscriptionClient] ✅ WebSocket connection opened")
        isConnected = true
        reconnectAttempts = 0
    }
    
    func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didCloseWith closeCode: URLSessionWebSocketTask.CloseCode,
        reason: Data?
    ) {
        print("[TranscriptionClient] ⚠️ WebSocket closed with code: \(closeCode.rawValue)")
        isConnected = false
        
        if shouldReconnect {
            reconnect()
        }
    }
    
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        if let error = error {
            print("[TranscriptionClient] ❌ WebSocket task completed with error: \(error.localizedDescription)")
            isConnected = false
            
            if shouldReconnect {
                reconnect()
            }
        } else {
            print("[TranscriptionClient] WebSocket task completed normally")
        }
    }
}
