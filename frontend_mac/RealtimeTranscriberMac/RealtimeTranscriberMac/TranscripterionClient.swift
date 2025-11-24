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
    private var isConnected: Bool = false
    private var shouldReconnect: Bool = false  // 🔧 新增：是否应该重连
    private var currentMode: String = ""  // 🔧 新增：保存当前模式
    private var reconnectAttempts: Int = 0  // 🔧 新增：重连尝试次数
    private let maxReconnectAttempts: Int = 5  // 🔧 新增：最大重连次数
    
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
    
    func connect() {
        guard !isConnected else {
            print("[TranscriptionClient] Already connected")
            return
        }
        
        shouldReconnect = true  // 🔧 新增：标记应该保持连接
        reconnectAttempts = 0  // 🔧 新增：重置重连计数
        
        print("[TranscriptionClient] Connecting to \(serverURL.absoluteString)")
        
        guard let session = urlSession else {
            print("[TranscriptionClient] ❌ URLSession not initialized")
            return
        }
        
        webSocketTask = session.webSocketTask(with: serverURL)
        webSocketTask?.resume()
        isConnected = true
        
        print("[TranscriptionClient] WebSocket: connect() called, task resumed")
        startReceiving()
    }
    
    // 🔧 新增：自动重连方法
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
            self.isConnected = false
            
            // 建立新连接
            guard let session = self.urlSession else { return }
            self.webSocketTask = session.webSocketTask(with: self.serverURL)
            self.webSocketTask?.resume()
            self.isConnected = true
            
            print("[TranscriptionClient] ✅ Reconnected to WebSocket")
            
            // 开始接收消息
            self.startReceiving()
            
            // 重新发送模式配置
            if !self.currentMode.isEmpty {
                print("[TranscriptionClient] Resending mode: \(self.currentMode)")
                self.send(text: "MODE:\(self.currentMode)")
            }
        }
    }
    
    func disconnect() {
        shouldReconnect = false  // 🔧 新增：标记不应重连
        isConnected = false
        
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
        
        print("[TranscriptionClient] Disconnected")
    }
    
    func send(text: String) {
        guard isConnected, let task = webSocketTask else {
            print("[TranscriptionClient] ❌ Cannot send text, not connected")
            return
        }
        
        // 🔧 新增：保存模式信息
        if text.uppercased().hasPrefix("MODE:") {
            currentMode = text.split(separator: ":")[1].trimmingCharacters(in: .whitespaces)
        }
        
        let message = URLSessionWebSocketTask.Message.string(text)
        task.send(message) { error in
            if let error = error {
                print("[TranscriptionClient] ❌ Failed to send text: \(error.localizedDescription)")
                // 🔧 新增：发送失败时触发重连
                if !self.isConnected {
                    self.reconnect()
                }
            } else {
                print("[TranscriptionClient] Text sent: \(text)")
            }
        }
    }
    
    func sendAudio(_ data: Data) {
        guard isConnected, let task = webSocketTask else {
            print("[TranscriptionClient] ❌ Cannot send audio, not connected")
            return
        }
        
        let message = URLSessionWebSocketTask.Message.data(data)
        task.send(message) { error in
            if let error = error {
                print("[TranscriptionClient] ❌ Failed to send audio chunk: \(error.localizedDescription)")
                // 🔧 新增：发送失败时触发重连
                DispatchQueue.main.async {
                    if !self.isConnected {
                        self.reconnect()
                    }
                }
            }
        }
    }
    
    private func startReceiving() {
        webSocketTask?.receive { [weak self] result in
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
                
                // 继续接收
                if self.isConnected {
                    self.startReceiving()
                }
                
            case .failure(let error):
                print("[TranscriptionClient] ❌ WebSocket receive error: \(error.localizedDescription)")
                self.isConnected = false
                
                // 🔧 新增：接收错误时自动重连
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
        didOpenWithProtocol protocol: String?
    ) {
        print("[TranscriptionClient] ✅ WebSocket connection opened")
        isConnected = true
        reconnectAttempts = 0  // 🔧 新增：重置重连计数
    }
    
    func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didCloseWith closeCode: URLSessionWebSocketTask.CloseCode,
        reason: Data?
    ) {
        print("[TranscriptionClient] ⚠️ WebSocket closed with code: \(closeCode.rawValue)")
        isConnected = false
        
        // 🔧 新增：连接关闭时自动重连
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
            
            // 🔧 新增：任务错误时自动重连
            if shouldReconnect {
                reconnect()
            }
        } else {
            print("[TranscriptionClient] WebSocket task completed normally")
        }
    }
}
