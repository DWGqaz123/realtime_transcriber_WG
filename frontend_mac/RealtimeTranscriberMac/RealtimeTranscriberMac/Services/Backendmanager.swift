//
//  Backendmanager.swift
//  RealtimeTranscriberMac
//
//  Created by 董文光 on 2026/1/12.
//
// BackendManager.swift
import Foundation
import Combine

class BackendManager: ObservableObject {
    @Published var isRunning = false
    @Published var error: String?
    
    private var process: Process?
    private var outputPipe: Pipe?
    private var errorPipe: Pipe?
    
    init() {
        // 延迟启动，等 App 完全初始化
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            self.startBackend()
        }
    }
    
    func startBackend() {
        // 查找后端可执行文件
        guard let backendPath = Bundle.main.path(forResource: "backend", ofType: nil) else {
            let message = "❌ Backend executable not found in bundle"
            error = message
            print(message)
            return
        }
        
        print("🔍 Found backend at: \(backendPath)")
        
        // 检查文件是否存在
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: backendPath) else {
            error = "Backend file does not exist"
            return
        }
        
        // 设置可执行权限
        do {
            let attributes = try fileManager.attributesOfItem(atPath: backendPath)
            let permissions = attributes[.posixPermissions] as? Int ?? 0
            
            if permissions & 0o111 == 0 {
                print("⚠️ Backend not executable, setting permissions...")
                try fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: backendPath)
            }
        } catch {
            self.error = "Failed to set permissions: \(error.localizedDescription)"
            print("❌ Permission error: \(error)")
            return
        }
        
        // 创建进程
        process = Process()
        process?.executableURL = URL(fileURLWithPath: backendPath)
        
        // 设置工作目录（重要！）
        let supportDir = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("RealtimeTranscriber")
        try? fileManager.createDirectory(at: supportDir, withIntermediateDirectories: true)
        process?.currentDirectoryURL = supportDir
        
        // 设置环境变量
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        
        // 🔑 从 UserDefaults 读取 API Keys
        if let openaiKey = UserDefaults.standard.string(forKey: "openai_api_key"), !openaiKey.isEmpty {
            environment["OPENAI_API_KEY"] = openaiKey
            print("🔑 OpenAI API key loaded from settings")
        } else {
            print("⚠️ OpenAI API key not configured")
        }
        
        if let elevenlabsKey = UserDefaults.standard.string(forKey: "elevenlabs_api_key"), !elevenlabsKey.isEmpty {
            environment["ELEVENLABS_API_KEY"] = elevenlabsKey
            print("🔑 ElevenLabs API key loaded from settings")
        } else {
            print("⚠️ ElevenLabs API key not configured")
        }
        
        process?.environment = environment
        
        // 捕获标准输出
        outputPipe = Pipe()
        process?.standardOutput = outputPipe
        
        // 捕获错误输出
        errorPipe = Pipe()
        process?.standardError = errorPipe
        
        // 监听标准输出
        outputPipe?.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if let output = String(data: data, encoding: .utf8), !output.isEmpty {
                print("📤 Backend stdout: \(output)")
                self?.checkForStartupMessage(output)
            }
        }
        
        // 监听错误输出
        errorPipe?.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if let output = String(data: data, encoding: .utf8), !output.isEmpty {
                print("📤 Backend stderr: \(output)")
                self?.checkForStartupMessage(output)
            }
        }
        
        // 启动进程
        do {
            try process?.run()
            print("✅ Backend process started (PID: \(process?.processIdentifier ?? -1))")
            
            // 等待后端启动
            DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                self.checkServerHealth()
            }
        } catch {
            let message = "Failed to start backend: \(error.localizedDescription)"
            self.error = message
            print("❌ \(message)")
        }
        
    }
    
    private func checkForStartupMessage(_ output: String) {
        // 检测 Uvicorn 启动成功的消息
        if output.contains("Uvicorn running on") || output.contains("Application startup complete") {
            DispatchQueue.main.async {
                self.isRunning = true
                print("✅ Backend confirmed running")
            }
        }
    }
    
    func checkServerHealth() {
        guard let url = URL(string: "http://127.0.0.1:8000/health") else { return }
        
        var request = URLRequest(url: url)
        request.timeoutInterval = 5
        
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                    self?.isRunning = true
                    print("✅ Backend health check passed")
                } else {
                    print("⚠️ Backend health check failed: \(error?.localizedDescription ?? "Unknown error")")
                    // 再等待一会儿重试
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                        self?.checkServerHealth()
                    }
                }
            }
        }.resume()
    }
    
    func stopBackend() {
        print("🛑 Stopping backend...")
        process?.terminate()
        
        // 等待进程结束
        DispatchQueue.global().asyncAfter(deadline: .now() + 1) {
            if self.process?.isRunning == true {
                self.process?.interrupt()
            }
        }
        
        process = nil
        isRunning = false
        print("✅ Backend stopped")
    }
    
    deinit {
        stopBackend()
    }
}
