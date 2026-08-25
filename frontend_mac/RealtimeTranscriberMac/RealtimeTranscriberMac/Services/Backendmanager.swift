//
//  Backendmanager.swift
//  RealtimeTranscriberMac
//
//  Created by 董文光 on 2026/1/12.
//
// BackendManager.swift
import Foundation
import Combine
import Darwin

class BackendManager: ObservableObject {
    @Published var isRunning = false
    @Published var error: String?
    
    private var process: Process?
    private var outputPipe: Pipe?
    private var errorPipe: Pipe?
    private var healthCheckAttempts = 0
    private let maxHealthCheckAttempts = 10

    private var backendHost: String { AppConfig.backendHost }
    private var backendPort: Int { AppConfig.backendPort }
    
    init() {
        // 延迟启动，等 App 完全初始化
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            self.startBackend()
        }
    }
    
    /// 启动后端。`reuseExisting` 为 true 时，若端口上已有健康的后端就直接复用，
    /// 避免第二个实例抢占端口后启动失败（重启配置时必须传 false）。
    func startBackend(reuseExisting: Bool = true) {
        guard reuseExisting else {
            launchBackendProcess()
            return
        }

        probeExistingBackend { [weak self] isAlive in
            guard let self = self else { return }
            if isAlive {
                self.isRunning = true
                self.error = nil
                self.healthCheckAttempts = 0
                return
            }
            self.launchBackendProcess()
        }
    }

    /// 短超时探测端口上是否已经有一个可用后端
    private func probeExistingBackend(completion: @escaping (Bool) -> Void) {
        var request = URLRequest(url: AppConfig.healthURL)
        request.timeoutInterval = 1.5
        request.cachePolicy = .reloadIgnoringLocalCacheData

        URLSession.shared.dataTask(with: request) { _, response, _ in
            let isAlive = (response as? HTTPURLResponse)?.statusCode == 200
            DispatchQueue.main.async { completion(isAlive) }
        }.resume()
    }

    private func launchBackendProcess() {
        // 查找后端可执行文件
        guard let backendPath = Bundle.main.path(forResource: "backend", ofType: nil) else {
            let message = "❌ Backend executable not found in bundle"
            error = message
            return
        }
        
        
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
                try fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: backendPath)
            }
        } catch {
            self.error = "Failed to set permissions: \(error.localizedDescription)"
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
        environment["HOST"] = backendHost
        environment["PORT"] = String(backendPort)
        
        // 🔑 从 UserDefaults 读取 API Keys
        if let openaiKey = UserDefaults.standard.string(forKey: "openai_api_key"), !openaiKey.isEmpty {
            environment["OPENAI_API_KEY"] = openaiKey
        } else {
        }
        
        if let elevenlabsKey = UserDefaults.standard.string(forKey: "elevenlabs_api_key"), !elevenlabsKey.isEmpty {
            environment["ELEVENLABS_API_KEY"] = elevenlabsKey
        } else {
        }

        if let language = UserDefaults.standard.string(forKey: "transcription_language") {
            environment["TRANSCRIPTION_LANGUAGE"] = language
        }

        let summaryInterval = UserDefaults.standard.integer(forKey: "summary_interval_seconds")
        if summaryInterval > 0 {
            environment["SUMMARY_INTERVAL_SECONDS"] = String(summaryInterval)
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
                self?.checkForStartupMessage(output)
            }
        }
        
        // 监听错误输出
        errorPipe?.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if let output = String(data: data, encoding: .utf8), !output.isEmpty {
                self?.checkForStartupMessage(output)
            }
        }
        
        // 启动进程
        do {
            try process?.run()
            
            // 等待后端启动
            DispatchQueue.main.asyncAfter(deadline: .now() + 10) {
                self.checkServerHealth()
            }
        } catch {
            let message = "Failed to start backend: \(error.localizedDescription)"
            self.error = message
        }
        
    }
    
    private func checkForStartupMessage(_ output: String) {
        // 检测 Uvicorn 启动成功的消息
        if output.contains("Uvicorn running on") || output.contains("Application startup complete") {
            DispatchQueue.main.async {
                self.isRunning = true
            }
        }
    }
    
    func checkServerHealth() {
        guard healthCheckAttempts < maxHealthCheckAttempts else {
            DispatchQueue.main.async {
                self.error = "Backend failed to start after \(self.maxHealthCheckAttempts) attempts"
            }
            return
        }
        
        healthCheckAttempts += 1
        let url = AppConfig.healthURL
        
        
        var request = URLRequest(url: url)
        request.timeoutInterval = 10  // ← 增加超时时间到 10 秒
        
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            if error != nil {
                
                // 不要立即放弃，多重试几次
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                    self?.checkServerHealth()
                }
                return
            }
            
            if let httpResponse = response as? HTTPURLResponse {
                
                if httpResponse.statusCode == 200 {
                    DispatchQueue.main.async {
                        self?.isRunning = true
                        self?.error = nil
                        self?.healthCheckAttempts = 0
                    }
                } else {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                        self?.checkServerHealth()
                    }
                }
            }
        }.resume()
    }
    
    /// 停止后端并**等待它真正退出**。调用方（退出 App / 重启配置）依赖这一点：
    /// 残留进程会占住端口，让下一次启动连不上。
    func stopBackend() {
        // Clear pipe handlers before stopping to prevent retain cycles
        outputPipe?.fileHandleForReading.readabilityHandler = nil
        errorPipe?.fileHandleForReading.readabilityHandler = nil

        if let proc = process, proc.isRunning {
            BackendManager.terminateAndWait(proc)
        }

        process = nil
        outputPipe = nil
        errorPipe = nil
        isRunning = false
        healthCheckAttempts = 0
    }

    /// SIGTERM，最多等 `timeout` 秒，仍未退出则 SIGKILL。
    /// PyInstaller 的 onefile bootloader 会把信号转发给它派生的子进程。
    private static func terminateAndWait(_ proc: Process, timeout: TimeInterval = 3.0) {
        let pid = proc.processIdentifier
        proc.terminate()

        let deadline = Date().addingTimeInterval(timeout)
        while proc.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }

        if proc.isRunning {
            kill(pid, SIGKILL)
            proc.waitUntilExit()
        }
    }

    deinit {
        // Inline cleanup to avoid using self in async context during deallocation
        outputPipe?.fileHandleForReading.readabilityHandler = nil
        errorPipe?.fileHandleForReading.readabilityHandler = nil
        if let proc = process, proc.isRunning {
            BackendManager.terminateAndWait(proc, timeout: 1.0)
        }
    }
}
