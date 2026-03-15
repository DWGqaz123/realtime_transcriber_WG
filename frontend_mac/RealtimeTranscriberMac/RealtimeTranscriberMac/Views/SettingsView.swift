//
//  SettingsView.swift
//  RealtimeTranscriberMac
//
//  Created by 董文光 on 2026/1/12.
//

import SwiftUI

struct SettingsView: View {
    // MARK: - State Variables
    
    @State private var openaiKey: String = UserDefaults.standard.string(forKey: "openai_api_key") ?? ""
    @State private var elevenlabsKey: String = UserDefaults.standard.string(forKey: "elevenlabs_api_key") ?? ""
    @State private var backendHost: String = UserDefaults.standard.string(forKey: "backend_host") ?? "127.0.0.1"
    @State private var backendPort: String = {
        let storedPort = UserDefaults.standard.integer(forKey: "backend_port")
        return String((1...65535).contains(storedPort) ? storedPort : 9123)
    }()
    @State private var summaryIntervalSeconds: String = {
        let storedValue = UserDefaults.standard.integer(forKey: "summary_interval_seconds")
        return String(storedValue > 0 ? storedValue : 30)
    }()
    @State private var showSaveSuccess = false
    @State private var validationMessage: String?
    
    // MARK: - Body
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Settings")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            Divider()
            
            // OpenAI API Key
            VStack(alignment: .leading, spacing: 8) {
                Text("OpenAI API Key")
                    .font(.headline)
                
                SecureField("sk-proj-...", text: $openaiKey)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 500)
                
                Text("Used for generating summaries with GPT-4")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            // ElevenLabs API Key
            VStack(alignment: .leading, spacing: 8) {
                Text("ElevenLabs API Key")
                    .font(.headline)
                
                SecureField("sk_...", text: $elevenlabsKey)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 480)
                
                Text("Used for real-time speech transcription")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Backend Host")
                    .font(.headline)

                TextField("127.0.0.1", text: $backendHost)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 260)

                Text("Local backend address used by the app.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Backend Port")
                    .font(.headline)

                TextField("9123", text: $backendPort)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 180)

                Text("Change this if the default port is already occupied.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Auto Summary Interval")
                    .font(.headline)

                TextField("30", text: $summaryIntervalSeconds)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 180)

                Text("Main auto-summary interval in seconds. Frontend countdown follows the backend value.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            // Save Button
            HStack(spacing: 12) {
                Button("Save & Restart Backend") {
                    saveAPIKeys()
                }
                .buttonStyle(.borderedProminent)
                
                if showSaveSuccess {
                    HStack(spacing: 4) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                        Text("Saved! Backend restarting...")
                            .foregroundColor(.green)
                    }
                    .transition(.opacity)
                }
            }

            if let validationMessage {
                Text(validationMessage)
                    .font(.caption)
                    .foregroundColor(.red)
            }
            
            Spacer()
        }
        .padding()
        .frame(minWidth: 600, minHeight: 400)
    }
    
    // MARK: - Methods
    
    private func saveAPIKeys() {
        let trimmedHost = backendHost.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedHost.isEmpty else {
            validationMessage = "Backend host cannot be empty."
            return
        }

        guard let port = Int(backendPort), (1...65535).contains(port) else {
            validationMessage = "Backend port must be between 1 and 65535."
            return
        }

        guard let summaryInterval = Int(summaryIntervalSeconds), summaryInterval > 0 else {
            validationMessage = "Auto summary interval must be a positive number."
            return
        }

        validationMessage = nil

        // 保存到 UserDefaults
        UserDefaults.standard.set(openaiKey, forKey: "openai_api_key")
        UserDefaults.standard.set(elevenlabsKey, forKey: "elevenlabs_api_key")
        UserDefaults.standard.set(trimmedHost, forKey: "backend_host")
        UserDefaults.standard.set(port, forKey: "backend_port")
        UserDefaults.standard.set(summaryInterval, forKey: "summary_interval_seconds")
        
        // 保存到配置文件
        saveAPIKeysToConfigFile()
        
        // 显示成功提示
        withAnimation {
            showSaveSuccess = true
        }
        
        // 重启后端
        NotificationCenter.default.post(name: .restartBackend, object: nil)
        
        // 隐藏成功提示
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            withAnimation {
                showSaveSuccess = false
            }
        }
    }
    
    private func saveAPIKeysToConfigFile() {
        // 🔧 在方法开头捕获 @State 变量的值
        let openaiKeyValue = openaiKey
        let elevenlabsKeyValue = elevenlabsKey
        
        
        let fileManager = FileManager.default
        
        // 获取应用支持目录
        guard let appSupportURL = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            return
        }
        
        // 创建 RealtimeTranscriber 目录
        let appDir = appSupportURL.appendingPathComponent("RealtimeTranscriber")
        let configFile = appDir.appendingPathComponent("api_keys.json")
        
        
        // 检查目录是否存在
        var isDirectory: ObjCBool = false
        let dirExists = fileManager.fileExists(atPath: appDir.path, isDirectory: &isDirectory)
        
        
        // 创建目录（如果不存在）
        do {
            try fileManager.createDirectory(at: appDir, withIntermediateDirectories: true, attributes: nil)
        } catch {
            return
        }
        
        // 创建配置字典
        let config: [String: String] = [
            "openai_api_key": openaiKeyValue,
            "elevenlabs_api_key": elevenlabsKeyValue
        ]
        
        
        do {
            // 编码为 JSON
            let jsonData = try JSONEncoder().encode(config)
            
            // 转换为字符串查看
            if let jsonString = String(data: jsonData, encoding: .utf8) {
            }
            
            // 写入文件
            try jsonData.write(to: configFile, options: .atomic)
            
            // 验证文件已写入
            if fileManager.fileExists(atPath: configFile.path) {
                
                // 读取文件大小
                if let attributes = try? fileManager.attributesOfItem(atPath: configFile.path),
                   let fileSize = attributes[.size] as? Int {
                }
                
                // 尝试读回内容验证
                if let readData = try? Data(contentsOf: configFile),
                   let readString = String(data: readData, encoding: .utf8) {
                }
            } else {
            }
            
            
        } catch {
        }
    }
}

// MARK: - Preview

#Preview {
    SettingsView()
}
