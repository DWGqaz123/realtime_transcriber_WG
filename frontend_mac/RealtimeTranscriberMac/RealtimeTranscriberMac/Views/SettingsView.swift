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
    @State private var showSaveSuccess = false
    
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
                    .frame(maxWidth: 500)
                
                Text("Used for real-time speech transcription")
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
            
            Spacer()
        }
        .padding()
        .frame(minWidth: 600, minHeight: 400)
    }
    
    // MARK: - Methods
    
    private func saveAPIKeys() {
        // 保存到 UserDefaults
        UserDefaults.standard.set(openaiKey, forKey: "openai_api_key")
        UserDefaults.standard.set(elevenlabsKey, forKey: "elevenlabs_api_key")
        
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
        
        print("\n" + String(repeating: "=", count: 60))
        print("💾 STEP 1: Saving API Keys to Config File")
        print(String(repeating: "=", count: 60))
        print("OpenAI Key: \(openaiKeyValue.prefix(10))...(\(openaiKeyValue.count) chars)")
        print("ElevenLabs Key: \(elevenlabsKeyValue.prefix(10))...(\(elevenlabsKeyValue.count) chars)")
        
        let fileManager = FileManager.default
        
        // 获取应用支持目录
        guard let appSupportURL = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            print("❌ Failed to get application support directory")
            print(String(repeating: "=", count: 60) + "\n")
            return
        }
        
        // 创建 RealtimeTranscriber 目录
        let appDir = appSupportURL.appendingPathComponent("RealtimeTranscriber")
        let configFile = appDir.appendingPathComponent("api_keys.json")
        
        print("Config file path: \(configFile.path)")
        
        // 检查目录是否存在
        var isDirectory: ObjCBool = false
        let dirExists = fileManager.fileExists(atPath: appDir.path, isDirectory: &isDirectory)
        
        print("Directory status:")
        print("  - Path: \(appDir.path)")
        print("  - Exists: \(dirExists)")
        print("  - Is directory: \(isDirectory.boolValue)")
        
        // 创建目录（如果不存在）
        do {
            try fileManager.createDirectory(at: appDir, withIntermediateDirectories: true, attributes: nil)
            print("✅ Directory created/verified")
        } catch {
            print("❌ Failed to create directory: \(error)")
            print(String(repeating: "=", count: 60) + "\n")
            return
        }
        
        // 创建配置字典
        let config: [String: String] = [
            "openai_api_key": openaiKeyValue,
            "elevenlabs_api_key": elevenlabsKeyValue
        ]
        
        print("\nJSON content to save:")
        
        do {
            // 编码为 JSON
            let jsonData = try JSONEncoder().encode(config)
            
            // 转换为字符串查看
            if let jsonString = String(data: jsonData, encoding: .utf8) {
                print(jsonString)
            }
            
            // 写入文件
            try jsonData.write(to: configFile, options: .atomic)
            print("\n✅ File written successfully")
            
            // 验证文件已写入
            if fileManager.fileExists(atPath: configFile.path) {
                print("✅ File exists after write")
                
                // 读取文件大小
                if let attributes = try? fileManager.attributesOfItem(atPath: configFile.path),
                   let fileSize = attributes[.size] as? Int {
                    print("✅ File size: \(fileSize) bytes")
                }
                
                // 尝试读回内容验证
                if let readData = try? Data(contentsOf: configFile),
                   let readString = String(data: readData, encoding: .utf8) {
                    print("✅ File verification - content read back:")
                    print(readString)
                }
            } else {
                print("❌ File does not exist after write!")
            }
            
            print(String(repeating: "=", count: 60) + "\n")
            
        } catch {
            print("❌ Failed to save: \(error)")
            print("   Error type: \(type(of: error))")
            print("   Error description: \(error.localizedDescription)")
            print(String(repeating: "=", count: 60) + "\n")
        }
    }
}

// MARK: - Preview

#Preview {
    SettingsView()
}
