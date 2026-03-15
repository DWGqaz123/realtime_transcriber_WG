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
    @State private var transcriptionLanguage: String = UserDefaults.standard.string(forKey: "transcription_language") ?? "en"
    @State private var showSaveSuccess = false
    @State private var validationMessage: String?

    private let languageOptions: [(code: String, label: String)] = [
        ("", "Auto-detect"),
        ("en", "English"),
        ("zh", "Chinese (中文)"),
        ("ja", "Japanese (日本語)"),
        ("fr", "French (Français)"),
        ("de", "German (Deutsch)"),
        ("es", "Spanish (Español)"),
    ]
    
    // MARK: - Body
    
    var body: some View {
        ScrollView {
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
                Text("Transcription & Summary Language")
                    .font(.headline)

                Picker("", selection: $transcriptionLanguage) {
                    ForEach(languageOptions, id: \.code) { option in
                        Text(option.label).tag(option.code)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: 260)

                Text("Locks ElevenLabs speech recognition to the selected language. Summaries will also be generated in this language. Defaults to English.")
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
            
        }
        .padding()
        .frame(minWidth: 600)
        } // ScrollView
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
        UserDefaults.standard.set(transcriptionLanguage, forKey: "transcription_language")
        
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
        let fileManager = FileManager.default
        guard let appSupportURL = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else { return }

        let appDir = appSupportURL.appendingPathComponent("RealtimeTranscriber")
        let configFile = appDir.appendingPathComponent("api_keys.json")

        do {
            try fileManager.createDirectory(at: appDir, withIntermediateDirectories: true, attributes: nil)
            let config: [String: String] = [
                "openai_api_key": openaiKey,
                "elevenlabs_api_key": elevenlabsKey,
                "transcription_language": transcriptionLanguage,
            ]
            try JSONEncoder().encode(config).write(to: configFile, options: .atomic)
        } catch {
            // 写入失败时后端将沿用旧配置
        }
    }
}

// MARK: - Preview

#Preview {
    SettingsView()
}
