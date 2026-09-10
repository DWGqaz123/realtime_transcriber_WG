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
    @State private var secondaryLanguages: Set<String> = Set(
        UserDefaults.standard.stringArray(forKey: "secondary_languages") ?? []
    )
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
            VStack(alignment: .leading, spacing: Theme.Spacing.xl) {
                Text("Settings")
                    .font(.system(size: 20, weight: .semibold))
                    .foregroundColor(Theme.textPrimary)

                field("OPENAI API KEY",
                      hint: "Used for summaries and for search embeddings.") {
                    SecureField("sk-proj-…", text: $openaiKey)
                        .textFieldStyle(ThemedTextFieldStyle())
                        .frame(maxWidth: 460)
                }

                field("ELEVENLABS API KEY",
                      hint: "Used for real-time speech transcription.") {
                    SecureField("sk_…", text: $elevenlabsKey)
                        .textFieldStyle(ThemedTextFieldStyle())
                        .frame(maxWidth: 460)
                }

                HStack(alignment: .top, spacing: Theme.Spacing.lg) {
                    field("BACKEND HOST", hint: "Local backend address.") {
                        TextField("127.0.0.1", text: $backendHost)
                            .textFieldStyle(ThemedTextFieldStyle())
                            .frame(width: 200)
                    }
                    field("PORT", hint: "Change if the default is occupied.") {
                        TextField("9123", text: $backendPort)
                            .textFieldStyle(ThemedTextFieldStyle())
                            .frame(width: 110)
                    }
                }

                field("PRIMARY LANGUAGE",
                      hint: "Locks speech recognition to this language. Summaries follow it too.") {
                    Picker("", selection: $transcriptionLanguage) {
                        ForEach(languageOptions, id: \.code) { option in
                            Text(option.label).tag(option.code)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                    .frame(width: 240)
                }

                field("SECONDARY LANGUAGES",
                      hint: "Extra languages allowed in the audio — for sentences that switch mid-way. Leave empty if you speak only the language above.") {
                    LazyVGrid(
                        columns: [GridItem(.adaptive(minimum: 150), alignment: .leading)],
                        alignment: .leading,
                        spacing: 2
                    ) {
                        ForEach(languageOptions.filter { !$0.code.isEmpty && $0.code != transcriptionLanguage }, id: \.code) { option in
                            Toggle(option.label, isOn: Binding(
                                get: { secondaryLanguages.contains(option.code) },
                                set: { isOn in
                                    if isOn { secondaryLanguages.insert(option.code) }
                                    else { secondaryLanguages.remove(option.code) }
                                }
                            ))
                            .toggleStyle(.checkbox)
                            .font(.system(size: Theme.FontSize.body))
                            .foregroundColor(Theme.textSecondary)
                        }
                    }
                    .frame(maxWidth: 460, alignment: .leading)
                }

                field("AUTO SUMMARY INTERVAL",
                      hint: "Seconds. A window still waits for the current sentence to finish, so the real gap is slightly longer.") {
                    TextField("45", text: $summaryIntervalSeconds)
                        .textFieldStyle(ThemedTextFieldStyle())
                        .frame(width: 110)
                }

                Rectangle().fill(Theme.border).frame(height: 1)

                HStack(spacing: Theme.Spacing.lg) {
                    Button("Save & Restart Backend") { saveAPIKeys() }
                        .buttonStyle(ThemedPrimaryButtonStyle())

                    if showSaveSuccess {
                        HStack(spacing: 5) {
                            Image(systemName: "checkmark.circle.fill")
                                .font(.system(size: 10))
                            Text("Saved — backend restarting")
                                .font(.system(size: Theme.FontSize.small))
                        }
                        .foregroundColor(Theme.success)
                        .transition(.opacity)
                    }

                    Spacer()
                }

                if let validationMessage {
                    Text(validationMessage)
                        .font(.system(size: Theme.FontSize.small))
                        .foregroundColor(Theme.danger)
                }
            }
            .padding(Theme.Spacing.xl + 6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(Theme.contentBg)
        .frame(minWidth: 560, minHeight: 460)
    }

    /// 统一的表单项：全大写标签 + 控件 + 灰色说明
    @ViewBuilder
    private func field<Content: View>(
        _ title: String,
        hint: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            Text(title).sectionCaption()
            content()
            Text(hint)
                .font(.system(size: Theme.FontSize.micro))
                .foregroundColor(Theme.textFaint)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: 460, alignment: .leading)
        }
    }

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
        // 主语言不该同时出现在次要语言里
        let secondary = secondaryLanguages.filter { $0 != transcriptionLanguage }.sorted()
        UserDefaults.standard.set(secondary, forKey: "secondary_languages")
        
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
            let config: [String: Any] = [
                "openai_api_key": openaiKey,
                "elevenlabs_api_key": elevenlabsKey,
                "transcription_language": transcriptionLanguage,
                "secondary_languages": secondaryLanguages.filter { $0 != transcriptionLanguage }.sorted(),
            ]
            let data = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: configFile, options: .atomic)
        } catch {
            // 写入失败时后端将沿用旧配置
        }
    }
}

// MARK: - Preview

#Preview {
    SettingsView()
}
