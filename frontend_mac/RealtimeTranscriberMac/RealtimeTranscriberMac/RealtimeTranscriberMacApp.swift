//
//  RealtimeTranscriberMacApp.swift
//  RealtimeTranscriberMac
//
//  Created by 董文光 on 2026/1/12.
//


import SwiftUI

@main
struct RealtimeTranscriberMacApp: App {
    @StateObject private var backendManager = BackendManager()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(backendManager)
                .onAppear {
                    // 🔧 设置通知监听器
                    setupNotificationObservers()
                    

                }
        }
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About Realtime Transcriber") {
                    let statusText = backendManager.isRunning ? "✅ Running" : "❌ Stopped"
                    let errorText = backendManager.error ?? "No errors"
                    
                    NSApplication.shared.orderFrontStandardAboutPanel(
                        options: [
                            .credits: NSAttributedString(
                                string: "Backend Status: \(statusText)\n\(errorText)",
                                attributes: [
                                    .font: NSFont.systemFont(ofSize: 10),
                                    .foregroundColor: NSColor.secondaryLabelColor
                                ]
                            ),
                            .applicationVersion: "1.0.0"
                        ]
                    )
                }
            }
        }
        
        // 添加设置窗口
        Settings {
            SettingsView()
        }
    }
    
    // MARK: - Private Methods
    
    /// 设置通知监听器
    private func setupNotificationObservers() {
        // 🔧 监听重启后端通知
        NotificationCenter.default.addObserver(
            forName: .restartBackend,
            object: nil,
            queue: .main
        ) { [backendManager] _ in
            // 停止后端
            backendManager.stopBackend()
            
            // 延迟 1 秒后重启
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                backendManager.startBackend()
            }
        }
    }
}
