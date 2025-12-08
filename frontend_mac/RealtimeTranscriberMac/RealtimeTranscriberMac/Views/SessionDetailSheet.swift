//
//  SessionDetailSheet.swift
//  RealtimeTranscriberMac
//
//  Created by 董文光 on 2025/12/7.
//
//  浮窗展示 Session 详情
//

import SwiftUI
import UniformTypeIdentifiers 

struct SessionDetailSheet: View {
    let session: RecordingSession
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Session Transcript")
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    HStack(spacing: 16) {
                        Label(session.modeDisplayName, systemImage: session.mode == "lecture" ? "book.fill" : "bubble.left.and.bubble.right.fill")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        
                        Label(session.formattedDuration, systemImage: "clock.fill")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                        
                        Label(session.formattedStartDate, systemImage: "calendar")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                }
                
                Spacer()
                
                Button(action: {
                    dismiss()
                }) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundColor(.secondary)
                        .font(.title2)
                }
                .buttonStyle(.plain)
            }
            .padding(24)
            
            Divider()
            
            // Stats bar
            HStack(spacing: 24) {
                StatItem(
                    icon: "text.alignleft",
                    label: "Sentences",
                    value: "\(session.sentenceCount)"
                )
                
                Divider()
                    .frame(height: 30)
                
                StatItem(
                    icon: "character",
                    label: "Characters",
                    value: "\(session.charCount)"
                )
                
                Spacer()
                
                // Export button
                Button(action: {
                    exportTranscript()
                }) {
                    HStack(spacing: 6) {
                        Image(systemName: "square.and.arrow.up")
                        Text("Export")
                    }
                }
                .buttonStyle(.bordered)
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 16)
            .background(Color.gray.opacity(0.05))
            
            Divider()
            
            // Transcript content
            ScrollView {
                if let transcript = session.transcriptText, !transcript.isEmpty {
                    VStack(alignment: .leading, spacing: 16) {
                        ForEach(Array(transcript.components(separatedBy: "\n").enumerated()), id: \.offset) { index, sentence in
                            if !sentence.isEmpty {
                                HStack(alignment: .top, spacing: 12) {
                                    Text("\(index + 1)")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                        .frame(width: 40, alignment: .trailing)
                                        .padding(.top, 2)
                                    
                                    Text(sentence)
                                        .font(.body)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .textSelection(.enabled)
                                }
                            }
                        }
                    }
                    .padding(24)
                } else {
                    VStack(spacing: 16) {
                        Image(systemName: "doc.text")
                            .font(.system(size: 48))
                            .foregroundColor(.gray.opacity(0.5))
                        
                        Text("No transcript available")
                            .font(.headline)
                            .foregroundColor(.secondary)
                        
                        Text("This session has no recorded transcript.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding(40)
                }
            }
        }
        .frame(width: 700, height: 600)
    }
    
    // MARK: - Actions
    
    private func exportTranscript() {
        guard let transcript = session.transcriptText, !transcript.isEmpty else {
            return
        }
        
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "session_\(session.id)_\(session.mode).txt"
        panel.canCreateDirectories = true
        panel.allowedContentTypes = [.plainText]
        
        panel.begin { response in
            if response == .OK, let url = panel.url {
                do {
                    // 创建完整内容
                    var content = ""
                    content += "Mode: \(session.modeDisplayName)\n"
                    content += "Date: \(session.formattedStartDate)\n"
                    content += "Duration: \(session.formattedDuration)\n"
                    content += "Sentences: \(session.sentenceCount)\n"
                    content += "=" + String(repeating: "=", count: 59) + "\n\n"
                    content += transcript
                    
                    try content.write(to: url, atomically: true, encoding: .utf8)
                    print("✅ Exported transcript to: \(url.path)")
                } catch {
                    print("❌ Failed to export: \(error)")
                }
            }
        }
    }
}

// MARK: - Stat Item Component

struct StatItem: View {
    let icon: String
    let label: String
    let value: String
    
    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .foregroundColor(.blue)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(label)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Text(value)
                    .font(.headline)
            }
        }
    }
}

// MARK: - Preview

#Preview {
    SessionDetailSheet(
        session: RecordingSession(
            id: 1,
            mode: "lecture",
            durationSeconds: 120,
            sentenceCount: 5,
            charCount: 250,
            startedAt: Date(),
            endedAt: Date(),
            transcriptText: "Hello, this is a test.\nThis is the second sentence.\nAnd another one.\nMore content here.\nFinal sentence."
        )
    )
}
