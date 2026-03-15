//
//  SessionDetailSheet.swift
//  RealtimeTranscriberMac
//
//  Session 详情浮窗 - 包含转录和摘要（带删除功能）
//

import SwiftUI
import UniformTypeIdentifiers

struct SessionDetailSheet: View {
    let session: RecordingSession
    @Binding var isPresented: Bool
    
    @State private var selectedTab: Tab = .transcript
    @State private var showDeleteSessionConfirmation = false
    @State private var showDeleteSummaryConfirmation = false
    @State private var summaryToDelete: SessionSummary? = nil
    @State private var localSummaries: [SessionSummary]  // 本地副本，用于删除后更新
    
    // 从外部传入的删除回调
    var onDeleteSession: (() -> Void)? = nil
    var onDeleteSummary: ((Int) -> Void)? = nil
    
    init(session: RecordingSession, isPresented: Binding<Bool>, onDeleteSession: (() -> Void)? = nil, onDeleteSummary: ((Int) -> Void)? = nil) {
        self.session = session
        self._isPresented = isPresented
        self.onDeleteSession = onDeleteSession
        self.onDeleteSummary = onDeleteSummary
        self._localSummaries = State(initialValue: session.summaries ?? [])
    }
    
    enum Tab {
        case transcript
        case summaries
    }
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            headerView
            
            Divider()
            
            // Tab selector
            tabSelectorView
            
            Divider()
            
            // Content
            contentView
        }
        .frame(width: 700, height: 600)
        .alert("Delete Session", isPresented: $showDeleteSessionConfirmation) {
                Button("Cancel", role: .cancel) {
                }
                Button("Delete", role: .destructive) {
                    deleteSessionConfirmed()
                }
            } message: {
                Text("Are you sure you want to delete this entire session? This will delete the transcript and all \(localSummaries.count) summaries. This cannot be undone.")
            }
            .alert("Delete Summary", isPresented: $showDeleteSummaryConfirmation) {
                Button("Cancel", role: .cancel) {
                    summaryToDelete = nil
                }
                Button("Delete", role: .destructive) {
                    deleteSummaryConfirmed()
                }
            } message: {
                Text("Are you sure you want to delete this summary? This cannot be undone.")
            }
    }
    
    // MARK: - Header
    
    private var headerView: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Session Details")
                    .font(.headline)
                
                HStack(spacing: 12) {
                    Label(session.modeDisplayName, systemImage: session.mode == "lecture" ? "book.fill" : "bubble.left.and.bubble.right.fill")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Label(session.formattedDuration, systemImage: "clock")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Label(session.formattedStartDate, systemImage: "calendar")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            Spacer()
            
            // 🔧 新增：删除 Session 按钮
            Button(action: {
                showDeleteSessionConfirmation = true
            }) {
                Label("Delete Session", systemImage: "trash")
                    .foregroundColor(.red)
                    .font(.caption)
            }
            .buttonStyle(.bordered)
            .help("Delete this entire session")
            
            Button(action: { isPresented = false }) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundColor(.secondary)
                    .imageScale(.large)
            }
            .buttonStyle(.plain)
        }
        .padding()
    }
    
    // MARK: - Tab Selector
    
    private var tabSelectorView: some View {
        HStack(spacing: 0) {
            tabButton(
                title: "Transcript",
                icon: "doc.text",
                tab: .transcript,
                count: session.sentenceCount
            )
            
            tabButton(
                title: "Summaries",
                icon: "sparkles",
                tab: .summaries,
                count: localSummaries.count
            )
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color(NSColor.controlBackgroundColor))
    }
    
    private func tabButton(title: String, icon: String, tab: Tab, count: Int) -> some View {
        Button(action: {
            withAnimation(.easeInOut(duration: 0.2)) {
                selectedTab = tab
            }
        }) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                Text(title)
                
                if count > 0 {
                    Text("\(count)")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(
                            Capsule()
                                .fill(Color.secondary.opacity(0.2))
                        )
                }
            }
            .font(.subheadline)
            .foregroundColor(selectedTab == tab ? .primary : .secondary)
            .padding(.vertical, 6)
            .padding(.horizontal, 12)
            .background(
                selectedTab == tab
                ? Color.indigo.opacity(0.12)
                    : Color.clear
            )
            .cornerRadius(6)
        }
        .buttonStyle(.plain)
    }
    
    // MARK: - Content
    
    private var contentView: some View {
        Group {
            switch selectedTab {
            case .transcript:
                transcriptView
            case .summaries:
                summariesView
            }
        }
    }
    
    // MARK: - Transcript View
    
    private var transcriptView: some View {
        VStack(spacing: 0) {
            // Stats bar
            HStack {
                Label("\(session.sentenceCount) sentences", systemImage: "text.quote")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Label("\(session.charCount) characters", systemImage: "character")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Button(action: exportTranscript) {
                    Label("Export", systemImage: "square.and.arrow.up")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            
            Divider()
            
            // Transcript content
            ScrollView {
                if let transcript = session.transcriptText, !transcript.isEmpty {
                    Text(transcript)
                        .font(.system(size: 13))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                } else {
                    Text("No transcript available")
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .padding()
                }
            }
        }
    }
    
    // MARK: - Summaries View
    
    private var summariesView: some View {
        VStack(spacing: 0) {
            if !localSummaries.isEmpty {
                // Stats bar
                HStack {
                    Label("\(localSummaries.count) summaries", systemImage: "sparkles")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Button(action: exportAllSummaries) {
                        Label("Export All", systemImage: "square.and.arrow.up")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                }
                .padding()
                .background(Color(NSColor.controlBackgroundColor))
                
                Divider()
                
                // Summaries list
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(localSummaries) { summary in
                            SessionSummaryCardView(
                                summary: summary,
                                onDelete: {  // 🔧 新增删除回调
                                    summaryToDelete = summary
                                    showDeleteSummaryConfirmation = true
                                }
                            )
                        }
                    }
                    .padding()
                }
            } else {
                emptyStateView
            }
        }
    }
    
    private var emptyStateView: some View {
        VStack(spacing: 16) {
            Image(systemName: "sparkles.rectangle.stack")
                .font(.system(size: 48))
                .foregroundColor(.gray.opacity(0.5))
            
            Text("No summaries generated")
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text("This session doesn't have any AI-generated summaries yet")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 300)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(24)
    }
    
    // MARK: - Export Functions
    
    private func exportTranscript() {
        guard let transcript = session.transcriptText, !transcript.isEmpty else {
            return
        }
        
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "session_\(session.id)_transcript.txt"
        panel.allowedContentTypes = [.plainText]
        
        if panel.runModal() == .OK, let url = panel.url {
            do {
                try transcript.write(to: url, atomically: true, encoding: .utf8)
            } catch {
            }
        }
    }
    
    private func exportAllSummaries() {
        guard !localSummaries.isEmpty else {
            return
        }
        
        let content = localSummaries.enumerated().map { index, summary in
            """
            ## Summary \(index + 1)
            **Time**: \(summary.formattedTime)
            **Duration**: \(summary.formattedDuration)
            **Sentences**: \(summary.sentenceCount)
            
            \(summary.content)
            """
        }.joined(separator: "\n\n---\n\n")
        
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "session_\(session.id)_summaries.md"
        panel.allowedContentTypes = [UTType(filenameExtension: "md") ?? .plainText]
        
        if panel.runModal() == .OK, let url = panel.url {
            do {
                try content.write(to: url, atomically: true, encoding: .utf8)
            } catch {
            }
        }
    }
    
    // MARK: - Delete Handlers

    private func deleteSummaryConfirmed() {
        guard let summary = summaryToDelete else {
            return
        }
        
        
        // 从本地列表移除
        withAnimation {
            localSummaries.removeAll { $0.id == summary.id }
        }
        
        
        // 调用外部回调
        if let callback = onDeleteSummary {
            callback(summary.id)
        } else {
        }
        
        summaryToDelete = nil
    }

    private func deleteSessionConfirmed() {
        
        // 关闭浮窗
        isPresented = false
        
        // 调用外部回调
        if let callback = onDeleteSession {
            callback()
        } else {
        }
    }
}

// MARK: - Session Summary Card View

struct SessionSummaryCardView: View {
    let summary: SessionSummary
    let onDelete: () -> Void  // 🔧 新增删除回调
    
    @State private var isExpanded: Bool = true
    @State private var isHovering: Bool = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Image(systemName: "sparkles")
                    .foregroundColor(.orange)
                    .font(.caption)
                
                Text(summary.formattedTime)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                // Metadata
                HStack(spacing: 8) {
                    Label("\(summary.sentenceCount)", systemImage: "text.quote")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    
                    Label(summary.formattedDuration, systemImage: "clock")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                
                // 🔧 新增：删除按钮（hover 时显示）
                if isHovering {
                    Button(action: {
                        onDelete()
                    }) {
                        Image(systemName: "trash")
                            .foregroundColor(.red)
                            .font(.caption)
                    }
                    .buttonStyle(.plain)
                    .help("Delete summary")
                }
                
                // Expand/Collapse button
                Button(action: {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isExpanded.toggle()
                    }
                }) {
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .foregroundColor(.secondary)
                        .font(.caption)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color.orange.opacity(0.1))
            
            // Content
            if isExpanded {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(parseMarkdownBullets(summary.content), id: \.self) { bullet in
                        HStack(alignment: .top, spacing: 8) {
                            Text("•")
                                .foregroundColor(.orange)
                                .font(.system(size: 14, weight: .bold))
                            
                            Text(bullet)
                                .font(.system(size: 13))
                                .foregroundColor(.primary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.gray.opacity(0.15), lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.05), radius: 2, x: 0, y: 1)
        .onHover { hovering in
            isHovering = hovering
        }
    }
    
    private func parseMarkdownBullets(_ text: String) -> [String] {
        let lines = text.components(separatedBy: .newlines)
        var bullets: [String] = []
        
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            
            if trimmed.hasPrefix("- ") {
                bullets.append(String(trimmed.dropFirst(2)))
            } else if trimmed.hasPrefix("* ") {
                bullets.append(String(trimmed.dropFirst(2)))
            } else if !trimmed.isEmpty && !bullets.isEmpty {
                if let last = bullets.last {
                    bullets[bullets.count - 1] = last + " " + trimmed
                }
            }
        }
        
        return bullets
    }
}
