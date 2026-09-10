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
            headerView
            Rectangle().fill(Theme.border).frame(height: 1)
            tabSelectorView
            Rectangle().fill(Theme.border).frame(height: 1)
            contentView
        }
        .frame(width: 720, height: 620)
        .background(Theme.contentBg)
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
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 3) {
                Text(session.name?.isEmpty == false ? session.name! : session.modeDisplayName)
                    .font(.system(size: Theme.FontSize.title, weight: .semibold))
                    .foregroundColor(Theme.textPrimary)

                HStack(spacing: 5) {
                    Text(session.formattedStartDate)
                    Text("·")
                    Text(session.formattedDuration)
                    Text("·")
                    Text("\(session.sentenceCount) lines")
                    Text("·")
                    Text(session.modeDisplayName)
                }
                .font(.system(size: Theme.FontSize.micro))
                .foregroundColor(Theme.textFaint)

                if let notes = session.notes, !notes.isEmpty {
                    Text(notes)
                        .font(.system(size: Theme.FontSize.small))
                        .foregroundColor(Theme.textMuted)
                        .padding(.top, 3)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            Spacer()

            HStack(spacing: Theme.Spacing.sm) {
                IconButton(icon: "trash", tint: Theme.danger, background: Theme.dangerBg,
                           help: "Delete this entire session") {
                    showDeleteSessionConfirmation = true
                }
                IconButton(icon: "xmark") { isPresented = false }
            }
        }
        .padding(.horizontal, Theme.Spacing.xl)
        .padding(.vertical, Theme.Spacing.lg)
    }

    // MARK: - Tabs

    private var tabSelectorView: some View {
        HStack {
            SegmentedControl(
                options: [Tab.transcript, Tab.summaries],
                selection: $selectedTab,
                label: { $0 == .transcript ? "Transcript" : "Summaries" },
                trailing: { tab in
                    AnyView(
                        Text("\(tab == .transcript ? session.sentenceCount : localSummaries.count)")
                            .font(.system(size: Theme.FontSize.micro, design: .monospaced))
                            .foregroundColor(Theme.textFaint)
                    )
                }
            )
            Spacer()
        }
        .padding(.horizontal, Theme.Spacing.xl)
        .padding(.vertical, Theme.Spacing.md)
    }

    // MARK: - Content

    private var contentView: some View {
        Group {
            switch selectedTab {
            case .transcript: transcriptView
            case .summaries:  summariesView
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var transcriptView: some View {
        VStack(spacing: 0) {
            if let text = session.transcriptText, !text.isEmpty {
                HStack {
                    Text("FULL TRANSCRIPT").sectionCaption()
                    Spacer()
                    TextChipButton(title: "Export", icon: "square.and.arrow.up", help: "Export transcript", action: exportTranscript)
                }
                .padding(.horizontal, Theme.Spacing.xl)
                .padding(.top, Theme.Spacing.lg)
                .padding(.bottom, Theme.Spacing.md)

                ScrollView {
                    Text(text)
                        .font(.system(size: Theme.FontSize.medium))
                        .foregroundColor(Theme.textSecondary)
                        .textSelection(.enabled)
                        .lineSpacing(4)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(Theme.Spacing.lg)
                        .background(Theme.surface)
                        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.card))
                        .padding(.horizontal, Theme.Spacing.xl)
                        .padding(.bottom, Theme.Spacing.xl)
                }
            } else {
                EmptyState(icon: "doc.text", title: "No transcript", hint: "This session has no saved transcript.")
            }
        }
    }

    private var summariesView: some View {
        VStack(spacing: 0) {
            if localSummaries.isEmpty {
                EmptyState(icon: "sparkles", title: "No summaries",
                           hint: "This session has no AI-generated summaries yet.")
            } else {
                HStack {
                    Text("SUMMARIES").sectionCaption()
                    Spacer()
                    TextChipButton(title: "Export all", icon: "square.and.arrow.up", help: "Export all summaries", action: exportAllSummaries)
                }
                .padding(.horizontal, Theme.Spacing.xl)
                .padding(.top, Theme.Spacing.lg)
                .padding(.bottom, Theme.Spacing.md)

                ScrollView {
                    LazyVStack(spacing: Theme.Spacing.md) {
                        ForEach(localSummaries) { summary in
                            SummaryCard(summary: summary) {
                                summaryToDelete = summary
                                showDeleteSummaryConfirmation = true
                            }
                        }
                    }
                    .padding(.horizontal, Theme.Spacing.xl)
                    .padding(.bottom, Theme.Spacing.xl)
                }
            }
        }
    }

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
