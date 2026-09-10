//
//  ContentView.swift
//  RealtimeTranscriberMac
//
//  Main content view with project sidebar
//

import SwiftUI


struct ContentView: View {
    @StateObject private var viewModel = TranscribeViewModel()
    @StateObject private var projectViewModel = ProjectListViewModel()
    @State private var showSaveSheet = false

    var body: some View {
        NavigationSplitView {
            ProjectSidebarView(
                viewModel: projectViewModel,
                onNewSession: { project in
                    projectViewModel.selectProject(project)
                    viewModel.currentProjectId = project.id
                    if viewModel.fullTranscript.isEmpty {
                        viewModel.startNewSession()
                    } else {
                        showSaveSheet = true
                    }
                }
            )
        } detail: {
            // 使用 HSplitView 实现双列布局
            HSplitView {
                // 左侧：录音和转录区域
                RecordingView(
                    viewModel: viewModel,
                    projectViewModel: projectViewModel,
                    onRequestNewSession: { showSaveSheet = true }
                )
                .frame(minWidth: 400, idealWidth: 600)
                .task(id: projectViewModel.selectedProject?.id) {
                    if let project = projectViewModel.selectedProject {
                        viewModel.currentProjectId = project.id
                    }
                }
                .onAppear {
                    viewModel.currentProjectId = projectViewModel.selectedProject?.id
                }
                
                // 右侧：智能笔记流
                SummaryPanelView(viewModel: viewModel)
                    .frame(minWidth: 300, idealWidth: 350, maxWidth: 450)
            }
        }
        .navigationSplitViewStyle(.balanced)
        .sheet(isPresented: $showSaveSheet) {
            SaveSessionSheet(isPresented: $showSaveSheet) { name, notes in
                Task {
                    await viewModel.updateSessionMetadata(name: name, notes: notes)
                    viewModel.startNewSession()
                }
            }
        }
        .onAppear {
            viewModel.onSaveComplete = {
                Task {
                    try? await Task.sleep(nanoseconds: 500_000_000)
                    await projectViewModel.loadProjects()
                    await projectViewModel.refreshSelectedProject()
                }
            }
        }
    }
}

// MARK: - Recording View (原有的录音界面)

struct RecordingView: View {
    @ObservedObject var viewModel: TranscribeViewModel
    @ObservedObject var projectViewModel: ProjectListViewModel
    var onRequestNewSession: () -> Void = {}

    var body: some View {
        VStack(spacing: 0) {
            header
            Rectangle().fill(Theme.border).frame(height: 1)
            content
        }
        .background(Theme.contentBg)
        .alert("Microphone Permission Required", isPresented: $viewModel.showPermissionAlert) {
            Button("Open System Settings") {
                if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
                    NSWorkspace.shared.open(url)
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This app needs microphone access to record audio. Please enable microphone access in System Settings → Privacy & Security → Microphone.")
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 2) {
                Text(viewModel.isRecording ? "Recording session"
                     : (viewModel.canResume ? "Paused session" : "New session"))
                    .font(.system(size: Theme.FontSize.title, weight: .semibold))
                    .foregroundColor(Theme.textPrimary)

                Text(projectViewModel.selectedProject?.name ?? "No project selected")
                    .font(.system(size: Theme.FontSize.small))
                    .foregroundColor(projectViewModel.selectedProject == nil ? Theme.warning : Theme.textFaint)
            }

            Spacer()

            modeSelector
        }
        .padding(.horizontal, Theme.Spacing.xl)
        .padding(.vertical, Theme.Spacing.lg)
    }

    private var modeSelector: some View {
        SegmentedControl(
            options: RecordingMode.allCases,
            selection: $viewModel.mode,
            label: { $0.displayName }
        )
        .opacity(viewModel.isRecording ? 0.45 : 1)
        .disabled(viewModel.isRecording)
        .help(viewModel.isRecording
              ? "Stop recording to change mode"
              : "Lecture: fixed-interval commits · Discussion: voice-activity commits")
    }

    // MARK: - Content

    private var content: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Spacing.lg) {
                statusDock
                subtitleSection
                transcriptSection
            }
            .padding(Theme.Spacing.xl)
        }
    }

    // MARK: - 录音状态区

    private var statusDock: some View {
        HStack(spacing: Theme.Spacing.lg) {
            primaryButton

            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: Theme.Spacing.sm) {
                    if viewModel.isRecording {
                        Circle()
                            .fill(Theme.danger)
                            .frame(width: 7, height: 7)
                            .opacity(viewModel.isDetectingSound ? 1 : 0.35)
                            .animation(.easeInOut(duration: 0.25), value: viewModel.isDetectingSound)
                        Text("Recording")
                            .font(.system(size: Theme.FontSize.body, weight: .semibold))
                            .foregroundColor(Theme.danger)
                    } else if viewModel.canResume {
                        Circle().fill(Theme.warning).frame(width: 7, height: 7)
                        Text("Paused · saved")
                            .font(.system(size: Theme.FontSize.body, weight: .semibold))
                            .foregroundColor(Theme.warning)
                    } else {
                        Circle().fill(Theme.textFaint).frame(width: 7, height: 7)
                        Text(projectViewModel.selectedProject == nil ? "Select a project to begin" : "Ready to record")
                            .font(.system(size: Theme.FontSize.body))
                            .foregroundColor(Theme.textMuted)
                    }
                }

                Text(viewModel.formattedDuration)
                    .font(.system(size: Theme.FontSize.large, weight: .medium, design: .monospaced))
                    .foregroundColor(viewModel.isRecording ? Theme.textPrimary : Theme.textFaint)
            }
            .frame(width: 150, alignment: .leading)

            audioLevelGroup

            Spacer()

            if viewModel.canResume {
                Button(action: onRequestNewSession) {
                    Text("New Session")
                        .font(.system(size: Theme.FontSize.body, weight: .medium))
                        .foregroundColor(Theme.textSecondary)
                        .padding(.horizontal, 12)
                        .frame(height: 30)
                        .background(Theme.surface)
                        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.row))
                }
                .buttonStyle(.plain)
                .help("Finish this session and start a new one")
            }
        }
        .padding(Theme.Spacing.lg)
        .background(Theme.panelBg)
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.card))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.card)
                .stroke(Theme.border, lineWidth: 1)
        )
    }

    private var primaryButton: some View {
        let disabled = projectViewModel.selectedProject == nil && !viewModel.isRecording
        return Button {
            if viewModel.isRecording {
                viewModel.stopRecording()
            } else if projectViewModel.selectedProject != nil {
                viewModel.startRecording()
            }
        } label: {
            Image(systemName: viewModel.isRecording ? "stop.fill"
                              : (viewModel.canResume ? "play.fill" : "mic.fill"))
                .font(.system(size: 15, weight: .semibold))
                .foregroundColor(viewModel.isRecording ? Theme.danger : Theme.accent)
                .frame(width: 44, height: 44)
                .background(viewModel.isRecording ? Theme.dangerBg : Theme.accentBg)
                .clipShape(Circle())
        }
        .buttonStyle(.plain)
        .disabled(disabled)
        .opacity(disabled ? 0.4 : 1)
        .help(viewModel.isRecording ? "Stop" : (viewModel.canResume ? "Resume this session" : "Start recording"))
    }

    /// 24 段电平表。高度递增，与设计稿一致；未点亮的段保留可见的底色，
    /// 这样静音时也能看出量程，而不是一片空白。
    private var audioLevelGroup: some View {
        let level = Double(min(sqrt(max(viewModel.audioLevel, 0) * 10.0), 1.0))
        return VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
            HStack(alignment: .bottom, spacing: 2) {
                ForEach(0..<24, id: \.self) { i in
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Double(i) / 24.0 < level ? Theme.accent : Theme.borderStrong)
                        .frame(width: 3, height: 6 + CGFloat(i) * 0.55)
                }
            }
            .frame(height: 20, alignment: .bottom)
            .animation(.easeOut(duration: 0.08), value: viewModel.audioLevel)
            .opacity(viewModel.isRecording ? 1 : 0.3)

            HStack(spacing: Theme.Spacing.sm) {
                Text("Microphone")
                    .font(.system(size: Theme.FontSize.micro))
                    .foregroundColor(Theme.textFaint)
                if viewModel.isRecording {
                    Text(volumeHint.0)
                        .font(.system(size: Theme.FontSize.micro, weight: .medium))
                        .foregroundColor(volumeHint.1)
                }
            }
        }
    }

    private var volumeHint: (String, Color) {
        if viewModel.audioLevel < 0.01 { return ("Too quiet", Theme.warning) }
        if viewModel.audioLevel > 0.5 { return ("Too loud", Theme.danger) }
        if viewModel.audioLevel > 0.05 { return ("Good volume", Theme.success) }
        return ("Listening", Theme.textMuted)
    }

    // MARK: - 实时字幕

    private var subtitleSection: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            HStack {
                sectionLabel("Current Subtitle")
                Spacer()
                if viewModel.isRecording {
                    Chip(text: viewModel.currentSubtitle.isEmpty ? "waiting" : "partial",
                         tint: Theme.accent, background: Theme.accentBg, mono: true)
                }
            }

            Text(viewModel.currentSubtitle.isEmpty
                 ? (viewModel.isRecording ? "Listening…" : "Nothing yet.")
                 : viewModel.currentSubtitle)
                .font(.system(size: Theme.FontSize.medium))
                .foregroundColor(viewModel.currentSubtitle.isEmpty ? Theme.textFaint : Theme.textPrimary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(Theme.Spacing.lg)
                .background(Theme.surface)
                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.card))
        }
    }

    // MARK: - 完整转录

    private var transcriptSection: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            HStack(spacing: Theme.Spacing.md) {
                sectionLabel("Full Transcript")
                Text("\(viewModel.sentenceCount) lines")
                    .font(.system(size: Theme.FontSize.micro, design: .monospaced))
                    .foregroundColor(Theme.textFaint)
                Spacer()
                if !viewModel.fullTranscript.isEmpty {
                    TextChipButton(title: "Clear", help: "Clear transcript") {
                        viewModel.clearTranscript()
                    }
                }
            }

            if viewModel.fullTranscript.isEmpty {
                VStack(spacing: Theme.Spacing.sm) {
                    Image(systemName: "text.alignleft")
                        .font(.system(size: 22))
                        .foregroundColor(Theme.textFaint.opacity(0.6))
                    Text("Confirmed transcripts will appear here")
                        .font(.system(size: Theme.FontSize.body))
                        .foregroundColor(Theme.textFaint)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 34)
                .background(Theme.surface)
                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.card))
            } else {
                VStack(alignment: .leading, spacing: Theme.Spacing.md) {
                    ForEach(Array(transcriptLines.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.system(size: Theme.FontSize.medium))
                            .foregroundColor(Theme.textSecondary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding(Theme.Spacing.lg)
                .background(Theme.surface)
                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.card))
            }
        }
    }

    private var transcriptLines: [String] {
        viewModel.fullTranscript
            .components(separatedBy: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
    }

    private func sectionLabel(_ text: String) -> some View {
        Text(text.uppercased())
            .font(.system(size: Theme.FontSize.micro, weight: .semibold))
            .foregroundColor(Theme.textFaint)
            .tracking(0.6)
    }
}

// MARK: - Helper Extension

extension Optional where Wrapped == String {
    var isNilOrEmpty: Bool {
        return self?.isEmpty ?? true
    }
}

#Preview {
    ContentView()
}
