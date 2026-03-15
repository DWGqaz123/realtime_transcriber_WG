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
        VStack(alignment: .leading, spacing: 16) {
            
            // Project info bar
            if let project = projectViewModel.selectedProject {
                HStack {
                    Image(systemName: "folder.fill")
                        .foregroundColor(.indigo)
                    
                    Text(project.name)
                        .font(.headline)
                    
                    if !project.description.isNilOrEmpty {
                        Text("•")
                            .foregroundColor(.secondary)
                        Text(project.description!)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                    }
                    
                    Spacer()
                    
                    Text("\(project.sessionCount) sessions")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(.ultraThinMaterial)
                .cornerRadius(12)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.indigo.opacity(0.2), lineWidth: 1)
                )
            } else {
                HStack {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                    Text("Please select or create a project to start recording")
                        .foregroundColor(.secondary)
                }
                .padding()
                .frame(maxWidth: .infinity)
                .background(Color.orange.opacity(0.06))
                .cornerRadius(8)
            }
            
            // Mode selector and status bar
            HStack {
                Text("Mode:")
                Picker("Mode", selection: $viewModel.mode) {
                    ForEach(RecordingMode.allCases) { mode in
                        Text(mode.displayName).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 260)
                .disabled(viewModel.isRecording)
                
                Spacer()
                
                // 麦克风权限状态
                Text("Mic: \(viewModel.permissionStatus)")
                    .font(.caption)
                    .foregroundColor(viewModel.permissionStatus.contains("✅") ? .green : .orange)
            }
            
            // Control bar with recording indicator
            HStack(spacing: 12) {
                Button(action: {
                    if viewModel.isRecording {
                        viewModel.stopRecording()
                    } else {
                        // 检查是否选择了项目
                        guard projectViewModel.selectedProject != nil else {
                            return
                        }
                        viewModel.startRecording()
                    }
                }) {
                    HStack(spacing: 8) {
                        Image(systemName: viewModel.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                        Text(viewModel.isRecording ? "Stop" : "Start")
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                }
                .buttonStyle(.borderedProminent)
                .tint(viewModel.isRecording ? .red : .indigo)
                .disabled(projectViewModel.selectedProject == nil && !viewModel.isRecording)
                
                // New Session button
                if !viewModel.isRecording && !viewModel.fullTranscript.isEmpty {
                    Button(action: onRequestNewSession) {
                        HStack(spacing: 8) {
                            Image(systemName: "plus.circle.fill")
                            Text("New Session")
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.green)
                }

                // 录音状态指示器
                if viewModel.isRecording {
                    HStack(spacing: 8) {
                        Circle()
                            .fill(Color.red)
                            .frame(width: 8, height: 8)
                            .opacity(viewModel.isDetectingSound ? 1.0 : 0.3)
                            .animation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true), value: viewModel.isDetectingSound)
                        
                        Text("Recording")
                            .foregroundColor(.red)
                            .fontWeight(.semibold)
                        
                        Text(viewModel.formattedDuration)
                            .font(.system(.body, design: .monospaced))
                            .foregroundColor(.secondary)
                        if !viewModel.summaries.isEmpty {
                            Text("•")
                                .foregroundColor(.secondary)
                            
                            HStack(spacing: 4) {
                                Image(systemName: "sparkles")
                                    .foregroundColor(.orange)
                                Text("\(viewModel.summaries.count)")
                                    .foregroundColor(.orange)
                            }
                            .font(.caption)
                        }
                    }
                    .font(.caption)
                }else {
                    if projectViewModel.selectedProject != nil {
                        Text("Ready to record")
                            .foregroundColor(.secondary)
                    } else {
                        Text("Select a project to begin")
                            .foregroundColor(.orange)
                    }
                }
                
                Spacer()
            }
            
            // summary 生成进度
            if viewModel.isRecording && !viewModel.summaryGenerationProgress.isEmpty {
                HStack(spacing: 12) {
                    // 进度文本
                    Text(viewModel.summaryGenerationProgress)
                        .font(.system(size: 13))
                        .foregroundColor(viewModel.isGeneratingSummary ? .orange : .blue)
                    
                    Spacer()
                    
                    // 倒计时数字
                    if !viewModel.isGeneratingSummary && viewModel.nextSummaryCountdown > 0 {
                        Text("\(viewModel.nextSummaryCountdown)s")
                            .font(.system(size: 13, weight: .medium, design: .monospaced))
                            .foregroundColor(.secondary)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                            .background(Color.secondary.opacity(0.15))
                            .cornerRadius(8)
                    }
                    
                    // 生成中的加载动画
                    if viewModel.isGeneratingSummary {
                        ProgressView()
                            .scaleEffect(0.7)
                            .frame(width: 16, height: 16)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(
                            viewModel.isGeneratingSummary
                            ? Color.orange.opacity(0.1)
                            : Color.blue.opacity(0.08)
                        )
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(
                            viewModel.isGeneratingSummary
                            ? Color.orange.opacity(0.3)
                            : Color.blue.opacity(0.2),
                            lineWidth: 1
                        )
                )
            }
            
            // 🔧 ==================== 可选：进度条（视觉化）====================
            if viewModel.isRecording && !viewModel.isGeneratingSummary && viewModel.nextSummaryCountdown > 0 {
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        // 背景
                        Rectangle()
                            .fill(Color.secondary.opacity(0.15))
                            .frame(height: 3)
                            .cornerRadius(1.5)
                        
                        // 进度
                        Rectangle()
                            .fill(
                                LinearGradient(
                                    gradient: Gradient(colors: [.blue, .purple]),
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .frame(
                                width: max(
                                    0,
                                    geometry.size.width
                                        * CGFloat(viewModel.summaryIntervalSeconds - viewModel.nextSummaryCountdown)
                                        / CGFloat(max(viewModel.summaryIntervalSeconds, 1))
                                ),
                                height: 3
                            )
                            .cornerRadius(1.5)
                            .animation(.linear(duration: 0.5), value: viewModel.nextSummaryCountdown)
                    }
                }
                .frame(height: 3)
                .padding(.horizontal, 16)
                .padding(.top, 4)
            }
            
            
            // 音频电平可视化区域
            if viewModel.isRecording {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Audio Level")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Spacer()
                        
                        // 音量状态指示
                        if viewModel.isDetectingSound {
                            HStack(spacing: 4) {
                                Image(systemName: "waveform")
                                    .foregroundColor(.green)
                                Text("Sound Detected")
                                    .font(.caption)
                                    .foregroundColor(.green)
                            }
                        } else {
                            HStack(spacing: 4) {
                                Image(systemName: "waveform")
                                    .foregroundColor(.gray)
                                Text("Speak now...")
                                    .font(.caption)
                                    .foregroundColor(.orange)
                            }
                        }
                    }
                    
                    // 音频电平条
                    GeometryReader { geometry in
                        ZStack(alignment: .leading) {
                            Rectangle()
                                .fill(Color.gray.opacity(0.2))
                                .frame(height: 24)
                                .cornerRadius(12)
                            
                            Rectangle()
                                .fill(
                                    LinearGradient(
                                        gradient: Gradient(colors: [
                                            .green,
                                            .yellow,
                                            .orange,
                                            .red
                                        ]),
                                        startPoint: .leading,
                                        endPoint: .trailing
                                    )
                                )
                                .frame(width: geometry.size.width * CGFloat(min(sqrt(viewModel.audioLevel * 10.0), 1.0)), height: 24)
                                .cornerRadius(12)
                                .animation(.easeOut(duration: 0.1), value: viewModel.audioLevel)
                        }
                    }
                    .frame(height: 24)
                    
                    // 电平数值
                    HStack {
                        Text("Level: \(String(format: "%.3f", viewModel.audioLevel))")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.secondary)
                        
                        Spacer()
                        
                        // 音量提示
                        if viewModel.audioLevel < 0.01 {
                            Text("Too quiet - Speak louder")
                                .font(.caption)
                                .foregroundColor(.orange)
                        } else if viewModel.audioLevel > 0.5 {
                            Text("Too loud - Reduce volume")
                                .font(.caption)
                                .foregroundColor(.red)
                        } else if viewModel.audioLevel > 0.05 {
                            Text("Good volume ✓")
                                .font(.caption)
                                .foregroundColor(.green)
                        }
                    }
                }
                .padding()
                .background(Color.gray.opacity(0.04))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.gray.opacity(0.15), lineWidth: 1)
                )
                .cornerRadius(8)
            }
            
            // Current subtitle
            VStack(alignment: .leading, spacing: 8) {
                Text("Current Subtitle")
                    .font(.headline)
                
                if viewModel.currentSubtitle.isEmpty {
                    if viewModel.isRecording {
                        HStack(spacing: 8) {
                            ProgressView()
                                .controlSize(.small)
                            Text("Listening...")
                                .foregroundColor(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                    } else {
                        Text("No current subtitle.")
                            .foregroundColor(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding()
                    }
                } else {
                    Text(viewModel.currentSubtitle)
                        .font(.title3)
                        .padding()
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(viewModel.isRecording ? Color.indigo.opacity(0.5) : Color.gray.opacity(0.15), lineWidth: 1)
            )
            
            // Full transcript
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Full Transcript")
                        .font(.headline)
                    
                    Spacer()
                    
                    if !viewModel.fullTranscript.isEmpty {
                        HStack(spacing: 12) {
                            Text("\(viewModel.sentenceCount) sentences")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            Text("•")
                                .foregroundColor(.secondary)
                            
                            Text("\(viewModel.fullTranscript.count) chars")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            Button(action: {
                                viewModel.clearTranscript()
                            }) {
                                Image(systemName: "trash")
                                    .foregroundColor(.red)
                            }
                            .buttonStyle(.plain)
                            .help("Clear transcript")
                        }
                    }
                }
                
                ScrollViewReader { proxy in
                    ScrollView {
                        if viewModel.fullTranscript.isEmpty {
                            VStack(spacing: 8) {
                                Image(systemName: "text.bubble")
                                    .font(.system(size: 40))
                                    .foregroundColor(.gray.opacity(0.5))
                                
                                Text("Final transcripts will appear here...")
                                    .foregroundColor(.secondary)
                                
                                if viewModel.isRecording {
                                    Text("Keep speaking to see your transcriptions")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 40)
                        } else {
                            VStack(alignment: .leading, spacing: 12) {
                                ForEach(Array(viewModel.fullTranscript.components(separatedBy: "\n").enumerated()), id: \.offset) { index, sentence in
                                    if !sentence.isEmpty {
                                        HStack(alignment: .top, spacing: 8) {
                                            Text("\(index + 1).")
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                                .frame(width: 30, alignment: .trailing)
                                            
                                            Text(sentence)
                                                .frame(maxWidth: .infinity, alignment: .leading)
                                                .textSelection(.enabled)
                                        }
                                        .padding(.vertical, 4)
                                        .id(index)  // 🔧 给每一行添加 ID
                                    }
                                }
                                
                                // 🔧 底部锚点（不可见）
                                Color.clear
                                    .frame(height: 1)
                                    .id("transcriptBottom")
                            }
                            .padding()
                        }
                    }
                    .frame(minHeight: 200)
                    .background(Color.gray.opacity(0.03))
                    .cornerRadius(12)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.gray.opacity(0.1), lineWidth: 1)
                    )
                    .onChange(of: viewModel.fullTranscript) { _ in
                        // 🔧 转录内容变化时自动滚动到底部
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                            withAnimation(.easeOut(duration: 0.3)) {
                                proxy.scrollTo("transcriptBottom", anchor: .bottom)
                            }
                        }
                    }
                    .onAppear {
                        // 🔧 首次加载时如果有内容，滚动到底部
                        if !viewModel.fullTranscript.isEmpty {
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                                withAnimation(.easeOut(duration: 0.3)) {
                                    proxy.scrollTo("transcriptBottom", anchor: .bottom)
                                }
                            }
                        }
                    }
                }
                
                Spacer()
            }
        }
        .padding()
        .frame(minWidth: 600, minHeight: 600)
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
