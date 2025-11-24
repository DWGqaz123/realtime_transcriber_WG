//
//  ContentView.swift
//  RealtimeTranscriberMac
//
//  Created by 董文光 on 2025/11/14.
//
import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = TranscribeViewModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {

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
                .tint(viewModel.isRecording ? .red : .blue)

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
                    }
                } else {
                    Text("Ready to record")
                        .foregroundColor(.secondary)
                }

                Spacer()
            }
            
            // 音频电平可视化区域
            if viewModel.isRecording {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Audio Level")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Spacer()
                        
                        // 静音状态指示
                        if viewModel.isSilent {
                            HStack(spacing: 4) {
                                Image(systemName: "speaker.slash.fill")
                                    .foregroundColor(.gray)
                                Text("Silent (not sending)")
                                    .font(.caption)
                                    .foregroundColor(.gray)
                            }
                        } else if viewModel.isDetectingSound {
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
                                .frame(
                                    width: geometry.size.width * CGFloat(min(sqrt(viewModel.audioLevel * 10.0), 1.0)),
            
                                    height: 24
                                )
                                .cornerRadius(12)
                                .animation(.easeOut(duration: 0.1), value: viewModel.audioLevel)
                            
                            Rectangle()
                                .fill(Color.white.opacity(0.3))
                                .frame(width: 2, height: 24)
                                .offset(x: geometry.size.width * 0.1)
                            
                            Rectangle()
                                .fill(Color.white.opacity(0.3))
                                .frame(width: 2, height: 24)
                                .offset(x: geometry.size.width * 0.5)
                        }
                    }
                    .frame(height: 24)
                    
                    // 电平数值和流量统计
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
                    
                    // 流量统计
                    if viewModel.totalChunks > 0 {
                        Divider()
                        
                        HStack(spacing: 16) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Traffic Statistics")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                
                                HStack(spacing: 12) {
                                    // 发送块数
                                    HStack(spacing: 4) {
                                        Image(systemName: "arrow.up.circle.fill")
                                            .foregroundColor(.blue)
                                            .font(.caption)
                                        Text("\(viewModel.sentChunks)")
                                            .font(.system(.caption, design: .monospaced))
                                    }
                                    
                                    // 跳过块数
                                    HStack(spacing: 4) {
                                        Image(systemName: "xmark.circle.fill")
                                            .foregroundColor(.gray)
                                            .font(.caption)
                                        Text("\(viewModel.skippedChunks)")
                                            .font(.system(.caption, design: .monospaced))
                                    }
                                    
                                    // 总块数
                                    HStack(spacing: 4) {
                                        Image(systemName: "square.grid.3x3.fill")
                                            .foregroundColor(.secondary)
                                            .font(.caption)
                                        Text("\(viewModel.totalChunks)")
                                            .font(.system(.caption, design: .monospaced))
                                    }
                                }
                            }
                            
                            Spacer()
                            
                            // 节省百分比
                            VStack(alignment: .trailing, spacing: 2) {
                                Text("Saved")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                
                                HStack(spacing: 4) {
                                    Text("\(viewModel.trafficSavedPercent)%")
                                        .font(.system(.body, design: .monospaced))
                                        .fontWeight(.bold)
                                        .foregroundColor(viewModel.trafficSavedPercent > 50 ? .green : .orange)
                                    
                                    Text("(\(viewModel.formattedTrafficSaved))")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                        }
                    }
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(8)
            }

            // Current subtitle
            VStack(alignment: .leading, spacing: 8) {
                Text("Current Subtitle")
                    .font(.headline)
                
                if viewModel.currentSubtitle.isEmpty {
                    if viewModel.isRecording {
                        HStack(spacing: 8) {
                            if viewModel.isSilent {
                                Image(systemName: "speaker.slash")
                                    .foregroundColor(.gray)
                                Text("Silence detected - waiting for speech...")
                                    .foregroundColor(.secondary)
                            } else {
                                ProgressView()
                                    .scaleEffect(0.8)
                                Text("Listening...")
                                    .foregroundColor(.secondary)
                            }
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
                RoundedRectangle(cornerRadius: 8)
                    .stroke(viewModel.isRecording ? Color.blue : Color.gray.opacity(0.3), lineWidth: 2)
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
                                }
                            }
                        }
                        .padding()
                    }
                }
                .frame(minHeight: 200)
                .background(Color.gray.opacity(0.05))
                .cornerRadius(8)
            }

            Spacer()
        }
        .padding()
        .frame(minWidth: 800, minHeight: 600)
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

#Preview {
    ContentView()
}



