//
//  SummaryPanelView.swift
//  RealtimeTranscriberMac
//
//  智能笔记流面板
//

import SwiftUI

struct SummaryPanelView: View {
    @ObservedObject var viewModel: TranscribeViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.lg) {
            header

            if viewModel.summaries.isEmpty {
                emptyState
            } else {
                list
            }

            if viewModel.isRecording {
                generationStatus
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, Theme.Spacing.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Theme.panelBg)
    }

    // MARK: - Header

    private var header: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: Theme.Spacing.sm) {
                    Text("Live Summary")
                        .font(.system(size: Theme.FontSize.medium, weight: .semibold))
                        .foregroundColor(Theme.textPrimary)

                    Chip(text: "AI", tint: Theme.violet)
                }

                // 倒计时。设计稿把它放在面板头部而非主区域——它描述的是
                // 这一栏什么时候会新增内容，放这里语义更贴合。
                Text(countdownText)
                    .font(.system(size: Theme.FontSize.micro, design: .monospaced))
                    .foregroundColor(viewModel.isGeneratingSummary ? Theme.accent : Theme.textFaint)
            }

            Spacer()

            if !viewModel.summaries.isEmpty {
                Button {
                    viewModel.clearSummaries()
                } label: {
                    Image(systemName: "trash")
                        .font(.system(size: 10))
                        .foregroundColor(Theme.textFaint)
                        .frame(width: 24, height: 24)
                        .background(Theme.surface)
                        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.chip))
                }
                .buttonStyle(.plain)
                .help("Clear all summaries")
            }
        }
    }

    private var countdownText: String {
        if viewModel.isGeneratingSummary { return "generating…" }
        guard viewModel.isRecording else {
            return "\(viewModel.summaries.count) note\(viewModel.summaries.count == 1 ? "" : "s")"
        }
        if viewModel.nextSummaryCountdown > 0 { return "next in \(viewModel.nextSummaryCountdown)s" }
        return "waiting for a full sentence"
    }

    // MARK: - List

    private var list: some View {
        ScrollView {
            LazyVStack(spacing: Theme.Spacing.md) {
                ForEach(viewModel.summaries) { summary in
                    SummaryCard(summary: summary)
                        .transition(.asymmetric(
                            insertion: .move(edge: .top).combined(with: .opacity),
                            removal: .opacity
                        ))
                }
            }
            .animation(.spring(response: 0.35, dampingFraction: 0.8), value: viewModel.summaries.count)
        }
    }

    // MARK: - Empty

    private var emptyState: some View {
        EmptyState(
            icon: "sparkles.rectangle.stack",
            title: "No summaries yet",
            // 间隔可在设置里改，也能用环境变量覆盖，所以读实际配置值而不是写死。
            // "about" 是必要的：到点后还要等当前这句说完才会截断。
            hint: "Summaries appear about every \(viewModel.summaryIntervalSeconds)s while recording, at the end of a sentence."
        )
    }

    // MARK: - Footer

    private var generationStatus: some View {
        HStack(spacing: Theme.Spacing.sm) {
            if viewModel.isGeneratingSummary {
                ProgressView()
                    .controlSize(.small)
                    .scaleEffect(0.65)
                    .frame(width: 12, height: 12)
            } else {
                Circle().fill(Theme.success).frame(width: 5, height: 5)
            }

            Text(viewModel.isGeneratingSummary ? "Generating summary…" : "Listening for the next window")
                .font(.system(size: Theme.FontSize.micro))
                .foregroundColor(Theme.textFaint)

            Spacer()
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .frame(maxWidth: .infinity)
        .background(Theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.row))
    }
}

#Preview {
    SummaryPanelView(viewModel: TranscribeViewModel())
        .frame(width: 286, height: 700)
}
