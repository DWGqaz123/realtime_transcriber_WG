//
//  SummaryPanelView.swift
//  RealtimeTranscriberMac
//
//  Created by 董文光 on 2025/12/8.
//  智能笔记流面板
//

import SwiftUI

struct SummaryPanelView: View {
    @ObservedObject var viewModel: TranscribeViewModel
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Label("Smart Notes", systemImage: "sparkles")
                    .font(.headline)
                    .foregroundColor(.orange)
                
                Spacer()
                
                if !viewModel.summaries.isEmpty {
                    Text("\(viewModel.summaries.count)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(
                            Capsule()
                                .fill(Color.orange.opacity(0.2))
                        )
                    
                    Button(action: {
                        viewModel.clearSummaries()
                    }) {
                        Image(systemName: "trash")
                            .foregroundColor(.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Clear all summaries")
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            
            Divider()
            
            // Content
            if viewModel.summaries.isEmpty {
                emptyStateView
            } else {
                summaryListView
            }
        }
        .background(Color(NSColor.windowBackgroundColor))
    }
    
    // MARK: - Empty State
    
    private var emptyStateView: some View {
        VStack(spacing: 16) {
            Image(systemName: "sparkles.rectangle.stack")
                .font(.system(size: 48))
                .foregroundColor(.gray.opacity(0.5))
            
            Text("No summaries yet")
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text("Summaries will appear here every 5 minutes during recording")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 250)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(24)
    }
    
    // MARK: - Summary List
    
    private var summaryListView: some View {
        ScrollView {
            LazyVStack(spacing: 16) {
                ForEach(Array(viewModel.summaries.enumerated()), id: \.element.id) { index, summary in
                    SummaryCardView(summary: summary)
                        .transition(.asymmetric(
                            insertion: .scale(scale: 0.8).combined(with: .opacity),
                            removal: .opacity
                        ))
                        .animation(.spring(response: 0.4, dampingFraction: 0.7), value: viewModel.summaries.count)
                }
            }
            .padding(20)
        }
    }
}

#Preview {
    SummaryPanelView(viewModel: TranscribeViewModel())
        .frame(width: 350, height: 600)
}
