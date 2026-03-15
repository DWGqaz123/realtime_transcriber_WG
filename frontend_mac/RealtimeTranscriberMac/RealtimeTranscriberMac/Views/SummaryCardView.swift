//
//  SummaryCardView.swift
//  RealtimeTranscriberMac
//
//  Created by 董文光 on 2025/12/8.
//
//  智能摘要卡片视图
//
import SwiftUI

struct SummaryCardView: View {
    let summary: Summary
    @State private var isExpanded: Bool = true
    
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                // 🔧 最终摘要使用特殊图标
                Image(systemName: summary.isFinal ? "flag.fill" : "sparkles")
                    .foregroundColor(summary.isFinal ? .orange : .orange)
                    .font(.caption)
                
                Text(summary.formattedTime)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                // 🔧 最终摘要徽章
                if summary.isFinal {
                    HStack(spacing: 4) {
                        Text("Final Summary")
                            .font(.caption2)
                            .fontWeight(.semibold)
                    }
                    .foregroundColor(.orange)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.orange.opacity(0.2))
                    .cornerRadius(6)
                }
                
                Spacer()
                
                // Metadata（只在非最终摘要或有数据时显示）
                if let sentenceCount = summary.sentenceCount,
                   let duration = summary.duration {
                    HStack(spacing: 8) {
                        Label("\(sentenceCount)", systemImage: "text.quote")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        
                        Label(summary.formattedDuration, systemImage: "clock")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
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
            .background(
                // 🔧 最终摘要使用更显眼的背景
                summary.isFinal
                    ? Color.orange.opacity(0.15)
                    : Color.orange.opacity(0.1)
            )
            
            // Content
            if isExpanded {
                VStack(alignment: .leading, spacing: 6) {
                    // Parse Markdown bullets
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
                .stroke(
                    summary.isFinal
                        ? Color.orange.opacity(0.4)
                        : Color.gray.opacity(0.15),
                    lineWidth: 1
                )
        )
        .shadow(color: Color.black.opacity(0.03), radius: 2, x: 0, y: 1)
    }
    
    // 解析 Markdown 格式的项目符号
    private func parseMarkdownBullets(_ text: String) -> [String] {
        let lines = text.components(separatedBy: .newlines)
        var bullets: [String] = []
        
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            
            // 匹配 "- " 或 "* " 开头的行
            if trimmed.hasPrefix("- ") {
                bullets.append(String(trimmed.dropFirst(2)))
            } else if trimmed.hasPrefix("* ") {
                bullets.append(String(trimmed.dropFirst(2)))
            } else if !trimmed.isEmpty && !bullets.isEmpty {
                // 如果不是空行且前面有内容，追加到上一个 bullet
                if let last = bullets.last {
                    bullets[bullets.count - 1] = last + " " + trimmed
                }
            }
        }
        
        return bullets
    }
}

// MARK: - Preview

#Preview {
    VStack(spacing: 12) {
        // 🔧 最终摘要预览
        SummaryCardView(summary: Summary(
            id: 1,
            content: """
            - 本次讨论涵盖了机器学习的基础概念和实际应用
            - 详细介绍了监督学习、无监督学习和强化学习三大类别
            - 深入分析了深度学习在图像识别和自然语言处理中的应用
            - 讨论了迁移学习如何加速模型训练过程
            - 总结了当前AI领域的主要挑战和未来发展方向
            """,
            created_at: ISO8601DateFormatter().string(from: Date().addingTimeInterval(-300)),
            is_final: true,  // 🔧 最终摘要
            sentenceCount: nil,
            duration: nil
        ))
        
        // 常规摘要预览
        SummaryCardView(summary: Summary(
            id: 2,
            content: """
            - 机器学习是人工智能的一个重要分支，让计算机能够从数据中自动学习
            - 监督学习需要标注数据，而无监督学习则不需要标签
            - 深度学习通过多层神经网络实现复杂的特征提取
            """,
            created_at: ISO8601DateFormatter().string(from: Date().addingTimeInterval(-600)),
            is_final: false,  // 🔧 常规摘要
            sentenceCount: 8,
            duration: 300
        ))
        
        SummaryCardView(summary: Summary(
            id: 3,
            content: """
            - 强化学习通过奖励机制训练智能体
            - 迁移学习可以利用预训练模型加速训练过程
            """,
            created_at: ISO8601DateFormatter().string(from: Date().addingTimeInterval(-900)),
            is_final: false,
            sentenceCount: 5,
            duration: 280
        ))
    }
    .padding()
    .frame(width: 400)
}
