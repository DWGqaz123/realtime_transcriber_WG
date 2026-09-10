//
//  SummaryDisplayable.swift
//  RealtimeTranscriberMac
//
//  摘要有两种来源：录音中经 WebSocket 推来的 Summary，和从 REST 读回的
//  SessionSummary。字段命名、时间类型都不一样，但展示需求完全相同——
//  用一个协议抹平差异，卡片视图就只需要一份实现。
//

import Foundation

protocol SummaryDisplayable: Identifiable {
    var id: Int { get }
    var content: String { get }
    /// 已格式化的时间，两种来源的时间类型不同，各自决定怎么显示
    var displayTime: String { get }
    /// 终版摘要（Stop 时对整段转录生成）
    var isFinalSummary: Bool { get }
    /// 覆盖的转录行数，未知时为 nil
    var lineCount: Int? { get }
}

extension SummaryDisplayable {
    /// 把 LLM 返回的 Markdown 列表拆成条目；非列表行并入上一条。
    /// 解析是数据的职责，不该散落在各个卡片视图里。
    var bullets: [String] {
        var result: [String] = []
        for line in content.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("- ") || trimmed.hasPrefix("* ") {
                result.append(String(trimmed.dropFirst(2)))
            } else if !trimmed.isEmpty {
                if let last = result.last {
                    result[result.count - 1] = last + " " + trimmed
                } else {
                    result.append(trimmed)
                }
            }
        }
        return result
    }
}

extension Summary: SummaryDisplayable {
    var displayTime: String { formattedTime }
    var isFinalSummary: Bool { isFinal }
    var lineCount: Int? { sentenceCount }
}

extension SessionSummary: SummaryDisplayable {
    var displayTime: String { formattedTime }
    var lineCount: Int? { sentenceCount > 0 ? sentenceCount : nil }
}
