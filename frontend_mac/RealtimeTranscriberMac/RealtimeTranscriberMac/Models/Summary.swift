//
//  Summary.swift
//  RealtimeTranscriberMac
//
//  Created by 董文光 on 2025/12/8.
//  智能摘要数据模型
//

import Foundation

struct Summary: Identifiable, Codable, Hashable {
    let id: UUID
    let content: String          // Markdown 格式的摘要内容
    let timestamp: Date          // 生成时间
    let sentenceCount: Int       // 覆盖的句子数
    let duration: Int            // 覆盖的时长（秒）
    
    // 本地生成的 ID（因为后端消息没有 ID）
    init(content: String, timestamp: Date, sentenceCount: Int, duration: Int) {
        self.id = UUID()
        self.content = content
        self.timestamp = timestamp
        self.sentenceCount = sentenceCount
        self.duration = duration
    }
    
    // 从 JSON 解码
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = UUID()
        self.content = try container.decode(String.self, forKey: .content)
        self.timestamp = try container.decode(Date.self, forKey: .timestamp)
        self.sentenceCount = try container.decode(Int.self, forKey: .sentenceCount)
        self.duration = try container.decode(Int.self, forKey: .duration)
    }
    
    enum CodingKeys: String, CodingKey {
        case content
        case timestamp
        case sentenceCount = "sentence_count"
        case duration
    }
    
    // 格式化时间
    var formattedTime: String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .short
        return formatter.localizedString(for: timestamp, relativeTo: Date())
    }
    
    // 格式化时长
    var formattedDuration: String {
        let minutes = duration / 60
        let seconds = duration % 60
        return "\(minutes)m \(seconds)s"
    }
}
