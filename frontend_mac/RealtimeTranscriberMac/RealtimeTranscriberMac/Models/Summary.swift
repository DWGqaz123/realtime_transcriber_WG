//
//  Summary.swift
//  RealtimeTranscriberMac
//
//  Created by 董文光 on 2025/12/8.
//  智能摘要数据模型
//

import Foundation

struct Summary: Identifiable, Codable, Hashable {
    let id: Int                  // 🔧 改为 Int（来自后端数据库 ID）
    let content: String          // Markdown 格式的摘要内容
    let created_at: String       // 🔧 后端返回的 ISO 8601 时间字符串
    let is_final: Bool?          // 🔧 新增：是否是最终摘要
    
    // 🔧 兼容旧字段（可选）
    let sentenceCount: Int?
    let duration: Int?
    
    // 计算属性：解析时间
    var timestamp: Date {
        let formatter = ISO8601DateFormatter()
        return formatter.date(from: created_at) ?? Date()
    }
    
    enum CodingKeys: String, CodingKey {
        case id
        case content
        case created_at
        case is_final
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
        guard let duration = duration else { return "N/A" }
        let minutes = duration / 60
        let seconds = duration % 60
        return "\(minutes)m \(seconds)s"
    }
    
    // 🔧 是否是最终摘要
    var isFinal: Bool {
        return is_final ?? false
    }
}
