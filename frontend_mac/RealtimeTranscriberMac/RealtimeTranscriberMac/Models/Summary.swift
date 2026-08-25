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
    // 后端发的是 datetime.utcnow().isoformat()：无时区、6 位小数，
    // 纯 ISO8601DateFormatter 解析不了，会静默退化成 Date()（永远显示"刚刚"）。
    var timestamp: Date {
        Summary.parseTimestamp(created_at) ?? Date()
    }

    private static let utcFormatters: [DateFormatter] = ["yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
                                                         "yyyy-MM-dd'T'HH:mm:ss"].map { format in
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(secondsFromGMT: 0)
        f.dateFormat = format
        return f
    }

    private static let iso8601Formatters: [ISO8601DateFormatter] = {
        let withFraction = ISO8601DateFormatter()
        withFraction.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        return [withFraction, plain]
    }()

    static func parseTimestamp(_ value: String) -> Date? {
        for formatter in iso8601Formatters {
            if let date = formatter.date(from: value) { return date }
        }
        for formatter in utcFormatters {
            if let date = formatter.date(from: value) { return date }
        }
        return nil
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
