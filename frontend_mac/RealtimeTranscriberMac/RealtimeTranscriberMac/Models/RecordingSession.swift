//
//  RecordingSession.swift
//  RealtimeTranscriberMac
//
//  Recording session data model
//

import Foundation

struct RecordingSession: Identifiable, Codable, Hashable {
    let id: Int
    let mode: String
    let durationSeconds: Int
    let sentenceCount: Int
    let charCount: Int
    let startedAt: Date
    let endedAt: Date?
    
    enum CodingKeys: String, CodingKey {
        case id
        case mode
        case durationSeconds = "duration_seconds"
        case sentenceCount = "sentence_count"
        case charCount = "char_count"
        case startedAt = "started_at"
        case endedAt = "ended_at"
    }
    
    // 格式化时长
    var formattedDuration: String {
        let hours = durationSeconds / 3600
        let minutes = (durationSeconds % 3600) / 60
        let seconds = durationSeconds % 60
        
        if hours > 0 {
            return String(format: "%dh %dm %ds", hours, minutes, seconds)
        } else if minutes > 0 {
            return String(format: "%dm %ds", minutes, seconds)
        } else {
            return String(format: "%ds", seconds)
        }
    }
    
    // 格式化日期
    var formattedStartDate: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter.string(from: startedAt)
    }
    
    // 会话状态
    var isCompleted: Bool {
        return endedAt != nil
    }
}
