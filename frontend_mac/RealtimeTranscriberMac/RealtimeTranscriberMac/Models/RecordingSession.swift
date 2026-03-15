//
//  RecordingSession.swift
//  RealtimeTranscriberMac
//
//  Recording Session 数据模型
//

import Foundation

struct RecordingSession: Identifiable, Codable, Hashable {
    let id: Int
    let mode: String
    let name: String?
    let notes: String?
    let durationSeconds: Int
    let sentenceCount: Int
    let charCount: Int
    let startedAt: Date
    let endedAt: Date?
    let transcriptText: String?
    let summaries: [SessionSummary]?

    enum CodingKeys: String, CodingKey {
        case id
        case mode
        case name
        case notes
        case durationSeconds = "duration_seconds"
        case sentenceCount = "sentence_count"
        case charCount = "char_count"
        case startedAt = "started_at"
        case endedAt = "ended_at"
        case transcriptText = "transcript_text"
        case summaries
    }
    
    // 🔧 手动实现 Hashable
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    // 🔧 手动实现 Equatable
    static func == (lhs: RecordingSession, rhs: RecordingSession) -> Bool {
        return lhs.id == rhs.id
    }
    
    // MARK: - Computed Properties
    
    var modeDisplayName: String {
        switch mode.lowercased() {
        case "lecture":
            return "Lecture"
        case "discussion":
            return "Discussion"
        default:
            return mode.capitalized
        }
    }
    
    var formattedDuration: String {
        let hours = durationSeconds / 3600
        let minutes = (durationSeconds % 3600) / 60
        let seconds = durationSeconds % 60
        
        if hours > 0 {
            return "\(hours)h \(minutes)m \(seconds)s"
        } else if minutes > 0 {
            return "\(minutes)m \(seconds)s"
        } else {
            return "\(seconds)s"
        }
    }
    
    var formattedStartDate: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter.string(from: startedAt)
    }
    
    var transcriptPreview: String {
        guard let text = transcriptText, !text.isEmpty else {
            return "No transcript"
        }
        
        let maxLength = 100
        if text.count > maxLength {
            return String(text.prefix(maxLength)) + "..."
        }
        return text
    }
}

// MARK: - Session Summary Model

struct SessionSummary: Identifiable, Codable, Hashable {
    let id: Int
    let content: String
    let createdAt: Date
    let sentenceCount: Int
    let durationSeconds: Int
    let startSentenceIdx: Int
    let endSentenceIdx: Int
    
    enum CodingKeys: String, CodingKey {
        case id
        case content
        case createdAt = "created_at"
        case sentenceCount = "sentence_count"
        case durationSeconds = "duration_seconds"
        case startSentenceIdx = "start_sentence_idx"
        case endSentenceIdx = "end_sentence_idx"
    }
    
    // 🔧 手动实现 Hashable
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    // 🔧 手动实现 Equatable
    static func == (lhs: SessionSummary, rhs: SessionSummary) -> Bool {
        return lhs.id == rhs.id
    }
    
    // MARK: - Computed Properties
    
    var formattedTime: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter.string(from: createdAt)
    }
    
    var formattedDuration: String {
        let minutes = durationSeconds / 60
        let seconds = durationSeconds % 60
        return "\(minutes)m \(seconds)s"
    }
}
