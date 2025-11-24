//
//  Project.swift
//  RealtimeTranscriberMac
//
//  Project data model
//

import Foundation

struct Project: Identifiable, Codable, Hashable {
    let id: Int
    var name: String
    var description: String?
    let createdAt: Date
    let updatedAt: Date
    let sessionCount: Int
    
    enum CodingKeys: String, CodingKey {
        case id
        case name
        case description
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case sessionCount = "session_count"
    }
    
    // 用于显示的格式化日期
    var formattedCreatedDate: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: createdAt)
    }
    
    var formattedUpdatedDate: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: updatedAt)
    }
}

// 用于创建新项目的请求模型
struct ProjectCreate: Codable {
    let name: String
    let description: String?
    
    init(name: String, description: String? = nil) {
        self.name = name
        self.description = description
    }
}
