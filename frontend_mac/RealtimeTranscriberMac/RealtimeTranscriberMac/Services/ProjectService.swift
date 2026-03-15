//ProjectService.swift
import Foundation

enum ProjectServiceError: LocalizedError {
    case invalidURL
    case networkError(Error)
    case decodingError(Error)
    case serverError(statusCode: Int)
    case unknown
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid API URL"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .decodingError(let error):
            return "Failed to decode response: \(error.localizedDescription)"
        case .serverError(let statusCode):
            return "Server error: HTTP \(statusCode)"
        case .unknown:
            return "Unknown error occurred"
        }
    }
}

class ProjectService {
    private let api = APIClient()
    
    // MARK: - Fetch all projects
    
    func fetchProjects() async throws -> [Project] {
        try await api.get("api/projects")
    }
    
    // MARK: - Create project
    
    func createProject(name: String, description: String = "") async throws -> Project {
        let projectCreate = ProjectCreate(name: name, description: description.isEmpty ? nil : description)
        return try await api.post("api/projects/", body: projectCreate)
    }
    
    // MARK: - Delete project
    
    func deleteProject(id: Int) async throws {
        try await api.delete("api/projects/\(id)")
    }
    
    // MARK: - Fetch project sessions
    
    func fetchProjectSessions(projectId: Int) async throws -> [RecordingSession] {
        try await api.get("api/projects/\(projectId)/sessions")
    }
    
    // MARK: - Fetch session detail

    func fetchSessionDetail(projectId: Int, sessionId: Int) async throws -> RecordingSession {
        try await api.get("api/projects/\(projectId)/sessions/\(sessionId)")
    }
    
    // MARK: - Delete session

    /// Delete a session
    func deleteSession(projectId: Int, sessionId: Int) async throws {
        try await api.delete("api/projects/\(projectId)/sessions/\(sessionId)")
    }
    
    // MARK: - Delete summary

    /// Delete a summary
    func deleteSummary(projectId: Int, sessionId: Int, summaryId: Int) async throws {
        try await api.delete("api/projects/\(projectId)/sessions/\(sessionId)/summaries/\(summaryId)")
    }
    
    // MARK: - Get single project
    
    func getProject(id: Int) async throws -> Project {
        try await api.get("api/projects/\(id)")
    }
}
