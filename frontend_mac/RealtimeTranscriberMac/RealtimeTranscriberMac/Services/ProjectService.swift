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
    private let baseURL = "http://127.0.0.1:8000/api/projects"
    
    // 🔧 创建一个配置好的 JSONDecoder
    private func createDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()

        // 提前创建好 formatter，避免每次都 new
        let iso8601WithTZAndFraction = ISO8601DateFormatter()
        iso8601WithTZAndFraction.formatOptions = [.withInternetDateTime, .withFractionalSeconds]

        let iso8601WithTZ = ISO8601DateFormatter()
        iso8601WithTZ.formatOptions = [.withInternetDateTime]

        let noTZWithFraction = DateFormatter()
        noTZWithFraction.locale = Locale(identifier: "en_US_POSIX")
        noTZWithFraction.timeZone = TimeZone(secondsFromGMT: 0)
        noTZWithFraction.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"

        let noTZNoFraction = DateFormatter()
        noTZNoFraction.locale = Locale(identifier: "en_US_POSIX")
        noTZNoFraction.timeZone = TimeZone(secondsFromGMT: 0)
        noTZNoFraction.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"

        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let dateString = try container.decode(String.self)

            // 1) 有时区 + 小数秒
            if let date = iso8601WithTZAndFraction.date(from: dateString) {
                return date
            }

            // 2) 有时区，无小数秒
            if let date = iso8601WithTZ.date(from: dateString) {
                return date
            }

            // 3) 无时区 + 小数秒（就是你现在的这种）
            if let date = noTZWithFraction.date(from: dateString) {
                return date
            }

            // 4) 无时区，无小数秒
            if let date = noTZNoFraction.date(from: dateString) {
                return date
            }

            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Cannot decode date string: \(dateString)"
            )
        }

        return decoder
    }
    
    // MARK: - Fetch all projects
    
    func fetchProjects() async throws -> [Project] {
        guard let url = URL(string: baseURL) else {
            throw ProjectServiceError.invalidURL
        }
        
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw ProjectServiceError.unknown
            }
            
            guard (200...299).contains(httpResponse.statusCode) else {
                throw ProjectServiceError.serverError(statusCode: httpResponse.statusCode)
            }
            
            // 🔧 打印原始 JSON（调试用）
            if let jsonString = String(data: data, encoding: .utf8) {
                print("📥 Received JSON: \(jsonString.prefix(200))...")
            }
            
            // 🔧 使用配置好的 decoder
            let decoder = createDecoder()
            let projects = try decoder.decode([Project].self, from: data)
            
            print("✅ Successfully decoded \(projects.count) projects")
            return projects
            
        } catch let error as ProjectServiceError {
            throw error
        } catch let error as DecodingError {
            print("❌ Decoding error details: \(error)")
            throw ProjectServiceError.decodingError(error)
        } catch {
            throw ProjectServiceError.networkError(error)
        }
    }
    
    // MARK: - Create project
    
    func createProject(name: String, description: String = "") async throws -> Project {
        let urlString = "\(baseURL)/"  
        print("📤 Creating project at URL: \(urlString)")
        
        guard let url = URL(string: urlString) else {
            throw ProjectServiceError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let projectCreate = ProjectCreate(name: name, description: description.isEmpty ? nil : description)
        
        do {
            let encoder = JSONEncoder()
            request.httpBody = try encoder.encode(projectCreate)
            
            let (data, response) = try await URLSession.shared.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw ProjectServiceError.unknown
            }
            
            guard (200...299).contains(httpResponse.statusCode) else {
                throw ProjectServiceError.serverError(statusCode: httpResponse.statusCode)
            }
            
            // 🔧 使用配置好的 decoder
            let decoder = createDecoder()
            let project = try decoder.decode(Project.self, from: data)
            return project
            
        } catch let error as ProjectServiceError {
            throw error
        } catch let error as DecodingError {
            throw ProjectServiceError.decodingError(error)
        } catch {
            throw ProjectServiceError.networkError(error)
        }
    }
    
    // MARK: - Delete project
    
    func deleteProject(id: Int) async throws {
        guard let url = URL(string: "\(baseURL)/\(id)") else {
            throw ProjectServiceError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw ProjectServiceError.unknown
            }
            
            guard (200...299).contains(httpResponse.statusCode) else {
                throw ProjectServiceError.serverError(statusCode: httpResponse.statusCode)
            }
            
        } catch let error as ProjectServiceError {
            throw error
        } catch {
            throw ProjectServiceError.networkError(error)
        }
    }
    
    // MARK: - Fetch project sessions
    
    func fetchProjectSessions(projectId: Int) async throws -> [RecordingSession] {
        guard let url = URL(string: "\(baseURL)/\(projectId)/sessions") else {
            throw ProjectServiceError.invalidURL
        }
        
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw ProjectServiceError.unknown
            }
            
            guard (200...299).contains(httpResponse.statusCode) else {
                throw ProjectServiceError.serverError(statusCode: httpResponse.statusCode)
            }
            
            // 🔧 使用配置好的 decoder
            let decoder = createDecoder()
            let sessions = try decoder.decode([RecordingSession].self, from: data)
            return sessions
            
        } catch let error as ProjectServiceError {
            throw error
        } catch let error as DecodingError {
            throw ProjectServiceError.decodingError(error)
        } catch {
            throw ProjectServiceError.networkError(error)
        }
    }
    
    // MARK: - Fetch session detail

    func fetchSessionDetail(projectId: Int, sessionId: Int) async throws -> RecordingSession {
        guard let url = URL(string: "\(baseURL)/\(projectId)/sessions/\(sessionId)") else {
            throw ProjectServiceError.invalidURL
        }
        
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw ProjectServiceError.unknown
            }
            
            guard (200...299).contains(httpResponse.statusCode) else {
                throw ProjectServiceError.serverError(statusCode: httpResponse.statusCode)
            }
            
            // 🔧 打印调试信息
            if let jsonString = String(data: data, encoding: .utf8) {
                print("📥 Session detail JSON: \(jsonString.prefix(200))...")
            }
            
            let decoder = createDecoder()
            let session = try decoder.decode(RecordingSession.self, from: data)
            
            print("✅ Successfully fetched session detail: \(session.id)")
            return session
            
        } catch let error as ProjectServiceError {
            throw error
        } catch let error as DecodingError {
            print("❌ Decoding error: \(error)")
            throw ProjectServiceError.decodingError(error)
        } catch {
            throw ProjectServiceError.networkError(error)
        }
    }
    
    // MARK: - Delete session

    /// Delete a session
    func deleteSession(projectId: Int, sessionId: Int) async throws {
        guard let url = URL(string: "\(baseURL)/\(projectId)/sessions/\(sessionId)") else {
            throw ProjectServiceError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        
        print("🗑️ DELETE \(url.absoluteString)")
        
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw ProjectServiceError.unknown
            }
            
            print("📥 Delete session response: \(httpResponse.statusCode)")
            
            guard (200...299).contains(httpResponse.statusCode) else {
                throw ProjectServiceError.serverError(statusCode: httpResponse.statusCode)
            }
            
            print("✅ Successfully deleted session \(sessionId)")
            
        } catch let error as ProjectServiceError {
            throw error
        } catch {
            throw ProjectServiceError.networkError(error)
        }
    }
    
    // MARK: - Delete summary

    /// Delete a summary
    func deleteSummary(projectId: Int, sessionId: Int, summaryId: Int) async throws {
        guard let url = URL(string: "\(baseURL)/\(projectId)/sessions/\(sessionId)/summaries/\(summaryId)") else {
            throw ProjectServiceError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"
        
        print("🗑️ DELETE \(url.absoluteString)")
        
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw ProjectServiceError.unknown
            }
            
            print("📥 Delete summary response: \(httpResponse.statusCode)")
            
            guard (200...299).contains(httpResponse.statusCode) else {
                throw ProjectServiceError.serverError(statusCode: httpResponse.statusCode)
            }
            
            print("✅ Successfully deleted summary \(summaryId)")
            
        } catch let error as ProjectServiceError {
            throw error
        } catch {
            throw ProjectServiceError.networkError(error)
        }
    }
    
    // MARK: - Get single project
    
    func getProject(id: Int) async throws -> Project {
        guard let url = URL(string: "\(baseURL)/\(id)") else {
            throw ProjectServiceError.invalidURL
        }
        
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw ProjectServiceError.unknown
            }
            
            guard (200...299).contains(httpResponse.statusCode) else {
                throw ProjectServiceError.serverError(statusCode: httpResponse.statusCode)
            }
            
            // 🔧 使用配置好的 decoder
            let decoder = createDecoder()
            let project = try decoder.decode(Project.self, from: data)
            return project
            
        } catch let error as ProjectServiceError {
            throw error
        } catch let error as DecodingError {
            throw ProjectServiceError.decodingError(error)
        } catch {
            throw ProjectServiceError.networkError(error)
        }
    }
}
