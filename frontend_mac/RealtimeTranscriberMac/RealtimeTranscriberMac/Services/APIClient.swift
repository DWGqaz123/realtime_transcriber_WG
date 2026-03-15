import Foundation

final class APIClient {
    private let session: URLSession
    private let decoder: JSONDecoder

    private var baseURL: URL { AppConfig.apiBaseURL }

    init(session: URLSession = .shared) {
        self.session = session
        self.decoder = APIClient.makeDecoder()
    }

    func get<T: Decodable>(_ path: String, queryItems: [URLQueryItem] = []) async throws -> T {
        try await request(path: path, method: "GET", queryItems: queryItems)
    }

    func post<T: Decodable, Body: Encodable>(_ path: String, body: Body) async throws -> T {
        let bodyData: Data
        do {
            bodyData = try JSONEncoder().encode(body)
        } catch {
            throw ProjectServiceError.networkError(error)
        }
        return try await request(path: path, method: "POST", body: bodyData)
    }

    func delete(_ path: String) async throws {
        let _: EmptyResponse = try await request(path: path, method: "DELETE")
    }

    private func request<T: Decodable>(
        path: String,
        method: String,
        body: Data? = nil,
        queryItems: [URLQueryItem] = []
    ) async throws -> T {
        guard var components = URLComponents(url: baseURL.appendingPathComponent(path), resolvingAgainstBaseURL: false) else {
            throw ProjectServiceError.invalidURL
        }
        if !queryItems.isEmpty {
            components.queryItems = queryItems
        }
        guard let url = components.url else {
            throw ProjectServiceError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = method
        if let body {
            request.httpBody = body
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }

        do {
            let (data, response) = try await session.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw ProjectServiceError.unknown
            }
            guard (200...299).contains(httpResponse.statusCode) else {
                throw ProjectServiceError.serverError(statusCode: httpResponse.statusCode)
            }

            if T.self == EmptyResponse.self {
                return EmptyResponse() as! T
            }
            return try decoder.decode(T.self, from: data)
        } catch let error as ProjectServiceError {
            throw error
        } catch let error as DecodingError {
            throw ProjectServiceError.decodingError(error)
        } catch {
            throw ProjectServiceError.networkError(error)
        }
    }

    private static let iso8601WithTZAndFraction: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    private static let iso8601WithTZ: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f
    }()

    private static let noTZWithFraction: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(secondsFromGMT: 0)
        f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
        return f
    }()

    private static let noTZNoFraction: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(secondsFromGMT: 0)
        f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
        return f
    }()

    private static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let dateString = try container.decode(String.self)

            if let date = iso8601WithTZAndFraction.date(from: dateString) { return date }
            if let date = iso8601WithTZ.date(from: dateString) { return date }
            if let date = noTZWithFraction.date(from: dateString) { return date }
            if let date = noTZNoFraction.date(from: dateString) { return date }

            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "Cannot decode date string: \(dateString)"
            )
        }
        return decoder
    }
}

private struct EmptyResponse: Decodable {}
