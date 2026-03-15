import Foundation

final class SearchService {
    private let api = APIClient()

    func search(projectId: Int, query: String, topK: Int) async throws -> SearchResponse {
        try await api.get(
            "api/search/projects/\(projectId)",
            queryItems: [
                URLQueryItem(name: "query", value: query),
                URLQueryItem(name: "top_k", value: String(topK)),
            ]
        )
    }
}
