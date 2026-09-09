import Foundation

struct ReindexResponse: Decodable {
    let success: Bool
    let indexed: Int
    let project_id: Int?
}

private struct EmptyBody: Encodable {}

final class SearchService {
    private let api = APIClient()

    /// 跨全部项目检索——记忆不按项目分区，这是默认入口
    func searchAll(query: String, topK: Int, mode: String = "hybrid") async throws -> SearchResponse {
        try await api.get(
            "api/search/all",
            queryItems: [
                URLQueryItem(name: "query", value: query),
                URLQueryItem(name: "top_k", value: String(topK)),
                URLQueryItem(name: "mode", value: mode),
            ]
        )
    }

    func reindexAll() async throws -> ReindexResponse {
        try await api.post("api/search/reindex-all", body: EmptyBody())
    }

}
