import Foundation

struct ReindexResponse: Decodable {
    let success: Bool
    let indexed: Int
    let project_id: Int?
}

private struct EmptyBody: Encodable {}

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

    /// 清空并重建项目索引。换 embedding 模型后，旧向量与新查询不在同一空间，
    /// 必须重建一次才能搜到历史摘要。
    func reindex(projectId: Int) async throws -> ReindexResponse {
        try await api.post("api/search/projects/\(projectId)/reindex", body: EmptyBody())
    }
}
