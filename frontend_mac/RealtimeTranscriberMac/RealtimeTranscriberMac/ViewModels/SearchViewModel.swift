import Foundation
import Combine
// MARK: - Data Models

enum SearchMode: String, CaseIterable, Identifiable {
    case hybrid, semantic, keyword
    var id: String { rawValue }
    var label: String {
        switch self {
        case .hybrid: return "Hybrid"
        case .semantic: return "Semantic"
        case .keyword: return "Keyword"
        }
    }
}

struct SearchResult: Identifiable, Codable {
    let summary_id: Int
    let content: String
    let similarity: Double
    let session_id: Int
    let session_mode: String
    let created_at: String
    let score: Double?
    let bm25: Double?
    let sources: [String]?

    var id: Int { summary_id }

    var matchedSemantic: Bool { sources?.contains("semantic") ?? false }
    var matchedKeyword: Bool { sources?.contains("keyword") ?? false }
    
    var formattedSimilarity: String {
        String(format: "%.0f%%", similarity * 100)
    }
    
    var createdDate: Date? {
        // 后端发的是无时区的 isoformat()，纯 ISO8601DateFormatter 解析不了
        Summary.parseTimestamp(created_at)
    }
    
    var formattedDate: String {
        guard let date = createdDate else { return "" }
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: date, relativeTo: Date())
    }
}

struct SearchResponse: Codable {
    let query: String
    let total: Int
    let results: [SearchResult]
    let mode: String?
    let semantic_hits: Int?
    let keyword_hits: Int?
}

// MARK: - ViewModel

@MainActor
class SearchViewModel: ObservableObject {
    @Published var searchQuery: String = ""
    @Published var results: [SearchResult] = []
    @Published var isSearching: Bool = false
    @Published var errorMessage: String?
    @Published var hasSearched: Bool = false
    @Published var isReindexing: Bool = false
    @Published var statusMessage: String?
    @Published var mode: SearchMode = .hybrid
    @Published var lastSemanticHits: Int = 0
    @Published var lastKeywordHits: Int = 0

    private let searchService = SearchService()

    /// 重建项目向量索引，然后重跑当前查询
    func reindex(projectId: Int) async {
        isReindexing = true
        errorMessage = nil
        statusMessage = "Rebuilding index..."

        do {
            let response = try await searchService.reindex(projectId: projectId)
            statusMessage = "Indexed \(response.indexed) summaries"
            isReindexing = false
            if !searchQuery.trimmingCharacters(in: .whitespaces).isEmpty {
                await search(in: projectId)
            }
        } catch {
            statusMessage = nil
            errorMessage = "Reindex failed: \(error.localizedDescription)"
            isReindexing = false
        }
    }
    
    func search(in projectId: Int, topK: Int = 10) async {
        guard !searchQuery.trimmingCharacters(in: .whitespaces).isEmpty else {
            errorMessage = "Please enter a search query"
            return
        }
        
        isSearching = true
        errorMessage = nil
        hasSearched = true
        results = []
        
        
        do {
            let searchResponse = try await searchService.search(
                projectId: projectId,
                query: searchQuery.trimmingCharacters(in: .whitespaces),
                topK: topK,
                mode: mode.rawValue
            )
            self.results = searchResponse.results
            self.lastSemanticHits = searchResponse.semantic_hits ?? 0
            self.lastKeywordHits = searchResponse.keyword_hits ?? 0
        } catch {
            errorMessage = "Search failed: \(error.localizedDescription)"
        }
        
        isSearching = false
    }
    
    func clearSearch() {
        searchQuery = ""
        results = []
        hasSearched = false
        errorMessage = nil
    }
}
