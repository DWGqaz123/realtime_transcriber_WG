import Foundation
import Combine
// MARK: - Data Models

struct SearchResult: Identifiable, Codable {
    let summary_id: Int
    let content: String
    let similarity: Double
    let session_id: Int
    let session_mode: String
    let created_at: String
    
    var id: Int { summary_id }
    
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
                topK: topK
            )
            self.results = searchResponse.results
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
