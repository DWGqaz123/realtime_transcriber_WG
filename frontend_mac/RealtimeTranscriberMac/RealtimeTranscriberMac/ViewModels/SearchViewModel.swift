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
        let formatter = ISO8601DateFormatter()
        return formatter.date(from: created_at)
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
    
    private let searchService = SearchService()
    
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
