import SwiftUI

struct SearchView: View {
    // 跨全部项目检索，因此不绑定 project；跳转时从结果里取 projectId
    let onSelectSession: (Int, Int) -> Void
    @StateObject private var viewModel = SearchViewModel()
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            headerView
            
            Divider()
            
            // Search Box
            searchBoxView
            
            Divider()
            
            // Results
            if viewModel.isSearching {
                loadingView
            } else if let error = viewModel.errorMessage {
                errorView(error)
            } else if viewModel.hasSearched && viewModel.results.isEmpty {
                emptyResultsView
            } else if !viewModel.results.isEmpty {
                resultsListView
            } else {
                placeholderView
            }
        }
        .frame(minWidth: 600, minHeight: 500)
    }
    
    // MARK: - Header
    
    private var headerView: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Search")
                    .font(.headline)

                Text("All projects")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            Button(action: { dismiss() }) {
                Image(systemName: "xmark.circle.fill")
                    .foregroundColor(.secondary)
                    .font(.title3)
            }
            .buttonStyle(.plain)
        }
        .padding()
    }
    
    // MARK: - Search Box
    
    private var searchBoxView: some View {
        VStack(spacing: 10) {
        HStack(spacing: 12) {
            Image(systemName: "magnifyingglass")
                .foregroundColor(.secondary)
            
            TextField("Search summaries...", text: $viewModel.searchQuery)
                .textFieldStyle(.plain)
                .onSubmit {
                    Task { await viewModel.search() }
                }
            
            if !viewModel.searchQuery.isEmpty {
                Button(action: {
                    viewModel.clearSearch()
                }) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            }
            
            Button(action: {
                Task { await viewModel.search() }
            }) {
                Text("Search")
                    .frame(minWidth: 80)
            }
            .buttonStyle(.borderedProminent)
            .disabled(viewModel.searchQuery.trimmingCharacters(in: .whitespaces).isEmpty || viewModel.isSearching)
        }
        .padding(.horizontal)
        .padding(.top)

        HStack(spacing: 12) {
            Picker("", selection: $viewModel.mode) {
                ForEach(SearchMode.allCases) { m in
                    Text(m.label).tag(m)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .frame(width: 260)
            .onChange(of: viewModel.mode) { _ in
                guard !viewModel.searchQuery.trimmingCharacters(in: .whitespaces).isEmpty else { return }
                Task { await viewModel.search() }
            }

            if viewModel.hasSearched {
                Text("语义 \(viewModel.lastSemanticHits) · 关键词 \(viewModel.lastKeywordHits)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding(.horizontal)
        .padding(.bottom, 12)
        }
        .background(Color.gray.opacity(0.03))
    }
    
    // MARK: - Results List
    
    private var resultsListView: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Results header
            HStack {
                Text("\(viewModel.results.count) results for '\(viewModel.searchQuery)'")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                
                Spacer()
            }
            .padding(.horizontal)
            .padding(.top, 12)
            
            // Results
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(viewModel.results) { result in
                        SearchResultCard(
                            result: result,
                            onTap: {  // 🔧 新增回调
                                onSelectSession(result.project_id, result.session_id)
                                dismiss()  // 关闭搜索界面
                            }
                        )
                    }
                }
                .padding()
            }
        }
    }
    
    // MARK: - States
    
    private var loadingView: some View {
        VStack(spacing: 16) {
            ProgressView()
            Text("Searching...")
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
    private func errorView(_ error: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 48))
                .foregroundColor(.orange)
            
            Text("Search Error")
                .font(.headline)
            
            Text(error)
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
    private var emptyResultsView: some View {
        VStack(spacing: 16) {
            Image(systemName: "doc.text.magnifyingglass")
                .font(.system(size: 48))
                .foregroundColor(.gray.opacity(0.5))
            
            Text("No Results Found")
                .font(.headline)
            
            Text("Try different keywords or check your spelling")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            Divider()
                .frame(maxWidth: 240)

            VStack(spacing: 6) {
                Text("Summaries recorded before an embedding-model change are not searchable until the index is rebuilt. This rebuilds every project.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 360)

                Button {
                    Task { await viewModel.reindexAll() }
                } label: {
                    if viewModel.isReindexing {
                        HStack(spacing: 6) {
                            ProgressView().controlSize(.small)
                            Text("Rebuilding index...")
                        }
                    } else {
                        Text("Rebuild Index")
                    }
                }
                .disabled(viewModel.isReindexing)

                if let status = viewModel.statusMessage {
                    Text(status)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
    private var placeholderView: some View {
        VStack(spacing: 16) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 48))
                .foregroundColor(.gray.opacity(0.5))
            
            Text("Semantic Search")
                .font(.headline)
            
            Text("Search through your summaries using natural language")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Try searching for:")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                ForEach(["深度学习", "Transformer", "RAG"], id: \.self) { example in
                    Button(action: {
                        viewModel.searchQuery = example
                        Task {
                            await viewModel.search()
                        }
                    }) {
                        HStack {
                            Image(systemName: "sparkles")
                                .font(.caption)
                            Text(example)
                                .font(.caption)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(8)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - Search Result Card

// MARK: - Search Result Card

struct SearchResultCard: View {
    let result: SearchResult
    let onTap: () -> Void  // 🔧 新增回调
    
    var body: some View {
        Button(action: onTap) {  // 🔧 包裹在 Button 中
            VStack(alignment: .leading, spacing: 12) {
                // Header with similarity score
                HStack(spacing: 6) {
                    // 语义命中：显示余弦相似度
                    if result.matchedSemantic || result.sources == nil {
                        HStack(spacing: 4) {
                            Image(systemName: "sparkles")
                                .font(.caption2)
                            Text(result.formattedSimilarity)
                                .font(.caption)
                                .fontWeight(.medium)
                        }
                        .foregroundColor(.orange)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.orange.opacity(0.1))
                        .cornerRadius(6)
                    }

                    // 关键词命中：BM25 没有直观量纲，只标出处
                    if result.matchedKeyword {
                        HStack(spacing: 4) {
                            Image(systemName: "text.magnifyingglass")
                                .font(.caption2)
                            Text("keyword")
                                .font(.caption)
                                .fontWeight(.medium)
                        }
                        .foregroundColor(.blue)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(6)
                    }
                    
                    Spacer()
                    
                    // Metadata
                    HStack(spacing: 8) {
                        if !result.project_name.isEmpty {
                            Text(result.project_name)
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.secondary.opacity(0.12))
                                .cornerRadius(4)
                        }

                        Text(result.formattedDate)
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Text("•")
                            .foregroundColor(.secondary)
                        
                        Label(result.session_mode.capitalized, systemImage: "doc.text")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                // Content
                Text(result.content)
                    .font(.body)
                    .foregroundColor(.primary)  // 🔧 确保文本颜色正确
                    .lineLimit(5)
                    .multilineTextAlignment(.leading)  // 🔧 左对齐
                
                // Session link
                HStack {
                    Image(systemName: "arrow.right.circle.fill")  // 🔧 改为填充图标
                        .font(.caption)
                    Text("Go to Session #\(result.session_id)")  // 🔧 更明确的文字
                        .font(.caption)
                        .foregroundColor(.blue)
                }
            }
            .padding()
            .background(Color(NSColor.controlBackgroundColor))
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.gray.opacity(0.12), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)  // 🔧 使用 plain 样式避免默认按钮样式
        .contentShape(Rectangle())  // 🔧 整个卡片可点击
        .onHover { isHovered in  // 🔧 添加悬停效果
            if isHovered {
                NSCursor.pointingHand.push()
            } else {
                NSCursor.pop()
            }
        }
    }
}

#Preview {
    SearchView(onSelectSession: { _, _ in })
}
