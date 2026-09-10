import SwiftUI

struct SearchView: View {
    // 跨全部项目检索，因此不绑定 project；跳转时从结果里取 projectId
    let onSelectSession: (Int, Int) -> Void
    @StateObject private var viewModel = SearchViewModel()
    @Environment(\.dismiss) var dismiss

    var body: some View {
        VStack(spacing: 0) {
            headerView
            Rectangle().fill(Theme.border).frame(height: 1)
            searchBoxView
            Rectangle().fill(Theme.border).frame(height: 1)

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
        .frame(minWidth: 620, minHeight: 520)
        .background(Theme.contentBg)
    }

    // MARK: - Header

    private var headerView: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Search")
                    .font(.system(size: Theme.FontSize.title, weight: .semibold))
                    .foregroundColor(Theme.textPrimary)
                Text("All projects")
                    .font(.system(size: Theme.FontSize.small))
                    .foregroundColor(Theme.textFaint)
            }

            Spacer()

            IconButton(icon: "xmark") { dismiss() }
        }
        .padding(.horizontal, Theme.Spacing.xl)
        .padding(.vertical, Theme.Spacing.lg)
    }

    // MARK: - Search box

    private var searchBoxView: some View {
        VStack(spacing: Theme.Spacing.md) {
            HStack(spacing: Theme.Spacing.md) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 11))
                    .foregroundColor(Theme.textFaint)

                TextField("Search summaries…", text: $viewModel.searchQuery)
                    .textFieldStyle(.plain)
                    .font(.system(size: Theme.FontSize.medium))
                    .foregroundColor(Theme.textPrimary)
                    .onSubmit { Task { await viewModel.search() } }

                if !viewModel.searchQuery.isEmpty {
                    Button(action: { viewModel.clearSearch() }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 11))
                            .foregroundColor(Theme.textFaint)
                    }
                    .buttonStyle(.plain)
                }

                Button {
                    Task { await viewModel.search() }
                } label: {
                    Text("Search")
                        .font(.system(size: Theme.FontSize.body, weight: .medium))
                        .foregroundColor(Theme.accent)
                        .padding(.horizontal, 12)
                        .frame(height: 26)
                        .background(Theme.accentBg)
                        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.row))
                }
                .buttonStyle(.plain)
                .disabled(viewModel.searchQuery.trimmingCharacters(in: .whitespaces).isEmpty || viewModel.isSearching)
            }
            .padding(.horizontal, 11)
            .frame(height: 38)
            .background(Theme.surface)
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.field))

            HStack(spacing: Theme.Spacing.md) {
                // 三种模式：hybrid 融合两路，另两个用于对比与排查
                SegmentedControl(
                    options: SearchMode.allCases,
                    selection: $viewModel.mode,
                    label: { $0.label },
                    height: 24,
                    onChange: {
                        guard !viewModel.searchQuery.trimmingCharacters(in: .whitespaces).isEmpty else { return }
                        Task { await viewModel.search() }
                    }
                )

                if viewModel.hasSearched {
                    Text("semantic \(viewModel.lastSemanticHits) · keyword \(viewModel.lastKeywordHits)")
                        .font(.system(size: Theme.FontSize.micro, design: .monospaced))
                        .foregroundColor(Theme.textFaint)
                }

                Spacer()
            }
        }
        .padding(.horizontal, Theme.Spacing.xl)
        .padding(.vertical, Theme.Spacing.lg)
    }

    // MARK: - States

    private var resultsListView: some View {
        ScrollView {
            LazyVStack(spacing: Theme.Spacing.md) {
                ForEach(viewModel.results) { result in
                    SearchResultCard(result: result) {
                        onSelectSession(result.project_id, result.session_id)
                    }
                }
            }
            .padding(Theme.Spacing.xl)
        }
    }

    private var loadingView: some View {
        VStack(spacing: Theme.Spacing.md) {
            ProgressView().controlSize(.small)
            Text("Searching…")
                .font(.system(size: Theme.FontSize.body))
                .foregroundColor(Theme.textFaint)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func errorView(_ error: String) -> some View {
        EmptyState(icon: "exclamationmark.triangle", title: "Search error", hint: error, maxHintWidth: 360)
    }

    private var emptyResultsView: some View {
        EmptyState(
            icon: "doc.text.magnifyingglass",
            title: "No results",
            hint: "Try different wording, or switch to Keyword mode for exact terms."
        ) {
            VStack(spacing: Theme.Spacing.sm) {
                Text("Summaries recorded before an embedding-model change stay unsearchable until the index is rebuilt.")
                    .font(.system(size: Theme.FontSize.micro))
                    .foregroundColor(Theme.textFaint)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 330)

                Button {
                    Task { await viewModel.reindexAll() }
                } label: {
                    HStack(spacing: Theme.Spacing.sm) {
                        if viewModel.isReindexing {
                            ProgressView().controlSize(.small).scaleEffect(0.6).frame(width: 10, height: 10)
                        }
                        Text(viewModel.isReindexing ? "Rebuilding…" : "Rebuild Index")
                    }
                }
                .buttonStyle(ThemedPrimaryButtonStyle())
                .disabled(viewModel.isReindexing)

                if let status = viewModel.statusMessage {
                    Text(status)
                        .font(.system(size: Theme.FontSize.micro))
                        .foregroundColor(Theme.textFaint)
                }
            }
            .padding(.top, Theme.Spacing.sm)
        }
    }

    private var placeholderView: some View {
        EmptyState(
            icon: "magnifyingglass",
            title: "Search your notes",
            hint: "Hybrid mode combines meaning with exact keywords across every project."
        )
    }
}

// MARK: - Result card

struct SearchResultCard: View {
    let result: SearchResult
    let onTap: () -> Void
    @State private var isHovering = false

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: Theme.Spacing.md) {
                HStack(spacing: Theme.Spacing.sm) {
                    // 语义命中显示相似度；BM25 没有直观量纲，只标出处
                    if result.matchedSemantic || result.sources == nil {
                        Chip(text: result.formattedSimilarity, icon: "sparkles", tint: Theme.accent)
                    }
                    if result.matchedKeyword {
                        Chip(text: "keyword", icon: "text.magnifyingglass", tint: Theme.violet)
                    }

                    Spacer()

                    if !result.project_name.isEmpty {
                        Chip(text: result.project_name, background: Theme.surface)
                    }
                    Text(result.formattedDate)
                        .font(.system(size: Theme.FontSize.micro))
                        .foregroundColor(Theme.textFaint)
                }

                Text(result.content)
                    .font(.system(size: Theme.FontSize.body))
                    .foregroundColor(Theme.textSecondary)
                    .lineLimit(4)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)

                HStack(spacing: 4) {
                    Text("Go to session #\(result.session_id)")
                        .font(.system(size: Theme.FontSize.micro, weight: .medium))
                    Image(systemName: "arrow.right")
                        .font(.system(size: 8, weight: .bold))
                }
                .foregroundColor(Theme.accent)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 12)
            .padding(.vertical, 11)
            .background(isHovering ? Theme.rowActive : Theme.card)
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.card))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.card)
                    .stroke(isHovering ? Theme.accent.opacity(0.35) : Theme.border, lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .onHover { isHovering = $0 }
    }

}

#Preview {
    SearchView(onSelectSession: { _, _ in })
}
