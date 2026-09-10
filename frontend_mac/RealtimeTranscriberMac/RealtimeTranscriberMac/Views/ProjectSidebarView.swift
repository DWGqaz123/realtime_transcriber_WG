//
//  ProjectSidebarView.swift
//  RealtimeTranscriberMac
//
//  Sidebar view for managing projects
//

import SwiftUI

struct ProjectSidebarView: View {
    @ObservedObject var viewModel: ProjectListViewModel
    var onNewSession: ((Project) -> Void)? = nil
    @State private var showCreateSheet = false
    @State private var showDeleteConfirmation = false
    @State private var showDeleteSessionConfirmation = false
    @State private var sessionToDelete: (projectId: Int, session: RecordingSession)? = nil
    @State private var projectToDelete: Project? = nil
    @State private var showSearchSheet = false

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.lg) {
            // 品牌
            HStack(spacing: Theme.Spacing.md) {
                RoundedRectangle(cornerRadius: Theme.Radius.row)
                    .fill(Theme.accentBg)
                    .frame(width: 22, height: 22)
                    .overlay(
                        Image(systemName: "waveform")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundColor(Theme.accent)
                    )
                Text("Transcriber")
                    .font(.system(size: Theme.FontSize.medium, weight: .semibold))
                    .foregroundColor(Theme.textPrimary)
                Spacer()
            }

            // 检索是跨项目的，不需要先选中项目，所以放在项目树之上
            Button {
                showSearchSheet = true
            } label: {
                HStack(spacing: Theme.Spacing.md) {
                    Image(systemName: "magnifyingglass")
                        .font(.system(size: 11))
                        .foregroundColor(Theme.textMuted)
                    Text("Search all projects")
                        .font(.system(size: Theme.FontSize.body))
                        .foregroundColor(Theme.textMuted)
                    Spacer()
                }
                .padding(.horizontal, 11)
                .frame(height: 38)
                .frame(maxWidth: .infinity)
                .background(Theme.surface)
                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.field))
            }
            .buttonStyle(.plain)

            // 工具栏
            HStack(spacing: Theme.Spacing.sm) {
                Button {
                    showCreateSheet = true
                } label: {
                    HStack(spacing: Theme.Spacing.sm) {
                        Image(systemName: "plus")
                            .font(.system(size: 9, weight: .bold))
                        Text("New Project")
                            .font(.system(size: Theme.FontSize.small, weight: .medium))
                    }
                    .foregroundColor(Theme.accent)
                    .padding(.horizontal, 9)
                    .frame(height: 26)
                    .background(Theme.accentBg)
                    .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.row))
                }
                .buttonStyle(.plain)
                .help("Create new project")

                Button {
                    Task { await viewModel.loadProjects() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 10))
                        .foregroundColor(Theme.textFaint)
                        .frame(width: 26, height: 26)
                        .background(Theme.surface)
                        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.row))
                }
                .buttonStyle(.plain)
                .disabled(viewModel.isLoading)
                .help("Refresh projects")

                Spacer()
            }

            // Project List
            if viewModel.isLoading && viewModel.projects.isEmpty {
                VStack(spacing: Theme.Spacing.md) {
                    ProgressView().controlSize(.small)
                    Text("Loading projects…")
                        .font(.system(size: Theme.FontSize.small))
                        .foregroundColor(Theme.textFaint)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if viewModel.projects.isEmpty {
                emptyStateView
            } else {
                ScrollView {
                    LazyVStack(spacing: Theme.Spacing.sm) {
                        ForEach(viewModel.projects) { project in
                            ProjectRowExpandable(
                                project: project,
                                isSelected: viewModel.selectedProject?.id == project.id,
                                isExpanded: viewModel.expandedProjects.contains(project.id),
                                sessions: viewModel.projectSessions[project.id] ?? [],
                                selectedSessionId: viewModel.selectedSession?.id,
                                onSelectProject: {
                                    viewModel.selectProject(project)
                                },
                                onToggleExpand: {
                                    viewModel.toggleProjectExpansion(project)
                                },
                                onSelectSession: { session in
                                    viewModel.selectSession(projectId: project.id, session: session)
                                },
                                onDeleteSession: { session in
                                    sessionToDelete = (projectId: project.id, session: session)
                                    showDeleteSessionConfirmation = true
                                },
                                onDelete: {
                                    projectToDelete = project
                                    showDeleteConfirmation = true
                                },
                            onNewSession: onNewSession.map { cb in { cb(project) } }
                            )
                        }
                    }
                }
            }

            Spacer(minLength: 0)
            footerView
        }
        .padding(.horizontal, Theme.Spacing.lg)
        .padding(.vertical, Theme.Spacing.xl)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .background(Theme.panelBg)
        .sheet(isPresented: $showCreateSheet) {
            CreateProjectSheet { name, description in
                await viewModel.createProject(
                    name: name,
                    description: description.isEmpty ? "" : description  // ✅ 传递空字符串
                )
                await viewModel.loadProjects()
            }
        }
        .sheet(isPresented: $viewModel.showSessionDetail) {
            if let session = viewModel.selectedSession {
                // 🔧 在闭包外部捕获 IDs
                let sessionId = session.id
                let projectId = viewModel.selectedProject?.id
                
                SessionDetailSheet(
                    session: session,
                    isPresented: $viewModel.showSessionDetail,
                    onDeleteSession: {
                        
                        if let pid = projectId {
                            Task {
                                await viewModel.deleteSession(
                                    projectId: pid,
                                    sessionId: sessionId
                                )
                            }
                        } else {
                        }
                    },
                    onDeleteSummary: { summaryId in
                        
                        if let pid = projectId {
                            Task {
                                await viewModel.deleteSummary(
                                    projectId: pid,
                                    sessionId: sessionId,
                                    summaryId: summaryId
                                )
                            }
                        } else {
                        }
                    }
                )
            } else {
                Text("No session selected")
                    .onAppear {
                    }
            }
        }
        .sheet(isPresented: $showSearchSheet) {
            SearchView(
                onSelectSession: { projectId, sessionId in
                    // 关闭搜索界面
                    showSearchSheet = false

                    // 跳转到对应的 session（可能在另一个项目里）
                    Task {
                        await viewModel.selectSessionById(
                            projectId: projectId,
                            sessionId: sessionId
                        )
                    }
                }
            )
        }
        .alert("Delete Project", isPresented: $showDeleteConfirmation) {
            Button("Cancel", role: .cancel) {
                projectToDelete = nil
            }
            Button("Delete", role: .destructive) {
                if let project = projectToDelete {
                    Task {
                        await viewModel.deleteProject(project)
                    }
                }
                projectToDelete = nil
            }
        } message: {
            if let project = projectToDelete {
                Text("Are you sure you want to delete '\(project.name)'? This will also delete all associated recordings and cannot be undone.")
            }
        }
        .alert("Error", isPresented: $viewModel.showError) {
            Button("OK", role: .cancel) {}
        } message: {
            if let error = viewModel.errorMessage {
                Text(error)
            }
        }
        .alert("Delete Session", isPresented: $showDeleteSessionConfirmation) {
            Button("Cancel", role: .cancel) {
                sessionToDelete = nil
            }
            Button("Delete", role: .destructive) {
                if let toDelete = sessionToDelete {
                    Task {
                        await viewModel.deleteSession(
                            projectId: toDelete.projectId,
                            sessionId: toDelete.session.id
                        )
                    }
                }
                sessionToDelete = nil
            }
        } message: {
            if let toDelete = sessionToDelete {
                Text("Are you sure you want to delete this session? This will delete the transcript and all \(toDelete.session.summaries?.count ?? 0) summaries. This cannot be undone.")
            }
        }
    }
    
    // MARK: - Empty State
    
    private var emptyStateView: some View {
        EmptyState(icon: "folder.badge.plus",
                   title: "No projects yet",
                   hint: "Create one to start recording")
    }

    // MARK: - Footer
    
    private var footerView: some View {
        HStack(spacing: Theme.Spacing.sm) {
            if viewModel.isLoading {
                ProgressView().controlSize(.small).scaleEffect(0.6).frame(width: 10, height: 10)
            }
            Text("Settings")
                .font(.system(size: Theme.FontSize.small))
                .foregroundColor(Theme.textFaint)
            Spacer()
            Chip(text: "⌘,", background: Theme.surface, mono: true)
        }
        .padding(.top, Theme.Spacing.md)
        .overlay(alignment: .top) {
            Rectangle().fill(Theme.border).frame(height: 1)
        }
    }
}

// MARK: - Expandable Project Row

struct ProjectRowExpandable: View {
    let project: Project
    let isSelected: Bool
    let isExpanded: Bool
    let sessions: [RecordingSession]
    let selectedSessionId: Int?
    let onSelectProject: () -> Void
    let onToggleExpand: () -> Void
    let onSelectSession: (RecordingSession) -> Void
    let onDeleteSession: (RecordingSession) -> Void
    let onDelete: () -> Void
    var onNewSession: (() -> Void)? = nil
    
    @State private var isHovering = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xs) {
            // 项目行
            HStack(spacing: Theme.Spacing.sm) {
                Button(action: onToggleExpand) {
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundColor(Theme.textFaint)
                        .frame(width: 12, height: 12)
                }
                .buttonStyle(.plain)
                .opacity(project.sessionCount > 0 ? 1 : 0.25)
                .disabled(project.sessionCount == 0)

                VStack(alignment: .leading, spacing: 1) {
                    Text(project.name)
                        .font(.system(size: Theme.FontSize.body, weight: isSelected ? .semibold : .medium))
                        .foregroundColor(isSelected ? Theme.textPrimary : Theme.textSecondary)
                        .lineLimit(1)

                    Text("\(project.sessionCount) session\(project.sessionCount == 1 ? "" : "s")")
                        .font(.system(size: Theme.FontSize.micro))
                        .foregroundColor(Theme.textFaint)
                }

                Spacer(minLength: 0)

                if isHovering {
                    if let onNewSession {
                        Button(action: onNewSession) {
                            Image(systemName: "plus")
                                .font(.system(size: 9, weight: .bold))
                                .foregroundColor(Theme.accent)
                        }
                        .buttonStyle(.plain)
                        .help("New session in this project")
                    }
                    Button(action: onDelete) {
                        Image(systemName: "trash")
                            .font(.system(size: 9))
                            .foregroundColor(Theme.danger)
                    }
                    .buttonStyle(.plain)
                    .help("Delete project")
                }
            }
            .padding(.horizontal, 7)
            .padding(.vertical, 6)
            .contentShape(Rectangle())
            .onTapGesture { onSelectProject() }
            .onHover { isHovering = $0 }

            // 会话
            if isExpanded {
                if sessions.isEmpty {
                    Text("No sessions yet")
                        .font(.system(size: Theme.FontSize.micro))
                        .foregroundColor(Theme.textFaint)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 6)
                } else {
                    ForEach(sessions) { session in
                        SessionRowView(
                            session: session,
                            isSelected: selectedSessionId == session.id,
                            onSelect: { onSelectSession(session) },
                            onDelete: { onDeleteSession(session) }
                        )
                    }
                }

                if let onNewSession {
                    Button(action: onNewSession) {
                        HStack(spacing: Theme.Spacing.sm) {
                            Image(systemName: "plus")
                                .font(.system(size: 8, weight: .bold))
                            Text("New session")
                                .font(.system(size: Theme.FontSize.micro, weight: .medium))
                            Spacer()
                        }
                        .foregroundColor(Theme.accent)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 6)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .padding(Theme.Spacing.xs)
        .background(isExpanded || isSelected ? Theme.cardAlt : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.card))
    }
}


// MARK: - Session Row

struct SessionRowView: View {
    let session: RecordingSession
    let isSelected: Bool
    let onSelect: () -> Void
    let onDelete: () -> Void
    
    @State private var isHovering = false
    
    var body: some View {
        HStack(spacing: Theme.Spacing.sm) {
            VStack(alignment: .leading, spacing: 2) {
                Text(session.name?.isEmpty == false ? session.name! : session.modeDisplayName)
                    .font(.system(size: Theme.FontSize.small, weight: isSelected ? .semibold : .regular))
                    .foregroundColor(isSelected ? Theme.textPrimary : Theme.textSecondary)
                    .lineLimit(1)

                HStack(spacing: 5) {
                    Text(session.formattedStartDate)
                    Text("·")
                    Text(session.formattedDuration)
                    Text("·")
                    Text("\(session.sentenceCount) lines")
                }
                .font(.system(size: Theme.FontSize.micro))
                .foregroundColor(Theme.textFaint)
                .lineLimit(1)
            }

            Spacer(minLength: 0)

            if isHovering {
                Button(action: onDelete) {
                    Image(systemName: "trash")
                        .font(.system(size: 9))
                        .foregroundColor(Theme.danger)
                }
                .buttonStyle(.plain)
                .help("Delete session")
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 7)
        .background(isSelected ? Theme.rowActive : (isHovering ? Theme.surface : Color.clear))
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.row))
        .contentShape(Rectangle())
        .onTapGesture { onSelect() }
        .onHover { isHovering = $0 }
    }
}


#Preview {
    ProjectSidebarView(viewModel: ProjectListViewModel())
}

