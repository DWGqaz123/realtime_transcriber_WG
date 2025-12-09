//
//  ProjectSidebarView.swift
//  RealtimeTranscriberMac
//
//  Sidebar view for managing projects
//

import SwiftUI

struct ProjectSidebarView: View {
    @ObservedObject var viewModel: ProjectListViewModel
    @State private var showCreateSheet = false
    @State private var showDeleteConfirmation = false
    @State private var showDeleteSessionConfirmation = false
    @State private var sessionToDelete: (projectId: Int, session: RecordingSession)? = nil
    @State private var projectToDelete: Project? = nil
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Projects")
                    .font(.headline)
                    .foregroundColor(.secondary)
                
                Spacer()
                
                Button(action: {
                    showCreateSheet = true
                }) {
                    Image(systemName: "plus.circle.fill")
                        .foregroundColor(.blue)
                        .font(.title3)
                }
                .buttonStyle(.plain)
                .help("Create new project")
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            
            Divider()
            
            // Project List
            if viewModel.isLoading && viewModel.projects.isEmpty {
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Loading projects...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if viewModel.projects.isEmpty {
                emptyStateView
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
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
                                onDeleteSession: { session in  // 🔧 新增
                                    sessionToDelete = (projectId: project.id, session: session)
                                    showDeleteSessionConfirmation = true
                                },
                                onDelete: {
                                    projectToDelete = project
                                    showDeleteConfirmation = true
                                }
                            )
                        }
                    }
                }
            }
            
            Divider()
            
            // Footer
            footerView
        }
        .frame(minWidth: 250, idealWidth: 300, maxWidth: 350)
        .sheet(isPresented: $viewModel.showSessionDetail) {
            if let session = viewModel.selectedSession {
                // 🔧 在闭包外部捕获 IDs
                let sessionId = session.id
                let projectId = viewModel.selectedProject?.id
                
                SessionDetailSheet(
                    session: session,
                    isPresented: $viewModel.showSessionDetail,
                    onDeleteSession: {
                        print("🔵 onDeleteSession callback in sheet")
                        print("   Captured sessionId: \(sessionId)")
                        print("   Captured projectId: \(String(describing: projectId))")
                        
                        if let pid = projectId {
                            Task {
                                await viewModel.deleteSession(
                                    projectId: pid,
                                    sessionId: sessionId
                                )
                            }
                        } else {
                            print("⚠️ No project ID captured")
                        }
                    },
                    onDeleteSummary: { summaryId in
                        print("🔵 onDeleteSummary callback in sheet")
                        print("   Captured sessionId: \(sessionId)")
                        print("   Captured projectId: \(String(describing: projectId))")
                        print("   SummaryId: \(summaryId)")
                        
                        if let pid = projectId {
                            Task {
                                await viewModel.deleteSummary(
                                    projectId: pid,
                                    sessionId: sessionId,
                                    summaryId: summaryId
                                )
                            }
                        } else {
                            print("⚠️ No project ID captured")
                        }
                    }
                )
            } else {
                Text("No session selected")
                    .onAppear {
                        print("⚠️ Sheet opened but no session selected")
                    }
            }
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
        VStack(spacing: 16) {
            Image(systemName: "folder.badge.plus")
                .font(.system(size: 48))
                .foregroundColor(.gray.opacity(0.5))
            
            Text("No Projects Yet")
                .font(.headline)
                .foregroundColor(.secondary)
            
            Text("Create your first project to get started")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            
            Button(action: {
                showCreateSheet = true
            }) {
                HStack(spacing: 6) {
                    Image(systemName: "plus.circle.fill")
                    Text("New Project")
                }
            }
            .buttonStyle(.borderedProminent)
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
    // MARK: - Footer
    
    private var footerView: some View {
        HStack {
            if viewModel.isLoading {
                ProgressView()
                    .scaleEffect(0.7)
            }
            
            Text("\(viewModel.projects.count) project\(viewModel.projects.count == 1 ? "" : "s")")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Spacer()
            
            Button(action: {
                Task {
                    await viewModel.loadProjects()
                }
            }) {
                Image(systemName: "arrow.clockwise")
                    .font(.caption)
            }
            .buttonStyle(.plain)
            .disabled(viewModel.isLoading)
            .help("Refresh projects")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
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
    
    @State private var isHovering = false
    
    var body: some View {
        VStack(spacing: 0) {
            // Project row
            HStack(spacing: 8) {
                // Expand/Collapse button
                Button(action: onToggleExpand) {
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .frame(width: 16, height: 16)
                }
                .buttonStyle(.plain)
                .opacity(project.sessionCount > 0 ? 1 : 0.3)
                .disabled(project.sessionCount == 0)
                
                // Icon
                Image(systemName: isSelected ? "folder.fill" : "folder")
                    .foregroundColor(isSelected ? .blue : .secondary)
                    .font(.title3)
                
                // Content
                VStack(alignment: .leading, spacing: 2) {
                    Text(project.name)
                        .font(.body)
                        .fontWeight(isSelected ? .semibold : .regular)
                        .lineLimit(1)
                    
                    Text("\(project.sessionCount) session\(project.sessionCount == 1 ? "" : "s")")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                // Delete button
                if isHovering {
                    Button(action: onDelete) {
                        Image(systemName: "trash")
                            .foregroundColor(.red)
                            .font(.caption)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(isSelected ? Color.blue.opacity(0.15) : Color.clear)
            )
            .contentShape(Rectangle())
            .onTapGesture {
                onSelectProject()
            }
            .onHover { hovering in
                isHovering = hovering
            }
            
            // Sessions list (when expanded)
            // Session rows (when expanded)
            if isExpanded && !sessions.isEmpty {
                ForEach(sessions) { session in
                    SessionRowView(
                        session: session,
                        isSelected: selectedSessionId == session.id,
                        onSelect: {
                            onSelectSession(session)
                        },
                        onDelete: {  // 🔧 新增
                            onDeleteSession(session)
                        }
                    )
                    
                    if sessions.isEmpty {
                        Text("No sessions yet")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .padding(.vertical, 12)
                            .padding(.leading, 52)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
        }
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
        HStack(spacing: 8) {
            // Mode icon
            Image(systemName: session.mode == "lecture" ? "book.fill" : "bubble.left.and.bubble.right.fill")
                .font(.caption)
                .foregroundColor(isSelected ? .blue : .secondary)
                .frame(width: 16)
            
            VStack(alignment: .leading, spacing: 2) {
                Text(session.modeDisplayName)
                    .font(.caption)
                    .fontWeight(isSelected ? .semibold : .regular)
                
                HStack(spacing: 4) {
                    Text(session.formattedStartDate)
                        .font(.caption2)
                    Text("•")
                        .font(.caption2)
                    Text(session.formattedDuration)
                        .font(.caption2)
                }
                .foregroundColor(.secondary)
            }
            
            Spacer()
            if isHovering {
                            Button(action: onDelete) {
                                Image(systemName: "trash")
                                    .foregroundColor(.red)
                                    .font(.caption)
                            }
                            .buttonStyle(.plain)
                            .help("Delete session")
                        }
            
            // Sentence count
            Text("\(session.sentenceCount)")
                .font(.caption2)
                .foregroundColor(.secondary)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.gray.opacity(0.2))
                )
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .padding(.leading, 36)
        .background(
            RoundedRectangle(cornerRadius: 4)
                .fill(isSelected ? Color.blue.opacity(0.1) : (isHovering ? Color.gray.opacity(0.05) : Color.clear))
        )
        .contentShape(Rectangle())
        .onTapGesture {
            onSelect()
        }
        .onHover { hovering in
            isHovering = hovering
        }
    }
}

#Preview {
    ProjectSidebarView(viewModel: ProjectListViewModel())
}

