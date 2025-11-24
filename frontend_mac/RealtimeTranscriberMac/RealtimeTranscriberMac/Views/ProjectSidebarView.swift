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
    @State private var projectToDelete: Project?
    
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
            } else {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(viewModel.projects) { project in
                            ProjectRowView(
                                project: project,
                                isSelected: viewModel.selectedProject?.id == project.id,
                                onSelect: {
                                    viewModel.selectProject(project)
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
        .frame(minWidth: 200, idealWidth: 250, maxWidth: 300)
        .sheet(isPresented: $showCreateSheet) {
            CreateProjectSheet { name, description in
                await viewModel.createProject(name: name, description: description)
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
    }
}

// MARK: - Project Row View

struct ProjectRowView: View {
    let project: Project
    let isSelected: Bool
    let onSelect: () -> Void
    let onDelete: () -> Void
    
    @State private var isHovering = false
    
    var body: some View {
        HStack(spacing: 12) {
            // Icon
            Image(systemName: isSelected ? "folder.fill" : "folder")
                .foregroundColor(isSelected ? .blue : .secondary)
                .font(.title3)
            
            // Content
            VStack(alignment: .leading, spacing: 4) {
                Text(project.name)
                    .font(.body)
                    .fontWeight(isSelected ? .semibold : .regular)
                    .lineLimit(1)
                
                HStack(spacing: 4) {
                    Text("\(project.sessionCount)")
                        .font(.caption2)
                    Text("session\(project.sessionCount == 1 ? "" : "s")")
                        .font(.caption2)
                }
                .foregroundColor(.secondary)
            }
            
            Spacer()
            
            // Delete button (shown on hover)
            if isHovering {
                Button(action: onDelete) {
                    Image(systemName: "trash")
                        .foregroundColor(.red)
                        .font(.caption)
                }
                .buttonStyle(.plain)
                .help("Delete project")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(isSelected ? Color.blue.opacity(0.15) : Color.clear)
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

