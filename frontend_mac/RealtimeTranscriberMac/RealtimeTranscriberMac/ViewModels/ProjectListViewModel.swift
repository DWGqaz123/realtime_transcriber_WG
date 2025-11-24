//
//  ProjectListViewModel.swift
//  RealtimeTranscriberMac
//
//  ViewModel for managing project list
//

import Foundation
import SwiftUI
import Combine

@MainActor
class ProjectListViewModel: ObservableObject {
    // MARK: - Published Properties
    
    @Published var projects: [Project] = []
    @Published var selectedProject: Project?
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?
    @Published var showError: Bool = false
    
    // MARK: - Private Properties
    
    private let projectService = ProjectService()
    
    // MARK: - Initialization
    
    init() {
        Task {
            await loadProjects()
        }
    }
    
    // MARK: - Public Methods
    
    /// Load all projects from server
    func loadProjects() async {
        isLoading = true
        errorMessage = nil
        
        do {
            let fetchedProjects = try await projectService.fetchProjects()
            self.projects = fetchedProjects
            
            // 如果还没有选中项目，自动选择第一个
            if selectedProject == nil && !fetchedProjects.isEmpty {
                selectedProject = fetchedProjects.first
            }
            
            // 如果当前选中的项目被删除了，重新选择
            if let selected = selectedProject,
               !fetchedProjects.contains(where: { $0.id == selected.id }) {
                selectedProject = fetchedProjects.first
            }
            
            print("✅ Loaded \(fetchedProjects.count) projects")
            
        } catch {
            errorMessage = error.localizedDescription
            showError = true
            print("❌ Failed to load projects: \(error.localizedDescription)")
        }
        
        isLoading = false
    }
    
    /// Create a new project
    func createProject(name: String, description: String = "") async {
        guard !name.trimmingCharacters(in: .whitespaces).isEmpty else {
            errorMessage = "Project name cannot be empty"
            showError = true
            return
        }
        
        isLoading = true
        errorMessage = nil
        
        do {
            let newProject = try await projectService.createProject(
                name: name.trimmingCharacters(in: .whitespaces),
                description: description.trimmingCharacters(in: .whitespaces)
            )
            
            // 添加到列表并选中
            self.projects.insert(newProject, at: 0)
            self.selectedProject = newProject
            
            print("✅ Created project: \(newProject.name)")
            
        } catch {
            errorMessage = error.localizedDescription
            showError = true
            print("❌ Failed to create project: \(error.localizedDescription)")
        }
        
        isLoading = false
    }
    
    /// Delete a project
    func deleteProject(_ project: Project) async {
        isLoading = true
        errorMessage = nil
        
        do {
            try await projectService.deleteProject(id: project.id)
            
            // 从列表中移除
            self.projects.removeAll { $0.id == project.id }
            
            // 如果删除的是当前选中的项目，选择另一个
            if selectedProject?.id == project.id {
                selectedProject = projects.first
            }
            
            print("✅ Deleted project: \(project.name)")
            
        } catch {
            errorMessage = error.localizedDescription
            showError = true
            print("❌ Failed to delete project: \(error.localizedDescription)")
        }
        
        isLoading = false
    }
    
    /// Select a project
    func selectProject(_ project: Project) {
        selectedProject = project
        print("📁 Selected project: \(project.name)")
    }
    
    /// Refresh current project data
    func refreshSelectedProject() async {
        guard let selected = selectedProject else { return }
        
        do {
            let updated = try await projectService.getProject(id: selected.id)
            
            // 更新列表中的项目
            if let index = projects.firstIndex(where: { $0.id == updated.id }) {
                projects[index] = updated
                selectedProject = updated
            }
            
            print("✅ Refreshed project: \(updated.name)")
            
        } catch {
            errorMessage = error.localizedDescription
            showError = true
            print("❌ Failed to refresh project: \(error.localizedDescription)")
        }
    }
}
