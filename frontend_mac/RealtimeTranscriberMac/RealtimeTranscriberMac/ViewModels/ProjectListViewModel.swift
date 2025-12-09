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
    
    @Published var projectSessions: [Int: [RecordingSession]] = [:]  // 项目ID -> sessions
    @Published var expandedProjects: Set<Int> = []  // 展开的项目ID
    @Published var selectedSession: RecordingSession?  // 选中的 session
    @Published var showSessionDetail: Bool = false  // 是否显示详情浮窗
    @Published var sessionDetailLoading: Bool = false  // 加载详情中
    
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
    
    // MARK: - Session Management

    // 切换项目展开/收起状态
    func toggleProjectExpansion(_ project: Project) {
        if expandedProjects.contains(project.id) {
            // 收起
            expandedProjects.remove(project.id)
            print("📁 Collapsed project: \(project.name)")
        } else {
            // 展开，并加载 sessions（不强制刷新，使用缓存）
            expandedProjects.insert(project.id)
            print("📂 Expanded project: \(project.name)")
            Task {
                await loadProjectSessions(project.id)  // 🔧 不传参数，默认 forceRefresh = false
            }
        }
    }

    /// 加载项目的 sessions
    func loadProjectSessions(_ projectId: Int, forceRefresh: Bool = false) async {
        // 如果已经加载过且不强制刷新，跳过
        if !forceRefresh && projectSessions[projectId] != nil {
            print("📦 Using cached sessions for project \(projectId)")
            return
        }
        
        do {
            print("🔄 Loading sessions for project \(projectId)...")
            let sessions = try await projectService.fetchProjectSessions(projectId: projectId)
            self.projectSessions[projectId] = sessions
            print("✅ Loaded \(sessions.count) sessions for project \(projectId)")
        } catch {
            errorMessage = error.localizedDescription
            showError = true
            print("❌ Failed to load sessions: \(error.localizedDescription)")
        }
    }

    /// 选择并显示 session 详情
    func selectSession(projectId: Int, session: RecordingSession) {
        selectedSession = session
        
        // 如果 session 没有完整转录，从服务器加载
        if session.transcriptText == nil || session.transcriptText!.isEmpty {
            Task {
                await loadSessionDetail(projectId: projectId, sessionId: session.id)
            }
        } else {
            // 已有转录，直接显示
            showSessionDetail = true
        }
    }

    /// 加载 session 详情（包含完整转录）
    private func loadSessionDetail(projectId: Int, sessionId: Int) async {
        sessionDetailLoading = true
        
        do {
            let detailSession = try await projectService.fetchSessionDetail(
                projectId: projectId,
                sessionId: sessionId
            )
            
            // 更新缓存
            if var sessions = projectSessions[projectId] {
                if let index = sessions.firstIndex(where: { $0.id == sessionId }) {
                    sessions[index] = detailSession
                    projectSessions[projectId] = sessions
                }
            }
            
            // 更新选中的 session
            selectedSession = detailSession
            showSessionDetail = true
            
            print("✅ Loaded session detail: \(detailSession.transcriptText?.count ?? 0) chars")
            
        } catch {
            errorMessage = error.localizedDescription
            showError = true
            print("❌ Failed to load session detail: \(error.localizedDescription)")
        }
        
        sessionDetailLoading = false
    }

    /// 关闭 session 详情
    func closeSessionDetail() {
        showSessionDetail = false
        selectedSession = nil
    }
    
    /// Delete a session
    func deleteSession(projectId: Int, sessionId: Int) async {
        do {
            print("🗑️ Deleting session \(sessionId) from project \(projectId)")
            
            // 先检查 session 是否在缓存中
            if let sessions = projectSessions[projectId] {
                let exists = sessions.contains { $0.id == sessionId }
                print("   Session exists in cache: \(exists)")
            }
            
            try await projectService.deleteSession(projectId: projectId, sessionId: sessionId)
            
            print("✅ Backend confirmed deletion")
            
            // 从缓存中移除
            if var sessions = projectSessions[projectId] {
                let beforeCount = sessions.count
                sessions.removeAll { $0.id == sessionId }
                projectSessions[projectId] = sessions
                print("   Removed from cache: \(beforeCount) → \(sessions.count)")
            }
            
            // 关闭详情窗口（如果正在显示被删除的 session）
            if selectedSession?.id == sessionId {
                closeSessionDetail()
                print("   Closed session detail sheet")
            }
            
            // 刷新项目信息（更新 session 计数）
            if let project = projects.first(where: { $0.id == projectId }) {
                let updated = try await projectService.getProject(id: projectId)
                if let index = projects.firstIndex(where: { $0.id == projectId }) {
                    projects[index] = updated
                    
                    // 如果是当前选中的项目，更新选中状态
                    if selectedProject?.id == projectId {
                        selectedProject = updated
                    }
                }
                print("   Updated project: sessions = \(updated.sessionCount)")
            }
            
            print("✅ Session deleted successfully")
            
        } catch ProjectServiceError.serverError(let statusCode) where statusCode == 404 {
            print("⚠️ Session \(sessionId) not found (404)")
            
            // 即使 404，也从前端缓存中移除
            if var sessions = projectSessions[projectId] {
                sessions.removeAll { $0.id == sessionId }
                projectSessions[projectId] = sessions
            }
            
            // 关闭详情窗口
            if selectedSession?.id == sessionId {
                closeSessionDetail()
            }
            
            errorMessage = "Session not found. It may have been deleted already."
            showError = true
            
        } catch {
            errorMessage = error.localizedDescription
            showError = true
            print("❌ Failed to delete session: \(error.localizedDescription)")
        }
    }
    
    /// Delete a summary
    func deleteSummary(projectId: Int, sessionId: Int, summaryId: Int) async {
        do {
            print("🗑️ Deleting summary \(summaryId) from session \(sessionId)")
            
            try await projectService.deleteSummary(
                projectId: projectId,
                sessionId: sessionId,
                summaryId: summaryId
            )
            
            print("✅ Summary deleted successfully")
            
            // 如果当前显示的 session 就是被修改的，重新加载详情
            if selectedSession?.id == sessionId {
                await loadSessionDetail(projectId: projectId, sessionId: sessionId)
            }
            
        } catch {
            errorMessage = error.localizedDescription
            showError = true
            print("❌ Failed to delete summary: \(error.localizedDescription)")
        }
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
            // 🔧 新增：清除该项目的 session 缓存，强制重新加载
            projectSessions[selected.id] = nil
            
            // 🔧 新增：如果项目是展开状态，重新加载 sessions
            if expandedProjects.contains(selected.id) {
                await loadProjectSessions(selected.id, forceRefresh: true)
            }
            
            print("✅ Refreshed project: \(updated.name), sessions: \(updated.sessionCount)")
            
        } catch {
            errorMessage = error.localizedDescription
            showError = true
            print("❌ Failed to refresh project: \(error.localizedDescription)")
        }
    }
}
