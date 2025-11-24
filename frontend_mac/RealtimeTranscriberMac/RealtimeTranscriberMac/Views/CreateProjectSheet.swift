//
//  CreateProjectSheet.swift
//  RealtimeTranscriberMac
//
//  Sheet for creating a new project
//

import SwiftUI

struct CreateProjectSheet: View {
    @Environment(\.dismiss) var dismiss
    
    @State private var projectName: String = ""
    @State private var projectDescription: String = ""
    @State private var isCreating: Bool = false
    
    let onCreate: (String, String) async -> Void
    
    var body: some View {
        VStack(spacing: 20) {
            // Header
            HStack {
                Text("Create New Project")
                    .font(.title2)
                    .fontWeight(.bold)
                
                Spacer()
                
                Button(action: {
                    dismiss()
                }) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundColor(.secondary)
                        .font(.title3)
                }
                .buttonStyle(.plain)
            }
            
            Divider()
            
            // Form
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Project Name")
                        .font(.headline)
                    
                    TextField("e.g., CMU Capstone Project", text: $projectName)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit {
                            if canCreate {
                                createProject()
                            }
                        }
                }
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("Description (Optional)")
                        .font(.headline)
                    
                    TextEditor(text: $projectDescription)
                        .font(.body)
                        .frame(height: 80)
                        .padding(4)
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(6)
                }
                
                Text("You can organize your recordings by project. All transcripts and sessions will be associated with this project.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            // Buttons
            HStack(spacing: 12) {
                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.escape)
                
                Spacer()
                
                Button(action: {
                    createProject()
                }) {
                    HStack(spacing: 6) {
                        if isCreating {
                            ProgressView()
                                .scaleEffect(0.8)
                                .frame(width: 16, height: 16)
                        } else {
                            Image(systemName: "plus.circle.fill")
                        }
                        Text("Create Project")
                    }
                    .frame(minWidth: 140)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canCreate || isCreating)
                .keyboardShortcut(.return)
            }
        }
        .padding(24)
        .frame(width: 500, height: 350)
    }
    
    // MARK: - Computed Properties
    
    private var canCreate: Bool {
        !projectName.trimmingCharacters(in: .whitespaces).isEmpty
    }
    
    // MARK: - Actions
    
    private func createProject() {
        guard canCreate else { return }
        
        isCreating = true
        
        Task {
            await onCreate(projectName, projectDescription)
            dismiss()
        }
    }
}

#Preview {
    CreateProjectSheet { name, description in
        print("Creating project: \(name)")
        try? await Task.sleep(nanoseconds: 1_000_000_000)
    }
}
