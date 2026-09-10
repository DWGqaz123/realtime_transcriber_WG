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
        VStack(alignment: .leading, spacing: Theme.Spacing.xl) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Create new project")
                        .font(.system(size: Theme.FontSize.title, weight: .semibold))
                        .foregroundColor(Theme.textPrimary)
                    Text("Sessions and summaries are grouped under a project.")
                        .font(.system(size: Theme.FontSize.micro))
                        .foregroundColor(Theme.textFaint)
                }

                Spacer()

                IconButton(icon: "xmark") { dismiss() }
            }

            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                Text("PROJECT NAME").sectionCaption()
                TextField("e.g., CMU Capstone Project", text: $projectName)
                    .textFieldStyle(ThemedTextFieldStyle())
                    .onSubmit(create)
            }

            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                Text("DESCRIPTION (OPTIONAL)").sectionCaption()
                TextEditor(text: $projectDescription)
                    .font(.system(size: Theme.FontSize.medium))
                    .foregroundColor(Theme.textPrimary)
                    .scrollContentBackground(.hidden)
                    .padding(6)
                    .frame(height: 70)
                    .background(Theme.surface)
                    .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.row))
                    .overlay(
                        RoundedRectangle(cornerRadius: Theme.Radius.row)
                            .stroke(Theme.border, lineWidth: 1)
                    )
            }

            HStack {
                Button("Cancel") { dismiss() }
                    .buttonStyle(ThemedSecondaryButtonStyle())
                    .keyboardShortcut(.escape)

                Spacer()

                Button(action: create) {
                    HStack(spacing: Theme.Spacing.sm) {
                        if isCreating {
                            ProgressView().controlSize(.small).scaleEffect(0.6).frame(width: 10, height: 10)
                        }
                        Text(isCreating ? "Creating…" : "Create Project")
                    }
                }
                .buttonStyle(ThemedPrimaryButtonStyle())
                .keyboardShortcut(.return)
                .disabled(projectName.trimmingCharacters(in: .whitespaces).isEmpty || isCreating)
            }
        }
        .padding(Theme.Spacing.xl + 6)
        .themedSheet(width: 430)
    }

    private func create() {
        let trimmed = projectName.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty, !isCreating else { return }
        isCreating = true
        Task {
            await onCreate(trimmed, projectDescription.trimmingCharacters(in: .whitespaces))
            isCreating = false
            dismiss()
        }
    }
}


#Preview {
    CreateProjectSheet { name, description in
        try? await Task.sleep(nanoseconds: 1_000_000_000)
    }
}
