//
//  SaveSessionSheet.swift
//  RealtimeTranscriberMac
//

import SwiftUI

struct SaveSessionSheet: View {
    @Binding var isPresented: Bool
    let onConfirm: (String, String) -> Void   // name, notes

    @State private var name: String = ""
    @State private var notes: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.xl) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Name this session")
                    .font(.system(size: Theme.FontSize.title, weight: .semibold))
                    .foregroundColor(Theme.textPrimary)
                Text("The recording is already saved — this only adds a label.")
                    .font(.system(size: Theme.FontSize.micro))
                    .foregroundColor(Theme.textFaint)
            }

            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                Text("NAME").sectionCaption()
                TextField("e.g. Week 3 Lecture", text: $name)
                    .textFieldStyle(ThemedTextFieldStyle())
            }

            VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                Text("NOTES (OPTIONAL)").sectionCaption()
                TextEditor(text: $notes)
                    .font(.system(size: Theme.FontSize.medium))
                    .foregroundColor(Theme.textPrimary)
                    .scrollContentBackground(.hidden)
                    .padding(6)
                    .frame(height: 80)
                    .background(Theme.surface)
                    .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.row))
                    .overlay(
                        RoundedRectangle(cornerRadius: Theme.Radius.row)
                            .stroke(Theme.border, lineWidth: 1)
                    )
            }

            HStack {
                Button("Skip") {
                    isPresented = false
                    onConfirm("", "")
                }
                .buttonStyle(ThemedSecondaryButtonStyle())
                .keyboardShortcut(.escape)

                Spacer()

                Button("Save & New Session") {
                    isPresented = false
                    onConfirm(name, notes)
                }
                .buttonStyle(ThemedPrimaryButtonStyle())
                .keyboardShortcut(.return)
            }
        }
        .padding(Theme.Spacing.xl + 6)
        .themedSheet(width: 400)
    }
}
