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
        VStack(alignment: .leading, spacing: 20) {
            Text("Name This Session")
                .font(.title2)
                .fontWeight(.semibold)

            VStack(alignment: .leading, spacing: 6) {
                Text("Name")
                    .font(.caption)
                    .foregroundColor(.secondary)
                TextField("e.g. Week 3 Lecture", text: $name)
                    .textFieldStyle(.roundedBorder)
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Notes (optional)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                TextEditor(text: $notes)
                    .frame(height: 80)
                    .font(.body)
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                    )
            }

            HStack {
                Button("Skip") {
                    isPresented = false
                    onConfirm("", "")
                }
                .buttonStyle(.plain)
                .foregroundColor(.secondary)

                Spacer()

                Button("Save & New Session") {
                    isPresented = false
                    onConfirm(name, notes)
                }
                .buttonStyle(.borderedProminent)
                .tint(.indigo)
            }
        }
        .padding(24)
        .frame(width: 380)
    }
}
