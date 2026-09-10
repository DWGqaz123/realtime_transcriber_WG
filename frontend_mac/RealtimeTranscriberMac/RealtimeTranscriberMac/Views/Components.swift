//
//  Components.swift
//  RealtimeTranscriberMac
//
//  跨界面复用的基础组件。这些模式原本在各视图里各写了一遍：
//  分段控件 3 处、chip 16 处、空状态 5 处、摘要卡片 2 份近乎相同的实现。
//

import SwiftUI

// MARK: - 分段控件
//
// 录音模式、检索模式、会话详情的标签页用的是同一套视觉：选中项以
// accentBg 垫底、accent 着色。

struct SegmentedControl<T: Hashable>: View {
    let options: [T]
    @Binding var selection: T
    let label: (T) -> String
    /// 尾随内容，例如标签页上的计数
    var trailing: ((T) -> AnyView)? = nil
    var height: CGFloat = 28
    var onChange: (() -> Void)? = nil

    var body: some View {
        HStack(spacing: Theme.Spacing.xs) {
            ForEach(options, id: \.self) { option in
                let isOn = selection == option
                Button {
                    selection = option
                    onChange?()
                } label: {
                    HStack(spacing: Theme.Spacing.sm) {
                        Text(label(option))
                            .font(.system(size: Theme.FontSize.body, weight: isOn ? .semibold : .regular))
                        if let trailing { trailing(option) }
                    }
                    .foregroundColor(isOn ? Theme.accent : Theme.textMuted)
                    .padding(.horizontal, 10)
                    .frame(height: height)
                    .background(isOn ? Theme.accentBg : Color.clear)
                    .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.control))
                }
                .buttonStyle(.plain)
            }
        }
        .padding(Theme.Spacing.xs)
        .background(Theme.panelBg)
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.field))
    }
}

// MARK: - 图标按钮

struct IconButton: View {
    let icon: String
    var tint: Color = Theme.textFaint
    var background: Color = Theme.surface
    var size: CGFloat = 24
    var help: String? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 10, weight: .semibold))
                .foregroundColor(tint)
                .frame(width: size, height: size)
                .background(background)
                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.chip))
        }
        .buttonStyle(.plain)
        .help(help ?? "")
    }
}

// MARK: - 标签

struct Chip: View {
    let text: String
    var icon: String? = nil
    var tint: Color = Theme.textFaint
    var background: Color? = nil
    var mono: Bool = false

    var body: some View {
        HStack(spacing: 3) {
            if let icon {
                Image(systemName: icon).font(.system(size: 8))
            }
            Text(text)
                .font(.system(size: Theme.FontSize.micro,
                              weight: .medium,
                              design: mono ? .monospaced : .default))
        }
        .foregroundColor(tint)
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(background ?? tint.opacity(0.14))
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.chip))
    }
}

// MARK: - 空状态

struct EmptyState<Actions: View>: View {
    let icon: String
    let title: String
    var hint: String? = nil
    var maxHintWidth: CGFloat = 320
    @ViewBuilder var actions: () -> Actions

    var body: some View {
        VStack(spacing: Theme.Spacing.md) {
            Image(systemName: icon)
                .font(.system(size: 26))
                .foregroundColor(Theme.textFaint.opacity(0.55))

            Text(title)
                .font(.system(size: Theme.FontSize.body, weight: .medium))
                .foregroundColor(Theme.textMuted)

            if let hint {
                Text(hint)
                    .font(.system(size: Theme.FontSize.micro))
                    .foregroundColor(Theme.textFaint)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: maxHintWidth)
            }

            actions()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(Theme.Spacing.lg)
    }
}

extension EmptyState where Actions == EmptyView {
    init(icon: String, title: String, hint: String? = nil, maxHintWidth: CGFloat = 320) {
        self.init(icon: icon, title: title, hint: hint, maxHintWidth: maxHintWidth) { EmptyView() }
    }
}

// MARK: - 小节标题

struct SectionHeader<Trailing: View>: View {
    let title: String
    @ViewBuilder var trailing: () -> Trailing

    var body: some View {
        HStack(spacing: Theme.Spacing.md) {
            Text(title.uppercased()).sectionCaption()
            Spacer()
            trailing()
        }
    }
}

extension SectionHeader where Trailing == EmptyView {
    init(_ title: String) {
        self.init(title: title) { EmptyView() }
    }
}

// MARK: - 摘要卡片
//
// 录音面板与会话详情共用。两处的数据来源不同，靠 SummaryDisplayable 抹平。

struct SummaryCard<S: SummaryDisplayable>: View {
    let summary: S
    /// 传入即显示删除按钮（仅历史详情需要）
    var onDelete: (() -> Void)? = nil

    @State private var isExpanded = true
    @State private var isHovering = false

    private var accent: Color { summary.isFinalSummary ? Theme.warning : Theme.accent }

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Spacing.md) {
            header

            if isExpanded {
                VStack(alignment: .leading, spacing: Theme.Spacing.sm) {
                    ForEach(summary.bullets, id: \.self) { bullet in
                        HStack(alignment: .top, spacing: Theme.Spacing.sm) {
                            Circle()
                                .fill(accent)
                                .frame(width: 3, height: 3)
                                .padding(.top, 6)
                            Text(bullet)
                                .font(.system(size: Theme.FontSize.body))
                                .foregroundColor(Theme.textSecondary)
                                .fixedSize(horizontal: false, vertical: true)
                                .textSelection(.enabled)
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 11)
        .background(Theme.card)
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.card))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Radius.card)
                .stroke(summary.isFinalSummary ? Theme.warning.opacity(0.35) : Theme.border, lineWidth: 1)
        )
        .onHover { isHovering = $0 }
    }

    private var header: some View {
        HStack(spacing: Theme.Spacing.sm) {
            Text(summary.displayTime)
                .font(.system(size: Theme.FontSize.micro, design: .monospaced))
                .foregroundColor(Theme.textFaint)

            if summary.isFinalSummary {
                Chip(text: "FINAL", tint: Theme.warning)
            }

            if let lines = summary.lineCount, lines > 0 {
                Text("\(lines) lines")
                    .font(.system(size: Theme.FontSize.micro, design: .monospaced))
                    .foregroundColor(Theme.textFaint)
            }

            Spacer()

            if isHovering, let onDelete {
                Button(action: onDelete) {
                    Image(systemName: "trash")
                        .font(.system(size: 9))
                        .foregroundColor(Theme.danger)
                }
                .buttonStyle(.plain)
                .help("Delete summary")
            }

            Button {
                withAnimation(.easeInOut(duration: 0.18)) { isExpanded.toggle() }
            } label: {
                Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundColor(Theme.textFaint)
            }
            .buttonStyle(.plain)
        }
    }
}

// MARK: - 小文字按钮
//
// Clear / Export / Export all 这类轻量操作，视觉上是可点的 chip。

struct TextChipButton: View {
    let title: String
    var icon: String? = nil
    var tint: Color = Theme.textMuted
    var help: String? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                if let icon {
                    Image(systemName: icon).font(.system(size: 9))
                }
                Text(title)
                    .font(.system(size: Theme.FontSize.micro, weight: .medium))
            }
            .foregroundColor(tint)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Theme.surface)
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.chip))
        }
        .buttonStyle(.plain)
        .help(help ?? "")
    }
}
