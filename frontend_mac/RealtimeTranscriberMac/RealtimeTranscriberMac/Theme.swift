//
//  Theme.swift
//  RealtimeTranscriberMac
//
//  设计 token —— 取自 Pencil 设计稿 "Realtime Transcriber — Recording"。
//  这套配色是固定深色，不跟随系统外观：设计稿本身就是为深色环境画的，
//  用语义色（NSColor.controlBackgroundColor 等）在浅色模式下会散架。
//

import SwiftUI

extension Color {
    init(hex: UInt32) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255,
            opacity: 1
        )
    }
}

enum Theme {

    // MARK: - 背景层次（由深到浅表示层级抬升）

    static let windowBg      = Color(hex: 0x080A0D)   // 窗口最底
    static let titleBar      = Color(hex: 0x0B0D11)
    static let panelBg       = Color(hex: 0x0B0E12)   // 左右两栏
    static let contentBg     = Color(hex: 0x0E1116)   // 中间主区域
    static let surface       = Color(hex: 0x12161C)   // 输入框、次级按钮
    static let card          = Color(hex: 0x131820)   // 摘要卡片
    static let cardAlt       = Color(hex: 0x121820)   // 展开的项目容器
    static let rowActive     = Color(hex: 0x1A2631)   // 选中的会话行

    // MARK: - 描边

    static let border        = Color(hex: 0x252C36)
    static let borderStrong  = Color(hex: 0x29313B)

    // MARK: - 文字

    static let textPrimary   = Color(hex: 0xF5F7FA)
    static let textSecondary = Color(hex: 0xBBC4CF)
    static let textMuted     = Color(hex: 0x9AA4B2)
    static let textFaint     = Color(hex: 0x626C79)

    // MARK: - 强调与状态

    static let accent        = Color(hex: 0x79C7FF)   // 主强调（蓝）
    static let accentSoft    = Color(hex: 0xB8E1FF)
    static let accentBg      = Color(hex: 0x213242)   // 选中的模式按钮底色

    static let success       = Color(hex: 0x5FE3A1)
    static let successBg     = Color(hex: 0x10251F)
    static let danger        = Color(hex: 0xFF6B78)
    static let dangerBg      = Color(hex: 0x33181E)
    static let warning       = Color(hex: 0xF6C85F)
    static let violet        = Color(hex: 0xB27CFF)

    // MARK: - 圆角

    enum Radius {
        static let chip: CGFloat = 4
        static let control: CGFloat = 5
        static let row: CGFloat = 6
        static let field: CGFloat = 7
        static let card: CGFloat = 8
        static let window: CGFloat = 14
    }

    // MARK: - 字号
    //
    // 设计稿的 8~11px 在 Mac 上偏小，这里整体抬到 macOS 可读区间，
    // 但保留原有的层级关系。

    enum FontSize {
        static let micro: CGFloat = 10   // 稿 8px  —— 标签、计数
        static let small: CGFloat = 11   // 稿 9px
        static let body: CGFloat = 12    // 稿 11px —— 正文
        static let medium: CGFloat = 13  // 稿 12px
        static let title: CGFloat = 15   // 稿 14~15px
        static let large: CGFloat = 17   // 稿 16~17px
    }

    // MARK: - 间距

    enum Spacing {
        static let xs: CGFloat = 3
        static let sm: CGFloat = 6
        static let md: CGFloat = 9
        static let lg: CGFloat = 14
        static let xl: CGFloat = 18
    }
}

// MARK: - 复用样式
//
// macOS 自带的 .roundedBorder / .borderedProminent 在强制深色下会用系统
// 灰蓝，和这套配色不是一个体系，所以自定义。

struct ThemedTextFieldStyle: TextFieldStyle {
    func _body(configuration: TextField<Self._Label>) -> some View {
        configuration
            .textFieldStyle(.plain)
            .font(.system(size: Theme.FontSize.medium))
            .foregroundColor(Theme.textPrimary)
            .padding(.horizontal, 10)
            .frame(height: 32)
            .background(Theme.surface)
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.row))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Radius.row)
                    .stroke(Theme.border, lineWidth: 1)
            )
    }
}

/// 主行动按钮：蓝底蓝字
struct ThemedPrimaryButtonStyle: ButtonStyle {
    var tint: Color = Theme.accent
    var background: Color = Theme.accentBg

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: Theme.FontSize.body, weight: .semibold))
            .foregroundColor(tint)
            .padding(.horizontal, 14)
            .frame(height: 30)
            .background(background.opacity(configuration.isPressed ? 0.55 : 1))
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.row))
    }
}

/// 次级按钮：中性底色
struct ThemedSecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: Theme.FontSize.body, weight: .medium))
            .foregroundColor(Theme.textMuted)
            .padding(.horizontal, 14)
            .frame(height: 30)
            .background(Theme.surface.opacity(configuration.isPressed ? 0.55 : 1))
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.row))
    }
}

extension View {
    /// sheet 通用容器：铺底色 + 深色外观
    func themedSheet(width: CGFloat? = nil, minHeight: CGFloat? = nil) -> some View {
        self
            .background(Theme.contentBg)
            .frame(width: width)
            .frame(minHeight: minHeight)
            .preferredColorScheme(.dark)
    }

    /// 小节标题：全大写 + 字距
    func sectionCaption() -> some View {
        self
            .font(.system(size: Theme.FontSize.micro, weight: .semibold))
            .foregroundColor(Theme.textFaint)
            .tracking(0.6)
    }
}
