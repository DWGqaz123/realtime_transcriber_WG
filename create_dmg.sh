#!/bin/bash
# create_dmg.sh - 创建 DMG 安装包

set -e

# ============================================================
# 配置区域
# ============================================================
APP_NAME="Realtime Transcriber"
VERSION="1.0.0"
DMG_NAME="RealtimeTranscriber_v${VERSION}.dmg"
BACKGROUND_COLOR="#2C3E50"  # 深蓝灰色背景

# ============================================================
# 颜色输出
# ============================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                        ║${NC}"
echo -e "${GREEN}║         🎯 Realtime Transcriber DMG 打包工具           ║${NC}"
echo -e "${GREEN}║                    Version ${VERSION}                      ║${NC}"
echo -e "${GREEN}║                                                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================================
# 1. 查找 App
# ============================================================
echo -e "${YELLOW}📦 Step 1: 查找 App...${NC}"

# 可能的位置
POSSIBLE_PATHS=(
    "./RealtimeTranscriberMac 2026-03-09 22-09-10/RealtimeTranscriberMac.app"
    "/Users/dwg/RealtimeTranscriberMac/RealtimeTranscriberMac.app"
    "$HOME/Desktop/RealtimeTranscriberMac/RealtimeTranscriberMac.app"
    "./RealtimeTranscriberMac.app"
    "$HOME/Desktop/RealtimeTranscriberMac.app"
)

APP_PATH=""
for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path" ]; then
        APP_PATH="$path"
        break
    fi
done

# 如果都没找到，让用户手动指定
if [ -z "$APP_PATH" ]; then
    echo -e "${RED}❌ 没有找到 App${NC}"
    echo -e "${YELLOW}请将 RealtimeTranscriberMac.app 拖到终端，然后按回车：${NC}"
    read -r APP_PATH
    APP_PATH=$(echo "$APP_PATH" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
fi

# 验证 App 存在
if [ ! -d "$APP_PATH" ]; then
    echo -e "${RED}❌ App 不存在: $APP_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 找到 App: $APP_PATH${NC}"

# ============================================================
# 2. 验证 App 内容
# ============================================================
echo -e "\n${YELLOW}🔍 Step 2: 验证 App 内容...${NC}"

# 检查后端文件
BACKEND_PATH="$APP_PATH/Contents/Resources/backend"
if [ ! -f "$BACKEND_PATH" ]; then
    echo -e "${RED}❌ Backend 文件不存在！${NC}"
    echo -e "${RED}   路径: $BACKEND_PATH${NC}"
    exit 1
fi

# 检查后端权限
if [ ! -x "$BACKEND_PATH" ]; then
    echo -e "${YELLOW}⚠️  Backend 不可执行，正在设置权限...${NC}"
    chmod +x "$BACKEND_PATH"
fi

# 获取文件信息
BACKEND_SIZE=$(du -h "$BACKEND_PATH" | cut -f1)
APP_SIZE=$(du -sh "$APP_PATH" | cut -f1)

echo -e "${GREEN}✅ Backend 文件验证通过${NC}"
echo -e "   Backend 大小: ${BACKEND_SIZE}"
echo -e "   App 总大小: ${APP_SIZE}"

# ============================================================
# 3. 创建临时目录结构
# ============================================================
echo -e "\n${YELLOW}📋 Step 3: 创建临时文件夹...${NC}"

TEMP_DIR="dmg_temp_$$"
rm -rf "$TEMP_DIR"
mkdir "$TEMP_DIR"

echo -e "${GREEN}✅ 临时文件夹: $TEMP_DIR${NC}"

# ============================================================
# 4. 复制 App
# ============================================================
echo -e "\n${YELLOW}📦 Step 4: 复制 App...${NC}"

cp -R "$APP_PATH" "$TEMP_DIR/"
echo -e "${GREEN}✅ App 复制完成${NC}"

# ============================================================
# 5. 创建 Applications 快捷方式
# ============================================================
echo -e "\n${YELLOW}🔗 Step 5: 创建快捷方式...${NC}"

ln -s /Applications "$TEMP_DIR/Applications"
echo -e "${GREEN}✅ Applications 快捷方式创建完成${NC}"

# ============================================================
# 6. 创建 README 文件
# ============================================================
echo -e "\n${YELLOW}📝 Step 6: 创建安装说明...${NC}"

cat > "$TEMP_DIR/README.txt" << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║         🎙️  Realtime Transcriber - 实时语音转录工具            ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

📦 安装步骤
═══════════════════════════════════════════════════════════════

1. 将 "RealtimeTranscriberMac.app" 拖到 "Applications" 文件夹
2. 首次打开可能显示安全警告：
   • 打开 "系统偏好设置" → "安全性与隐私"
   • 点击 "仍要打开"
3. 后端服务会自动启动

⚙️  配置 API Keys
═══════════════════════════════════════════════════════════════

启动 App 后：
1. 菜单栏 → RealtimeTranscriberMac → Settings (⌘+,)
2. 输入以下 API Keys：
   • OpenAI API Key (用于生成摘要)
   • ElevenLabs API Key (用于语音转文字)
3. 点击 "Save & Restart Backend"

🔑 获取 API Keys
═══════════════════════════════════════════════════════════════

OpenAI:     https://platform.openai.com/api-keys
ElevenLabs: https://elevenlabs.io/app/speech-synthesis

📊 系统要求
═══════════════════════════════════════════════════════════════

- macOS 12.0 (Monterey) 或更高版本
- Apple Silicon (M1/M2/M3) 或 Intel 处理器
- 需要互联网连接

💾 数据存储位置
═══════════════════════════════════════════════════════════════

所有录音和转录数据保存在：
~/Library/Application Support/RealtimeTranscriber/

🐛 故障排除
═══════════════════════════════════════════════════════════════

如果 App 无法启动：
1. 查看日志文件：
   ~/Library/Logs/RealtimeTranscriberMac.log

2. 检查后端进程：
   打开 "活动监视器"，搜索 "backend"

3. 手动重启后端：
   Settings → "Save & Restart Backend"

4. 完全重置：
   删除以下文件夹：
   ~/Library/Application Support/RealtimeTranscriber/

📧 技术支持
═══════════════════════════════════════════════════════════════

邮箱: your-email@example.com
项目: https://github.com/yourname/realtime-transcriber

版本: 1.0.0
构建日期: $(date +"%Y-%m-%d")

═══════════════════════════════════════════════════════════════
EOF

echo -e "${GREEN}✅ README.txt 创建完成${NC}"

# ============================================================
# 7. 创建 .DS_Store (窗口布局)
# ============================================================
echo -e "\n${YELLOW}🎨 Step 7: 配置窗口布局...${NC}"

# 这一步会在实际挂载 DMG 后手动调整
echo -e "${BLUE}ℹ️  窗口布局将在下一步创建${NC}"

# ============================================================
# 8. 创建 DMG
# ============================================================
echo -e "\n${YELLOW}🔨 Step 8: 创建 DMG...${NC}"

# 删除旧的 DMG
if [ -f "$DMG_NAME" ]; then
    echo -e "${YELLOW}⚠️  删除旧的 DMG: $DMG_NAME${NC}"
    rm "$DMG_NAME"
fi

# 创建 DMG
hdiutil create -volname "$APP_NAME" \
    -srcfolder "$TEMP_DIR" \
    -ov \
    -format UDZO \
    -imagekey zlib-level=9 \
    "$DMG_NAME"

echo -e "${GREEN}✅ DMG 创建成功${NC}"

# ============================================================
# 9. 清理临时文件
# ============================================================
echo -e "\n${YELLOW}🧹 Step 9: 清理临时文件...${NC}"

rm -rf "$TEMP_DIR"
echo -e "${GREEN}✅ 清理完成${NC}"

# ============================================================
# 10. 显示结果
# ============================================================
echo -e "\n${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                        ║${NC}"
echo -e "${GREEN}║                  ✅ DMG 创建成功！                       ║${NC}"
echo -e "${GREEN}║                                                        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
echo -e ""
echo -e "${BLUE}📦 文件名:${NC}   $DMG_NAME"
echo -e "${BLUE}📏 文件大小:${NC} $(du -h "$DMG_NAME" | cut -f1)"
echo -e "${BLUE}📍 位置:${NC}     $(pwd)/$DMG_NAME"
echo -e ""
echo -e "${YELLOW}📋 下一步操作：${NC}"
echo -e "   1. 双击打开 DMG 测试安装"
echo -e "   2. 拖动 App 到 Applications"
echo -e "   3. 运行并测试所有功能"
echo -e "   4. 如果一切正常，可以分发给用户"
echo -e ""
echo -e "${YELLOW}🔒 安全提示：${NC}"
echo -e "   首次运行时用户会看到安全警告"
echo -e "   需要在 '系统偏好设置' → '安全性与隐私' 中允许"
echo -e ""