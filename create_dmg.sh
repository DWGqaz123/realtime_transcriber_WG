#!/bin/bash
# create_dmg.sh - 创建 DMG 安装包
set -e

VERSION="1.1.0"
DMG_NAME="RealtimeTranscriber_v${VERSION}.dmg"
ARCHIVES_DIR="$HOME/Library/Developer/Xcode/Archives"

# 找最新的 RealtimeTranscriberMac archive
LATEST_ARCHIVE=$(find "$ARCHIVES_DIR" -name "RealtimeTranscriberMac*.xcarchive" -type d 2>/dev/null | sort | tail -1)

if [ -z "$LATEST_ARCHIVE" ]; then
    echo "❌ 未找到 Xcode Archive，请先在 Xcode 中 Archive"
    exit 1
fi

APP_PATH="$LATEST_ARCHIVE/Products/Applications/RealtimeTranscriberMac.app"

if [ ! -d "$APP_PATH" ]; then
    echo "❌ App 不存在: $APP_PATH"
    exit 1
fi

# 验证 backend binary
BACKEND_PATH="$APP_PATH/Contents/Resources/backend"
if [ ! -f "$BACKEND_PATH" ]; then
    echo "❌ Backend binary 不存在: $BACKEND_PATH"
    echo "   请先运行 PyInstaller 并将 dist/backend 添加到 Xcode Resources"
    exit 1
fi
chmod +x "$BACKEND_PATH"

echo "✅ Archive: $LATEST_ARCHIVE"
echo "✅ App 大小: $(du -sh "$APP_PATH" | cut -f1)"
echo "✅ Backend 大小: $(du -h "$BACKEND_PATH" | cut -f1)"

# 创建临时目录
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

cp -R "$APP_PATH" "$TEMP_DIR/"
ln -s /Applications "$TEMP_DIR/Applications"

# 删除旧 DMG 并创建新的
rm -f "$DMG_NAME"
hdiutil create -volname "Realtime Transcriber" \
    -srcfolder "$TEMP_DIR" \
    -ov \
    -format UDZO \
    -imagekey zlib-level=9 \
    "$DMG_NAME"

echo ""
echo "✅ DMG 创建成功: $(pwd)/$DMG_NAME"
echo "   大小: $(du -h "$DMG_NAME" | cut -f1)"
