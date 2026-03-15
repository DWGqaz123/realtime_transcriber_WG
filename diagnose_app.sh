#!/bin/bash
# diagnose_app.sh - 诊断已安装的 App

set +e

APP_PATH="/Applications/RealtimeTranscriberMac.app"

echo "╔════════════════════════════════════════════════════════╗"
echo "║      🔍 Realtime Transcriber 诊断工具                  ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# 1. 检查 App 是否存在
echo "📦 Step 1: 检查 App 安装"
if [ -d "$APP_PATH" ]; then
    echo "✅ App 已安装: $APP_PATH"
else
    echo "❌ App 未安装在 Applications 文件夹"
    exit 1
fi

# 2. 检查 backend 文件
echo ""
echo "🔍 Step 2: 检查后端文件"
BACKEND_PATH="$APP_PATH/Contents/Resources/backend"
if [ -f "$BACKEND_PATH" ]; then
    echo "✅ Backend 文件存在"
    
    # 检查大小
    SIZE=$(du -h "$BACKEND_PATH" | cut -f1)
    echo "   文件大小: $SIZE"
    
    # 检查权限
    PERMS=$(ls -l "$BACKEND_PATH" | awk '{print $1}')
    echo "   权限: $PERMS"
    
    if [ -x "$BACKEND_PATH" ]; then
        echo "   ✅ 可执行权限正常"
    else
        echo "   ❌ 缺少可执行权限"
        echo "   修复: chmod +x $BACKEND_PATH"
    fi
else
    echo "❌ Backend 文件不存在！"
    exit 1
fi

# 3. 检查隔离属性
echo ""
echo "🔒 Step 3: 检查安全属性"
QUARANTINE=$(xattr -l "$APP_PATH" | grep "com.apple.quarantine" || true)
if [ -n "$QUARANTINE" ]; then
    echo "⚠️  App 被隔离"
    echo "   属性: $QUARANTINE"
    echo "   修复: xattr -cr $APP_PATH"
else
    echo "✅ 无隔离属性"
fi

# 4. 检查进程
echo ""
echo "🏃 Step 4: 检查后端进程"
PROCESS=$(ps aux | grep "[b]ackend" || true)
if [ -n "$PROCESS" ]; then
    echo "✅ Backend 进程运行中:"
    echo "$PROCESS"
else
    echo "❌ Backend 进程未运行"
fi

# 5. 检查端口
echo ""
echo "🔌 Step 5: 检查端口监听"
PORT_CHECK=$(lsof -i :8000 || true)
if [ -n "$PORT_CHECK" ]; then
    echo "✅ 端口 8000 正在监听:"
    echo "$PORT_CHECK"
else
    echo "❌ 端口 8000 未监听"
fi

# 6. 检查日志
echo ""
echo "📝 Step 6: 检查日志文件"
LOG_PATH="$HOME/Library/Logs/RealtimeTranscriberMac.log"
if [ -f "$LOG_PATH" ]; then
    echo "✅ 日志文件存在: $LOG_PATH"
    echo "   最后 20 行:"
    echo "   ----------------------------------------"
    tail -20 "$LOG_PATH"
    echo "   ----------------------------------------"
else
    echo "⚠️  日志文件不存在"
fi

# 7. 手动测试后端
echo ""
echo "🧪 Step 7: 手动测试后端启动"
echo "尝试手动运行后端（5秒超时）..."
timeout 5 "$BACKEND_PATH" 2>&1 | head -30 || true

# 8. 检查数据目录
echo ""
echo "💾 Step 8: 检查数据目录"
DATA_DIR="$HOME/Library/Application Support/RealtimeTranscriber"
if [ -d "$DATA_DIR" ]; then
    echo "✅ 数据目录存在: $DATA_DIR"
    echo "   内容:"
    ls -la "$DATA_DIR"
else
    echo "⚠️  数据目录不存在（首次运行会创建）"
fi

# 9. 检查依赖库
echo ""
echo "📚 Step 9: 检查后端依赖"
if command -v otool &> /dev/null; then
    echo "动态库依赖:"
    otool -L "$BACKEND_PATH" 2>&1 | head -20
fi

# 总结
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║                   📋 诊断总结                          ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
echo "请将以上输出发送给开发者进行分析"
echo ""