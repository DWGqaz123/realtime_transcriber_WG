#!/bin/bash
# backend/build_backend.sh

set -e

echo "🔨 开始打包 Python 后端..."

# 检测 conda 路径（自动查找）
if [ -z "$CONDA_EXE" ]; then
    echo "⚠️  未检测到激活的 conda 环境"
    echo "请确保已经激活了 realtime_transcriber_env"
    exit 1
fi

# 检查当前环境
CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
if [ "$CURRENT_ENV" != "realtime_transcriber_env" ]; then
    echo "❌ 当前环境不是 realtime_transcriber_env"
    echo "请先运行: conda activate realtime_transcriber_env"
    exit 1
fi

echo "✅ 当前环境: $CURRENT_ENV"

# 确保 PyInstaller 是最新版本
echo "📦 检查 PyInstaller..."
pip install --upgrade pyinstaller

# 清理旧文件
echo "🧹 清理旧文件..."
rm -rf build dist __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 打包
echo "🔨 开始打包..."
pyinstaller backend.spec --clean

# 验证
if [ -f "dist/backend" ]; then
    echo ""
    echo "✅ 后端打包成功！"
    echo "📦 大小: $(du -h dist/backend | cut -f1)"
    echo "📍 位置: $(pwd)/dist/backend"
    echo ""
    
    # 设置权限
    chmod +x dist/backend
    
    # 测试运行
    echo "🧪 测试启动（5秒后自动停止）..."
    timeout 5 dist/backend 2>&1 | head -20 || true
    
    echo ""
    echo "✅ 打包完成！可以在 Xcode 中使用了"
else
    echo "❌ 打包失败"
    exit 1
fi