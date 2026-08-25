# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Realtime Transcriber backend.

    pyinstaller backend.spec --clean        →  dist/backend

产物是单文件可执行程序，由 macOS App 作为子进程启动
（RealtimeTranscriberMac.app/Contents/Resources/backend）。
BackendManager 直接 exec 这个路径，所以必须是 onefile，不能是 onedir。
"""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

BACKEND_DIR = Path(SPECPATH)

# 运行时才按字符串加载、静态分析看不见的模块
hiddenimports = [
    # uvicorn 的 protocol / loop / lifespan 实现都是按名字动态选的
    *collect_submodules("uvicorn"),
    *collect_submodules("websockets"),
    # SQLAlchemy 的 dialect 同样是运行时按名字加载
    "sqlalchemy.dialects.sqlite",
    "sqlalchemy.sql.default_comparator",
    # anyio 按后端名字 import，httpx/starlette 依赖它
    "anyio._backends._asyncio",
    "httpcore",
    "h11",
    "certifi",
    "dotenv",
    # 项目自身：main.py 用的是平铺导入（from routes import ...），
    # 不是包内相对导入，所以要显式列出
    "config",
    "live_session",
    "session_manager",
    "session_persistence",
    "summary_handler",
    "summary_service",
    "elevenlabs_client",
    "embedding_service",
    "faiss_manager",
    "indexing_service",
    "logger",
    "run_logger",
    "database.db",
    "database.models",
    "routes.projects",
    "routes.search",
]

datas = [
    # elevenlabs_client 走 certifi.where() 建 SSL context，证书必须打进去
    *collect_data_files("certifi"),
]

binaries = [
    # faiss 的 _swigfaiss 扩展及其依赖库
    *collect_dynamic_libs("faiss"),
]

a = Analysis(
    ["main.py"],
    pathex=[str(BACKEND_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 开发环境里装着但后端运行时用不到的重量级依赖。
        # 不排掉的话 onefile 会从几十 MB 涨到几百 MB
        # （历史上那个 601MB 的产物就是被 torch 拖大的）。
        "torch",
        "sentence_transformers",
        "transformers",
        "matplotlib",
        "pandas",
        "scipy",
        "notebook",
        "IPython",
        "jupyter",
        "tkinter",
        "PyQt5",
        "PySide2",
        "PIL",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="backend",
    debug=False,
    bootloader_ignore_signals=False,   # 必须 False：App 靠 SIGTERM 关闭后端
    strip=False,
    upx=False,                          # UPX 会破坏 macOS 代码签名
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,                   # 跟随构建机架构
    codesign_identity=None,             # 签名由外层 App 的构建流程处理
    entitlements_file=None,
)
