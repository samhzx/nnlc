# C3 rlog 下载器

这是一个独立的 Windows 客户端源码目录。应用不依赖 NNLC 的模块、运行时或训练依赖；构建时只安装本目录 `requirements-build.txt` 中的 Paramiko、PyInstaller 和图标生成用 Pillow。Tkinter 由 Windows Python 发行版提供，Python 运行时会打包进单文件 EXE。

## 构建

在 Windows 上安装 Python 3.11，并在 PowerShell 中运行：

```powershell
cd c3_rlog_downloader
.\build_windows.ps1
```

产物位于 `c3_rlog_downloader\dist\C3_Rlog_Downloader.exe`。构建环境、临时文件和发布目录都限制在本目录中，不会清理或覆盖 NNLC 的 `build`、`dist` 或 `release`。

图标设计源文件为 `make_icon.py`，多分辨率 Windows 图标为 `C3_Rlog_Downloader.ico`。构建时会自动重新生成，并用于 EXE 和 Windows 窗口图标；`C3_Rlog_Downloader_icon.png` 是预览图。

## GitHub Actions 构建

在仓库的 Actions 页面手动运行 `Build C3 Rlog Downloader` workflow。成功后，从该次运行的 Artifacts 下载 `C3_Rlog_Downloader-windows-x64`。

## 使用

1. 输入 C3 的 IP，选择 SSH 私钥，并选择本地下载目录。
2. 客户端以 `comma` 用户通过 SSH 22 端口连接设备，扫描 `/data/media/0/realdata`。
3. 只下载名称完全为 `rlog.zst` 的文件，按本机开始导出时的系统日期创建 `YYYYMMDD` 目录，并保留 `realdata` 下的原有相对目录。例如 `20261016/000000a0--68a50e9c1b--2/rlog.zst`。不使用路线目录日期或远端文件修改日期分类。
4. 最近 5 分钟修改过的文件会被跳过。下载成功后，客户端会显示清理范围和文件数量；确认清理会删除 `realdata` 中其他未下载的旧文件，并删除因此变空的路线目录，但会保留近期文件。

首次连接会自动接受并记录设备 SSH 主机指纹；同一 IP 的指纹之后发生变化时会拒绝连接。应用只记住 IP、密钥路径、本地下载目录和训练结果 JSON 路径，不保存私钥内容或口令。连接设备后可将所选 JSON 覆盖上传到 `/data/openpilot/sunnypilot/neural_network_data/neural_network_lateral_control`。密钥文件始终从用户选择的本机路径读取。

## 测试

从仓库根目录运行：

```powershell
py -3.11 -m pytest c3_rlog_downloader\tests
```
