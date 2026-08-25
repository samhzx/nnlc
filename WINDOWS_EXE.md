# Windows 单文件程序

`NNLC_Trainer.exe` 是 Tkinter 操作界面。双击后选择 reallog 目录、输出目录，填写车型；路线阈值保持“自动推荐”即可。训练中间 CSV、覆盖度图、验证图和最终车型 JSON 都会写到输出目录。

## 构建

由于 PyInstaller 不能把 macOS 的 Python/Julia 变成 Windows 程序，必须在 Windows 电脑上执行构建。需要一台 Windows x64 机器和 Julia 1.10.11 x64 压缩包（解压目录应直接包含 `bin\julia.exe`）。在 PowerShell 中运行：

```powershell
cd C:\path\to\openpilot-nnlc-tools
.\build_windows.ps1 -JuliaDir C:\tools\julia-1.10.11
```

脚本会把 Julia 运行时和全部 Julia 包放进 exe。首次安装包和 PyInstaller 分析可能需要较长时间，生成的单文件通常超过 1 GB，这是为了让目标电脑不需要安装 Python、Julia 或依赖包。产物位于 `dist\NNLC_Trainer.exe`，可直接复制到没有开发环境的 Windows 电脑运行。

如果已经准备好同一台 Windows 机器上的 Julia depot，可用 `-JuliaDepot C:\path\to\julia-depot -SkipJuliaPackages` 直接复用；depot 必须包含训练脚本需要的包和 artifacts。

## 命令行模式

开发调试时仍可使用 Python：

```powershell
python nnlc_auto_train.py --data D:\rlogs --car BYD_TANG_DMI_24 --output D:\nnlc-output
```

exe 的命令行模式也支持 `--data`、`--car`、`--min-score`、`--output`、`--skip-deploy` 和 `--skip-viz`；不带参数双击则启动界面。
