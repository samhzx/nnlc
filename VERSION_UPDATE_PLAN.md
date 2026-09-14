# NNLC 简单版本更新计划

## 1. 目标

保持当前发布结构：

```text
NNLC_Trainer.exe
NNLC_Runtime/
  runtime-manifest.json
  julia-runtime/
  julia-depot/
```

只增加简单的 EXE 更新：

- GUI 显示当前版本。
- 启动 GUI 后后台检查一次更新。
- 有新版本时显示更新说明。
- 把带版本号的新 EXE 下载到当前程序所在目录。
- 下载时显示进度。
- 下载失败时删除下载文件。
- 不关闭、不替换正在运行的 EXE。
- 保留原来的 `NNLC_Runtime`。

不实现签名、证书、自动替换当前 EXE，或 Julia Runtime 自动更新。

## 2. 版本管理

`pyproject.toml` 是唯一版本来源：

```toml
version = "1.0.0"
```

构建时生成：

```text
build/version-assets/
  build_info.json
  version_info.txt
  version.txt
```

`build_info.json`：

```json
{
  "version": "1.0.0"
}
```

用途：

- GUI 显示 `v1.0.0`。
- `NNLC_Trainer.exe --version` 输出 `1.0.0`。
- 更新检查使用该版本进行比较。
- Windows EXE 文件版本使用 `1.0.0.0`。

映射规则：

```text
1.2.3 -> 1.2.3.0
```

新增：

```text
nnlc_version.py
build_tools/__init__.py
build_tools/generate_version_assets.py
```

冻结后的 EXE 从内嵌 `build_info.json` 读取版本；源码运行时从
`pyproject.toml` 读取版本。

## 3. update.json

固定地址：

```text
https://file.897242746.xyz/data/NNLC_Trainer/update.json
```

只保留以下五个字段：

```json
{
  "version": "1.0.1",
  "size": 86543210,
  "filename": "NNLC_Trainer-1.0.1-windows-x64.exe",
  "notes": "修复训练流程问题。",
  "sha256": "64位小写十六进制摘要"
}
```

客户端只做基本检查：

- `version` 是三段式版本，例如 `1.0.1`。
- `size` 是正整数。
- `filename` 是单个 EXE 文件名，格式为
  `NNLC_Trainer-<version>-windows-x64.exe`。
- 文件名中的版本与 `version` 相同。
- `notes` 是字符串。
- `sha256` 是 64 位小写十六进制。
- 其他额外字段忽略。

下载地址：

```text
https://file.897242746.xyz/data/NNLC_Trainer/<filename>
```

版本使用数字比较，例如 `1.10.0` 大于 `1.9.0`。远端版本小于或等于当前版本时
不提示。

## 4. 代码改动

新增：

```text
nnlc_update.py
tests/test_version_info.py
tests/test_update.py
```

修改：

```text
nnlc_auto_train.py
nnlc_gui.py
nnlc_windows.spec
build_windows.ps1
.github/workflows/build-windows.yml
README.md
GUI使用说明.html
```

`nnlc_update.py` 负责：

- 下载和解析 `update.json`。
- 比较版本。
- 把新 EXE 下载到当前程序所在目录。
- 回传下载进度。
- 检查文件大小和 SHA-256。
- 下载失败时立即删除下载文件。

网络请求使用标准库 `urllib.request`，不增加网络依赖。

不实现 `--apply-update`、等待旧进程、备份旧 EXE 或自动重启。

## 5. 程序入口

`nnlc_auto_train.py:main()` 的处理顺序保持现有逻辑，只增加公开参数：

```text
NNLC_Trainer.exe --version
```

`--version`、`--help` 和 `--run-module` 都跳过 Julia 环境检查。

自动检查更新只在打包后的 Windows GUI 中执行。源码运行和命令行训练不检查更新。

## 6. GUI 流程

在 `nnlc_gui.py` 中：

1. 在当前顶部标题右侧显示 `v{version}`。
2. GUI 启动约 1 秒后创建后台线程检查更新。
3. 后台线程通过现有 `self.messages` 队列返回结果，不阻塞 Tkinter。
4. 没有更新或网络请求失败时不弹窗。
5. 有更新时显示：
   - 当前版本。
   - 最新版本。
   - 文件大小。
   - `notes`。
   - “立即下载”和“稍后”。
6. 用户选择“立即下载”后显示下载进度窗口，包含进度条、已下载大小、总大小、
   百分比和“取消”。
7. 下载期间禁用“开始训练”。关闭主窗口或进度窗口时，先确认是否取消下载。
8. 下载失败或用户取消后关闭进度窗口，提示失败原因，并删除未完成文件。
9. 下载成功后提示新文件路径，说明关闭当前程序后运行新 EXE 即可。可提供
   “打开所在目录”。

如果 `self.worker.is_alive()`，先保存更新信息，训练结束后再提示。训练期间不下载
EXE。

## 7. 下载

当前程序所在目录指 EXE 所在目录，不是进程工作目录。下载目标：

```text
<EXE目录>\NNLC_Trainer-<version>-windows-x64.exe
```

下载过程：

1. 如果目标文件已存在，并且大小和 SHA-256 都匹配，则跳过下载，直接提示文件
   已就绪。
2. 否则先下载到同目录的临时文件：

   ```text
   <EXE目录>\NNLC_Trainer-<version>-windows-x64.exe.part
   ```

3. 下载过程中定期回传进度：已下载字节、总大小和百分比。进度回调大约每 200ms
   一次。总大小使用 `update.json` 的 `size`。
4. 同时计算实际大小和 SHA-256。
5. 实际下载字节超过 `size` 时立即中止。
6. 下载完成后与 `update.json` 的 `size`、`sha256` 比较。
7. 校验成功后将 `.part` 改名为正式版本化 EXE。
8. 下载失败、网络中断、超时、用户取消、大小不符或 SHA-256 不符时，立即删除
   `.part` 和未完成的目标文件，不保留半成品。

清单请求和文件下载都设置超时。目录不可写时提示失败，不开始下载。

因为新文件带版本号，不会覆盖正在运行的 EXE，所以不需要等待旧程序退出，也不
需要 `.bak` 替换。

用户关闭当前程序后，双击新的版本化 EXE 即可。新 EXE 仍从所在目录读取
`NNLC_Runtime`。旧 EXE 可手动删除。

## 8. Julia Runtime

自动更新只用于 Julia 环境没有变化的日常 EXE 更新。

- Julia 版本和依赖不变：发布新的 EXE 和 `update.json`。
- Julia 版本或依赖变化：不要使用自动更新，手动发布新的 EXE 和 Runtime ZIP。

以下现有内容保持不变：

```text
nnlc_runtime.py
windows_runtime.json
build_julia_runtime.ps1
.github/workflows/build-windows-runtime.yml
```

## 9. 构建和发布

`nnlc_windows.spec`：

- 打入 `build_info.json`。
- 使用 `version_info.txt` 设置 Windows 文件版本。
- 将 `nnlc_version` 和 `nnlc_update` 加入 hiddenimports。
- 保持当前 one-file 和外置 Julia 配置。

`build_windows.ps1`：

1. 清理旧 `build` 和 `dist`。
2. 运行版本资源生成工具。
3. 构建 `dist/NNLC_Trainer.exe`。
4. 检查 `dist` 只有单个 EXE。
5. 测试 `--help`、`--version` 和现有 `--run-module` worker。

`.github/workflows/build-windows.yml` 增加 `notes` 输入，然后：

1. 构建并测试 EXE。
2. 复制为：

   ```text
   release/NNLC_Trainer-<version>-windows-x64.exe
   ```

3. 计算文件大小和 SHA-256。
4. 生成只包含五个字段的 `release/update.json`。
   SHA-256 使用小写十六进制，`notes` 允许为空字符串。
5. 上传 EXE 和 `update.json`。

文件站发布时先上传版本化 EXE，最后上传 `update.json`。

首次安装仍可把发布文件重命名为 `NNLC_Trainer.exe`。之后更新会在同目录下载
带版本号的新 EXE，不替换当前正在运行的文件。

## 10. 测试与验收

自动测试：

- 版本读取和三段式版本比较。
- `update.json` 五个字段校验。
- 文件名版本一致性。
- 下载到指定目录的版本化文件名。
- 文件大小和 SHA-256 校验。
- 下载进度回调。
- 已存在且校验通过的文件跳过下载。
- 下载失败、中断或取消后删除 `.part` 和未完成文件。
- `--help`、`--version` 和 worker 不需要 Julia 环境。

人工验收：

1. GUI 显示正确版本。
2. 无更新或断网时正常启动。
3. 有更新时弹出提示。
4. 下载时显示进度，可取消。
5. 新 EXE 下载到当前程序目录，文件名带版本号。
6. 下载失败或损坏时删除下载文件，正在运行的 EXE 不受影响。
7. 关闭当前程序后运行新 EXE，继续使用原来的 `NNLC_Runtime`。
8. 训练期间不执行下载。

## 11. 实施顺序

1. 版本显示、构建版本资源和 `--version`。
2. `update.json` 获取和版本比较。
3. EXE 下载到当前目录、进度显示、失败清理、大小和 SHA-256 校验。
4. GUI 后台检查和下载进度窗口。
5. Windows 构建工作流和使用说明。
