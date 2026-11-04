# MindSpore 与 torch_npu 联合 wheel

两套源码独立使用原生构建入口。MindSpore 使用自身的 CMake、`build.sh`、`setup.py` 和 CPack，
torch_npu 当前临时默认下载官方 2.13.0 wheel，也可显式选择原生 `ci/build.sh` 编译。外层脚本仅串联构建、下载和合包。

```text
build.sh                           → output/mindspore-*.whl
third_party/torch_npu/ci/build.sh   → third_party/torch_npu/dist/torch_npu-*.whl
或官网 torch_npu wheel 下载直链     → 临时下载目录（退出时清理）
scripts/packaging/merge_whl.py      → output/mindspore-*.whl
```

最终包名与版本以 MindSpore 的 `version.txt` 为准。下文用 `xxx` 表示版本号，
产物名为 `mindspore-xxx-<Python 标签>-<ABI 标签>-<平台标签>.whl`。
完整包含 MindSpore、torch_npu 及 torch_npu 原生 wheel 携带的其他模块；不携带 PyTorch。
MindSpore 保持原生 C++17 / ABI 0；源码构建的 torch_npu 保持原生 C++20 / 当前 PyTorch ABI，
下载的 torch_npu 保持官方二进制及其 PyTorch 兼容要求。
两者不链接彼此的 C++ 库，不跨框架传递 C++ 对象。

## 内置 torch_npu 与独立安装

合包将 torch_npu 原生文件完整放入 `mindspore/_bundled/torch_npu/`，保留 `torch_npu`、
`torchnpugen` 等模块名和内部目录结构。
`import torch` 通过原生 `torch.backends` 入口调用 `_mindspore_torch_npu_loader.autoload`：
优先使用正常导入路径中的用户 torch_npu；找不到时才将私有目录追加到 `sys.path`，
导入内置 `torch_npu` 并调用原生 `_autoload`。普通 Python 启动和标准库导入不执行此辅助模块。
仅安装联合包时，须先 `import torch`，随后可以正常 `import torch_npu`；首次直接
`import torch_npu` 不提供内置回退。独立安装的 torch_npu 保持其原生导入行为。

独立安装、卸载 torch_npu 不会覆盖内置文件。卸载外部 torch_npu 后，新启动的 Python 进程
回退到内置版本；运行中的进程不切换。外部 torch_npu 存在但加载失败时直接报错，不混用内置文件。
PyTorch 的 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 开关仍生效；关闭后若要使用内置 torch_npu，
须显式调用 `_mindspore_torch_npu_loader.autoload()`，再导入 `torch_npu`。
统一包保留原生命令名 `torch_npu_run`，包装函数先导入 `torch` 并加载 torch_npu，
再调用选中版本的原生 torch_npu 启动函数（也支持关闭后端自动加载的环境）。
两种安装顺序下，执行时都优先使用独立安装的 torch_npu；命令脚本本身由后安装的 wheel 写入。
两个发行包均登记同一个命令文件，卸载任意一方可能删除 `torch_npu_run`。
此时需重新安装保留的发行包以恢复命令；Python 模块不共享文件，导入规则不受影响。

## 构建命令

查看参数说明，或使用默认 MindSpore 参数 `-e ascend -V 910b` 构建：

```bash
bash scripts/build_unified.sh --help
bash scripts/build_unified.sh
```

未指定 `-e` 时默认构建 Ascend；此时未指定 `-V` 才补充 `910b`。
因此只传 `-j48` 仍构建默认 Ascend 包；显式指定设备时沿用 MindSpore 原生参数规则。

原来的构建命令无需增加参数：当前临时默认下载 torch_npu 2.13.0，编译 MindSpore 后合包。
`-j` 控制 MindSpore；`MAX_JOBS` 仅在源码编译 torch_npu 时生效：

```bash
MAX_JOBS=48 bash scripts/build_unified.sh -e ascend -V 910b -j48
```

显式使用 `--torch-npu-source` 恢复双源码构建，再选择 Python、校验 PyTorch 版本或调整组件：

```bash
bash scripts/build_unified.sh --torch-npu-source --python=3.10 --torch=2.13.0 -e ascend -V 910b -j48
bash scripts/build_unified.sh --torch-npu-source --clean --disable_fxrt -e ascend -V 910b -j48
```

`--python` 接受解释器版本或路径，相对路径以调用时的工作目录为准；`--torch` 校验已安装 PyTorch 的版本，不安装 PyTorch，
也不切换 torch_npu 源码。`--torch`、`--disable_torchair`、`--disable_fxrt`、`--disable_rpc`、
`--enable_lto`、`--enable_pgo=1|2` 传入原生 torch_npu 构建脚本，其余参数传给 MindSpore `build.sh`。
`--clean` 调用 torch_npu 原生清理；改变组件、编译器或 ABI 后使用。
MindSpore 的缓存清理遵循其原生构建流程。

只编译 MindSpore，自动选择官方 torch_npu wheel：

```bash
bash scripts/build_unified.sh --torch-npu-wheel=2.13.0 -e ascend -V 910b -j48
```

脚本通过 curl 查询 GitCode 官方 Release API，按选定 Python 的版本/ABI 和当前机器架构
（如 `aarch64`、`x86_64`）选取 Linux wheel，通过解析出的 HTTPS 直链调用 curl 下载。
不调用 pip，不按 CI 本机 glibc 过滤文件；无匹配包或有多个候选时，在编译前报错，
不自动切换版本。日志输出实际下载 URL，便于核对版本和平台。
当前将 `2.13.0` 固定映射到 `v26.2.0-pytorch2.13.0`，避免同一版本匹配多个 Ascend 发行版。
后续版本需更新 `scripts/packaging/resolve_torch_npu_wheel.py` 的 `RELEASE_TAGS`，或直接传入官方 wheel URL。
查询和附件下载均使用 curl，保留 HTTPS 证书校验；CI 如有自定义 CA，需配置 curl 的可信 CA。

默认下载合包模式不需要安装 PyTorch，根目录 `requirements.txt` 不再引入 torch_npu 源码构建依赖。
CI 前置步骤中的独立 `pip install torch==2.13.0` 也应跳过；需要源码编译时显式安装对应依赖：

```bash
bash scripts/build_unified.sh --prepare-only
python -m pip install -r third_party/torch_npu/requirements_2.13-build.txt
bash scripts/build_unified.sh --torch-npu-source -e ascend -V 910b -j48
```

也可手动指定直链。将 `TORCH_NPU_WHEEL_URL` 替换为
所选版本、Python 和架构对应的 HTTPS wheel 直链：

```bash
bash scripts/build_unified.sh --torch-npu-wheel="$TORCH_NPU_WHEEL_URL" -e ascend -V 910b -j48
```

`--torch-npu-wheel` 接受版本号或完整下载 URL；URL 路径末尾须为 `torch_npu-*.whl`，支持 URL 编码和查询参数，
与子模块的 commit 无关。脚本通过 `curl` 下载，不调用 pip、不安装或修改当前环境的包。
下载成功后才编译 MindSpore，最后复用同一个合包入口；下载失败直接退出，不回退到源码编译。
此模式跳过 torch_npu 子模块初始化、编译和本地 PyTorch ABI 探测；不能同时传入 `--prepare-only`、
`--clean` 或上述 torch_npu 构建选项。MindSpore 自身的构建依赖仍需满足。
下载目录在退出时清理；最终产物仍位于 `output/`。
运行时须使用与所选官方 wheel 匹配的 PyTorch、CANN 和系统运行库；例如 2.13.0 的
`manylinux_2_28` wheel 要求 glibc ≥ 2.28，合包不会降低这一要求。
GitCode 的 `v26.2.0-pytorch2.13.0` Release 当前注明配套 CANN 9.2.0；下载合包成功不代表
已验证其与 CANN 9.0.0 的运行兼容性。
下载与合包不加载 torch_npu 原生库，因此 glibc 2.17 的 CI 可下载并合入 `manylinux_2_28` wheel，
但不能据此认为该 CI 可以运行内置 torch_npu。请按最终部署环境选择官方包；
合包检查 Python/ABI 和架构标签，不验证 CANN、PyTorch 或运行库的实际兼容性。

原生 MindSpore wheel 先生成到 `output/`；合包完成后，以联合 wheel 替换同名原生包，并更新
对应的 `.whl.sha256`，最终 `output/` 只保留联合 wheel，不另建 `unified/` 子目录。
合包器在临时目录完成打包后才替换原生 wheel，避免打包失败提前丢失原生包。
源码构建的 torch_npu wheel 仍位于 `third_party/torch_npu/dist/`。
MindSpore 打包会重建 `output/`，再次构建前请保存历史产物。
失败时以脚本退出码为准，不使用残留 wheel。使用 NumPy 1.x 构建时，脚本给最终包补充 `numpy<2` 运行依赖。

## 源码版本

`third_party/torch_npu` 是 Git 子模块，其 gitlink 是唯一版本锁；没有补丁清单或私有提交。
显式选择源码构建时，先自动执行指定子模块的递归初始化；使用官方 wheel 时跳过。单独下载源码：

```bash
bash scripts/build_unified.sh --prepare-only
# 维护者更新到新的公开提交，再提交父仓库的 gitlink
git -C third_party/torch_npu fetch origin <commit>
git -C third_party/torch_npu checkout --detach <commit>
git -C third_party/torch_npu submodule update --init --recursive
git add third_party/torch_npu
```

## 已有 wheel 合包（含 GPU）

合包工具只需 Python 3.10+ 及 setuptools、wheel、packaging；打包用 Python 无需匹配输入 wheel 的 Python 版本。
不需要 PyTorch、CANN、编译器或源码子模块，也可单独拷贝脚本运行；默认输出到当前目录的 `output/`。
先提供核心 MS wheel，再提供其他后端：

```bash
python scripts/packaging/merge_whl.py ms-ascend/mindspore-*.whl \
    ms-gpu/mindspore-*.whl --module-wheel torch_npu/torch_npu-*.whl --output-dir merged
# 也支持把已含 torch_npu 的统一 wheel 与 GPU wheel 再合并
bash scripts/build/merge_whl_package.sh output/mindspore-*.whl gpu/mindspore-*.whl --output-dir merged
```

MS 输入必须同版本、同 Python/平台标签，torch_npu 的 Python/ABI 标签须匹配。
平台标签通常须一致；`linux_<架构>` 的 MS wheel 也接受同架构的 `manylinux` torch_npu wheel，
输出文件名和 `WHEEL` 元数据保持 MindSpore 原有平台标签。合包保留双方原生库，
不替换或捆绑系统 glibc；使用双方组件的运行环境须满足较高的 glibc 要求。
例如 MindSpore 要求 glibc 2.17、torch_npu 要求 2.28，则共同运行要求 glibc ≥ 2.28。
`linux_<架构>` 标签不编码这一最低版本，pip 不会据此阻止旧系统安装；
发布和部署时仍需确认双方的系统库依赖。
保留原有 GPU 插件、MPI、GE
及 dataengine 合并规则；单个 MindSpore wheel 保留原有库布局，多个 MindSpore wheel 才执行后端归并。
torch_npu 完整复制，冲突时报错。依赖、入口、许可来自原生 wheel
标准元数据；重建单一 dist-info、RECORD 和 SHA256。若手工合并 NumPy 1.x 构建产物，
追加 `--require 'numpy<2'`；合包器不推测构建环境。

## 安装与验证

```bash
pip install output/mindspore-*.whl
# MindSpore 不要求安装 torch；使用 torch_npu 时另装匹配 PyTorch，或安装 mindspore[torch_npu]。
python scripts/packaging/check_unified.py --wheel output/mindspore-*.whl
python scripts/packaging/check_unified.py --runtime --device 0
# 验证内置 torch_npu；另装独立 torch_npu 后，改为 --torch-npu-source external
python scripts/packaging/check_unified.py --runtime --torch-npu-source bundled --device 0
# 独立无 torch 环境
python scripts/packaging/check_unified.py --runtime --require-no-torch --device 0
```

检查覆盖独立导入、两种导入顺序、Ascend 前反向。两个框架同时操作同一设备不属于默认检查；
`--shared-ascend` 可单独测试。GPU 合包结构验证不能替代 GPU 设备上的运行测试。
