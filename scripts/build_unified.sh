#!/usr/bin/env bash
# Build MindSpore, build or download torch_npu, then merge the native wheels.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_command=${MS_PYTHON_EXECUTABLE:-python}
torch_npu_args=()
ms_args=()
prepare_only=false
clean=false
device_selected=false
version_selected=false
torch_npu_url=
torch_npu_source=false
for argument in "$@"; do
    case "$argument" in
        --python=*)
            python_command=${argument#*=}
            [[ ! "$python_command" =~ ^[0-9]+\.[0-9]+$ ]] || python_command="python$python_command"
            ;;
        --torch=*|--disable_torchair|--disable_fxrt|--disable_rpc|--enable_lto|--enable_pgo=*)
            torch_npu_args+=("$argument") ;;
        --torch-npu-wheel=*)
            torch_npu_url=${argument#*=}
            [[ -n "$torch_npu_url" ]] || { echo '--torch-npu-wheel requires a version or HTTPS wheel URL' >&2; exit 1; }
            ;;
        --prepare-only) prepare_only=true ;;
        --torch-npu-source) torch_npu_source=true ;;
        --clean) clean=true ;;
        -h|--help)
            echo "Usage: $0 [--python=PATH|VERSION] [--prepare-only] [--clean] [torch_npu options] [MindSpore build.sh options]"
            echo 'torch_npu options: --torch=VERSION --disable_torchair --disable_fxrt --disable_rpc --enable_lto --enable_pgo=1|2'
            echo '--torch-npu-wheel=VERSION|URL: automatically select an official wheel, or use an HTTPS wheel URL.'
            echo 'Default: download torch_npu 2.13.0. Use --torch-npu-source to build from source.'
            echo 'Cannot combine --torch-npu-wheel with --prepare-only, --clean or torch_npu build options.'
            echo 'Default MindSpore options: -e ascend -V 910b; all other arguments go to native build.sh.'
            exit 0 ;;
        -e*) device_selected=true; ms_args+=("$argument") ;;
        -V*) version_selected=true; ms_args+=("$argument") ;;
        *) ms_args+=("$argument") ;;
    esac
done
if $torch_npu_source && [[ -n "$torch_npu_url" ]]; then
    echo '--torch-npu-source cannot be combined with --torch-npu-wheel' >&2
    exit 1
fi
# Temporary default for CI; --prepare-only retains its source preparation behavior.
if [[ -z "$torch_npu_url" ]] && ! $torch_npu_source && ! $prepare_only; then
    torch_npu_url=2.13.0
fi
if [[ -n "$torch_npu_url" ]] && { $prepare_only || $clean || [[ ${#torch_npu_args[@]} != 0 ]]; }; then
    echo '--torch-npu-wheel cannot be combined with --prepare-only, --clean or torch_npu build options' >&2
    exit 1
fi
# Resolve relative paths before changing directory, without following a venv's Python symlink.
if ! $prepare_only; then
    requested_python=$python_command
    if ! python_command=$(command -v -- "$requested_python") || [[ ! -x "$python_command" || -d "$python_command" ]]; then
        echo "Python executable not found or not executable: $requested_python" >&2
        exit 1
    fi
    python_command="$(cd -- "$(dirname -- "$python_command")" && pwd)/$(basename -- "$python_command")"
fi
cd "$root"
if [[ -z "$torch_npu_url" ]]; then
    git submodule update --init --recursive -- third_party/torch_npu
fi
$prepare_only && exit 0
export PATH="$(dirname "$python_command"):$PATH"
python_version=$("$python_command" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
selected_prefix=$("$python_command" -c 'import sys; print(sys.prefix)')
if [[ -n "${CONDA_PREFIX:-${VIRTUAL_ENV:-}}" && "${CONDA_PREFIX:-${VIRTUAL_ENV:-}}" != "$selected_prefix" ]]; then
    echo 'Activate the selected Python environment before building (native MindSpore uses the active environment)' >&2
    exit 1
fi
# Both native entry points must discover the selected environment's interpreter.
for entry in python "python$python_version"; do
    if ! entry_prefix=$("$entry" -c 'import sys; print(sys.prefix)'); then
        echo "Cannot run Python entry point: $entry (selected: $python_command)" >&2
        exit 1
    fi
    if [[ "$entry_prefix" != "$selected_prefix" ]]; then
        echo "Python environment mismatch: $entry uses $entry_prefix; $python_command uses $selected_prefix" >&2
        exit 1
    fi
done
shopt -s nullglob
if [[ -n "$torch_npu_url" ]]; then
    if [[ "$torch_npu_url" != https://* ]]; then
        torch_npu_url=$("$python_command" scripts/packaging/resolve_torch_npu_wheel.py "$torch_npu_url")
    fi
    echo "Using torch_npu wheel: $torch_npu_url"
    # Download without loading the module or filtering by the build host's glibc.
    wheel_name=$("$python_command" - "$torch_npu_url" <<'PY'
from pathlib import PurePosixPath
import sys
from urllib.parse import unquote, urlsplit
from packaging.utils import parse_wheel_filename

name = PurePosixPath(unquote(urlsplit(sys.argv[1]).path)).name
if parse_wheel_filename(name)[0] != 'torch-npu':
    sys.exit('Expected a torch_npu wheel URL')
print(name)
PY
    )
    download_dir=$(mktemp -d "${TMPDIR:-/tmp}/mindspore-torch-npu.XXXXXX")
    trap 'rm -rf -- "$download_dir"' EXIT
    curl --fail --location --retry 3 --proto '=https' --proto-redir '=https' \
        --output "$download_dir/$wheel_name" "$torch_npu_url"
    torch_npu_wheels=("$download_dir/$wheel_name")
else
    # Source builds need the ABI of installed PyTorch; MindSpore keeps its native ABI.
    torch_npu_abi=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$python_command" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
    if [[ -n "${_GLIBCXX_USE_CXX11_ABI:-}" && "$_GLIBCXX_USE_CXX11_ABI" != "$torch_npu_abi" ]]; then
        echo 'torch_npu ABI override does not match installed PyTorch' >&2
        exit 1
    fi
fi
if ! $device_selected; then
    # Bash < 4.4 treats empty arrays as unset under nounset; guard optional arguments.
    ms_args=(-e ascend ${ms_args[@]+"${ms_args[@]}"})
    if ! $version_selected; then
        ms_args=(-V 910b ${ms_args[@]+"${ms_args[@]}"})
    fi
fi
bash build.sh ${ms_args[@]+"${ms_args[@]}"}
ms_wheels=("$root"/output/*.whl)
[[ ${#ms_wheels[@]} == 1 ]] || { echo 'Expected one native MindSpore wheel' >&2; exit 1; }
if [[ -z "$torch_npu_url" ]]; then
    (
        cd third_party/torch_npu
        export _GLIBCXX_USE_CXX11_ABI="$torch_npu_abi"
        if $clean; then
            "$python_command" setup.py clean
        fi
        bash ci/build.sh "--python=$python_version" ${torch_npu_args[@]+"${torch_npu_args[@]}"} || exit "$?"
    )
    torch_npu_wheels=("$root"/third_party/torch_npu/dist/*.whl)
    [[ ${#torch_npu_wheels[@]} == 1 ]] || { echo 'Expected one native torch_npu wheel; clear stale dist/ wheels and retry' >&2; exit 1; }
fi
# NumPy 1.x headers require a NumPy 1.x runtime. This fact belongs to the build.
requirements=()
if [[ $("$python_command" -c 'import numpy; print(numpy.__version__.split(".")[0])') == 1 ]]; then
    requirements+=(--require 'numpy<2')
fi
"$python_command" scripts/packaging/merge_whl.py ${requirements[@]+"${requirements[@]}"} \
    --module-wheel "${torch_npu_wheels[0]}" --output-dir "$root/output" "${ms_wheels[0]}"
