---
name: ms-build
description: Build MindSpore from source, configure build options, do incremental builds, or troubleshoot compilation errors. Use when compiling MindSpore for Ascend/GPU/CPU, setting build flags, building only specific components, enabling debug mode, or running build scripts. Triggers on phrases like "build mindspore", "compile", "build.sh", "cmake", "make", "build error", "compilation failed", "how to build".
---

# MindSpore Build Guide

## Quick Start

```bash
# CPU only (fastest, no hardware required)
bash build.sh -e cpu -j8

# GPU (requires CUDA)
bash build.sh -e gpu -j8

# Ascend (requires CANN toolkit)
bash build.sh -e ascend -j8

# Debug mode (slower but with symbols, asserts, ASAN-ready)
bash build.sh -e gpu -d -j8
```

After building, install the wheel:
```bash
pip install output/mindspore_*.whl --force-reinstall
```

## Key Build Options

| Flag                     | Description                    | Default    |
|--------------------------|--------------------------------|------------|
| `-e cpu\|gpu\|ascend\|d` | Target device                  | (required) |
| `-j[N]`                  | Parallel jobs                  | `-j8`      |
| `-d`                     | Debug mode (with symbols)      | off        |
| `-r`                     | Release mode                   | on         |
| `-i`                     | Incremental build (skip clean) | off        |
| `-v`                     | Verbose (show cmake commands)  | off        |
| `-a on`                  | Enable ASAN (AddressSanitizer) | off        |
| `-t on`                  | Build unit test binary         | off        |
| `-g on\|off`             | Use glog for logging           | on         |
| `-D on\|off`             | Dump IR graphs to file         | on         |

## Incremental Build (Fast Rebuild)

After the first full build, use `-i` to skip the clean step:

```bash
bash build.sh -e gpu -j8 -i
```

Or rebuild directly with ninja/make in the build directory:
```bash
cd build/mindspore
cmake --build . --target install -j8
```

## Build Only Specific Components

```bash
# Build only the Python wheel (skip tests)
cd build/mindspore
ninja mindspore/python/mindspore/_c_expression.so -j8

# Build a specific target
cmake --build build/mindspore --target <target_name> -j8
```

## Environment Setup

### For GPU builds
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### For Ascend builds
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## Code Generation (Required After Modifying YAML Op Definitions)

When adding or changing operators in `mindspore/ops/op_def/yaml/` or `mindspore/ops/api_def/`, regenerate the auto-generated files:

```bash
cd mindspore/python/mindspore/ops_generate
python gen_ops.py
```

Or let the full build handle it automatically.

## Common Build Errors

### Missing CANN / CUDA headers
```
fatal error: acl/acl.h: No such file or directory
```
→ Source the Ascend toolkit env or check CUDA path.

### CMake cache stale after option change
```bash
rm -rf build/mindspore/CMakeCache.txt
bash build.sh -e gpu -j8
```

### Linker errors on new C++ files
If you added new `.cc` files, ensure they're listed in the relevant `CMakeLists.txt`:
```bash
grep -r "ops_func_impl" mindspore/ops/CMakeLists.txt
```
Add your file to the appropriate `set(OBJ_LIST ...)` block.

### Python import error after rebuild
```bash
pip install output/mindspore_*.whl --force-reinstall
python -c "import mindspore; print(mindspore.__version__)"
```

## Build Output

```
output/
├── mindspore-{version}-cp39-*.whl   # Install this
└── _CPack_Packages/                 # Intermediate packaging files
```

## Running Tests

```bash
# Unit tests (C++) — two steps: build the binaries, then run them
bash build.sh -t on -j8        # -t on only BUILDS the tests; do NOT add -e (build.sh resets device to empty with -t)
bash tests/ut/cpp/runtest.sh   # runs the built ut_*_tests binaries

# System tests (Python)
pytest tests/st/ops/test_{op_name}.py -v

# Single test
python tests/st/ops/test_{op_name}.py
```

## Useful cmake Variables

When debugging build issues, check `build/mindspore/CMakeCache.txt`:
```
ENABLE_GPU:BOOL=ON
CMAKE_BUILD_TYPE:STRING=Debug
PYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3
```
