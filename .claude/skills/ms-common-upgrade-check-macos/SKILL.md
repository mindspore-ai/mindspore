---
name: ms-upgrade-check-macos
description: Check that MindSpore can still build from source and macOS Apple Silicon CPU CI can still pass after a macOS or toolchain upgrade. Use when validating macOS upgrades, Xcode, Python, or CMake upgrades, or post-upgrade regressions on Apple Silicon. Triggers on phrases like "macos upgrade check", "verify build after upgrade", "check ci after system upgrade", "post-upgrade compile validation", "apple silicon regression check".
---

# MindSpore macOS Post-Upgrade Check

## Overview

Systematic workflow for checking that, after a macOS Apple Silicon system upgrade, MindSpore still builds from source and macOS CPU CI-style validation still passes. Core principle: verify the upgraded environment first, then rebuild and run the relevant validation path to catch regressions early.

## When to Use

**Use this skill when:**
- Checking MindSpore after a macOS system upgrade on Apple Silicon
- Verifying that Xcode, Python, CMake, or dependency upgrades did not break macOS source builds
- Confirming that MindSpore can still compile and macOS CPU CI-style tests can still pass after environment changes
- Troubleshooting post-upgrade build failures or CI regressions on Apple Silicon
- Keywords: "system upgrade check", "macos upgrade verify", "verify build after upgrade", "check ci after upgrade", "apple silicon regression"

**Don't use for:**
- Installing pre-built MindSpore packages (use pip/conda instead)
- Linux or Windows compilation (different toolchain)
- Runtime errors after successful installation

## Quick Reference

| Stage | Key Command | Verification |
|-------|-------------|--------------|
| 1. Environment (FIRST) | `conda activate <env_name>` | `which python` points to conda env, version 3.9-3.12 |
| 2. Dependencies | `conda install cmake=3.22.3 patch autoconf -y` + `pip install ... pybind11` | `which cmake` points to conda env |
| 3. Source | `git clone https://gitcode.com/mindspore/mindspore.git` | `build.sh` exists |
| 4. Build | `bash build.sh -e cpu -S on -j4` | Check `output/` directory |
| 5. Install | `pip install output/mindspore-*.whl` | `import mindspore` works |
| 6. Verify | `mindspore.run_check()` | Prints success message |
| 7. CI Check | `reference/ci-testing.md` | macOS CPU level0 CI-style cases can run and pass |

**Typical build time:** 30-60 minutes (first build)
**Disk space required:** 20GB minimum
**Troubleshooting:** See `reference/troubleshooting.md` for error patterns and use `reference/ci-testing.md` as the final post-upgrade validation

## Prerequisites

- **OS**: macOS (Apple Silicon)
- **Compiler**: Apple Clang
- **Disk Space**: At least 20GB

## Validation Steps After Upgrade

### Step 1: Set Up and Activate Conda Environment (REQUIRED FIRST)

**CRITICAL**: All subsequent steps MUST run within an activated conda environment with Python 3.9-3.12.

```dot
digraph env_setup {
    rankdir=TB;
    node [shape=box, style=rounded];

    start [label="Start", shape=doublecircle];
    check_conda [label="Check: conda --version", shape=diamond];
    install_conda [label="User must install conda\n(Miniconda/Anaconda)"];
    list_envs [label="List: conda env list"];
    ask_user [label="Ask user:\nNew or existing env?", shape=diamond];
    ask_python [label="Ask: Python version?\n(3.9/3.10/3.11/3.12)", shape=diamond];
    create_new [label="Create new environment\nconda create -n <name> python=<ver>"];
    check_existing [label="Check existing env\nPython version", shape=diamond];
    activate_new [label="Activate new env\nconda activate <name>"];
    activate_existing [label="Activate existing env\nconda activate <name>"];
    verify [label="Verify:\nwhich python\npython --version"];
    proceed [label="Proceed to Step 2", shape=doublecircle];

    start -> check_conda;
    check_conda -> list_envs [label="installed"];
    check_conda -> install_conda [label="not found"];
    install_conda -> list_envs;
    list_envs -> ask_user;
    ask_user -> ask_python [label="new"];
    ask_user -> check_existing [label="existing"];
    ask_python -> create_new;
    create_new -> activate_new;
    check_existing -> activate_existing [label="3.9-3.12"];
    check_existing -> create_new [label="wrong version"];
    activate_new -> verify;
    activate_existing -> verify;
    verify -> proceed;
}
```

**First, check if conda is installed:**
```bash
conda --version
```

**Check existing environments:**
```bash
conda env list
```

**MUST ask the user to choose:**
1. Create a new conda environment (MUST ask which Python version: 3.9, 3.10, 3.11, or 3.12)
2. Use an existing conda environment (MUST ask for environment name)

**For new environment:**
```bash
# Create environment with user-specified Python version
# Example: conda create -n mindspore_py310 python=3.10 -y
conda create -n <env_name> python=<version> -y
conda activate <env_name>

# Verify activation and Python version
python --version
which python
```

**For existing environment:**
```bash
# Activate existing environment
conda activate <existing_env_name>

# Verify Python version is supported (3.9-3.12)
python --version
which python
```

**STOP HERE if environment is not activated.** All following commands assume you are in the activated conda environment.

### Step 2: Check Dependencies (Within Activated Environment)

**PREREQUISITE**: Conda environment must be activated (Step 1).

#### System Tools

**Xcode Command Line Tools** (Required)
```bash
xcode-select -p
# If not installed, prompt user to run:
# xcode-select --install
```

**Install build tools via conda** (within activated environment)
```bash
# These install into the active conda environment
conda install cmake=3.22.3 patch autoconf scipy -y

# Verify cmake is from conda environment
which cmake
cmake --version
```

#### Python Packages

```bash
# Install within activated conda environment
pip install wheel==0.46.3 PyYAML==6.0.2 numpy==1.26.4 pybind11 -i https://repo.huaweicloud.com/repository/pypi/simple/

# Verify packages are installed in conda environment
pip list | grep -E "wheel|PyYAML|numpy|pybind11"
```

### Step 3: Prepare Source Code

Navigate to or clone the MindSpore source directory.

```bash
# Check if in MindSpore source directory (check for build.sh)
if [ -f "build.sh" ]; then
    echo "Already in MindSpore source directory"
elif [ -d "mindspore" ] && [ -f "mindspore/build.sh" ]; then
    echo "Found MindSpore in ./mindspore"
    cd mindspore
else
    echo "Cloning MindSpore source code..."
    git clone -b master https://gitcode.com/mindspore/mindspore.git ./mindspore
    cd mindspore
fi
```

### Step 4: Compile MindSpore (Within Activated Environment)

**PREREQUISITE**: Conda environment must be activated with all dependencies installed.

Set environment variables:

```bash
# Use .mslib in current directory for cache
export MSLIBS_CACHE_PATH=$(pwd)/.mslib
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++

# Verify environment
echo "Python: $(which python)"
echo "CMake: $(which cmake)"
echo "CC: $CC"
echo "CXX: $CXX"
```

Execute compilation:

```bash
# Ensure in MindSpore source directory and conda environment is active
bash build.sh -e cpu -S on -j4
```

**Parameters**:
- `MSLIBS_CACHE_PATH`: Cache path for third-party libraries
- `-e cpu`: CPU-only build
- `-S on`: Enable symbol table
- `-j4`: Use 4 threads (adjust based on CPU cores)

### Step 5: Install MindSpore

```bash
# Re-install wheel package
pip uninstall mindspore -y
pip install output/mindspore-*.whl -i https://repo.huaweicloud.com/repository/pypi/simple/
```

### Step 6: Verify Installation

**Basic check:**
```bash
python -c "import mindspore;print(mindspore.__version__)"
python -c "import mindspore;mindspore.set_device(device_target='CPU');mindspore.run_check()"
```

**Expected:**
```
MindSpore version: [version number]
The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!
```

### Step 7: Run macOS CPU CI Validation

After the wheel import smoke test passes, run the CI-style checks in `reference/ci-testing.md` to confirm the upgraded macOS environment still supports the expected Apple Silicon CPU test set.

## Common Mistakes

**Most critical mistakes to avoid:**

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Conda environment not activated | Dependencies install to system Python | Verify: `which python` points to conda env |
| Missing Xcode tools | `clang: command not found` | Run `xcode-select --install` |
| Stale CMake cache | Error persists after installing packages | Delete `build/mindspore/CMakeCache.txt` and rebuild |

**For complete mistake reference:** See `reference/common-mistakes.md`

**When build fails:**
1. Check `reference/troubleshooting.md` for matching error pattern
2. Verify all environment variables are set
3. Check disk space: `df -h .`

## User Interaction Guidelines

- Explain each major step before execution
- Display version info and verification results after completion
- **When compilation fails**: First consult `reference/troubleshooting.md` for matching error patterns before suggesting generic fixes
- Provide error log location and context-specific solutions based on troubleshooting history
