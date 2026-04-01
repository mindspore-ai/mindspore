# Common Mistakes in MindSpore Linux CPU Compilation

This reference lists common mistakes encountered during MindSpore compilation on Linux x86_64 CPU, along with symptoms and fixes.

## Mistake Reference Table

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Wrong Python version | Import errors, ABI mismatch | Use Python 3.9-3.12 |
| Old compiler version | C++17 errors, unsupported flags | Install GCC 7.3.0-9.4.0 |
| Missing system packages | `fatal error: X.h not found` | Install system dependencies: `sudo apt-get install gcc-7 git tcl patch libnuma-dev -y` |
| Missing LLVM | Graph fusion errors, optimization warnings | Install LLVM-12: `sudo apt-get install llvm-12-dev -y` |
| Insufficient disk space | Build fails mid-compilation | Free up 20GB+ before starting |
| Skipping environment variables | Linker errors, missing symbols | Set MSLIBS_CACHE_PATH |
| Not in source directory | `build.sh: No such file` | Verify `pwd` shows mindspore/ |
| Reusing old cache with new source | Cryptic build failures | Clear `.mslib/` and rebuild |
| Installing without uninstalling old version | Version conflicts | `pip uninstall mindspore -y` first |
| Missing pybind11 | `'pybind11/pybind11.h' file not found` | Install: `pip install pybind11`, then clean CMake cache and rebuild |
| Stale CMake cache | Same error persists after installing missing package | Delete `build/mindspore/CMakeCache.txt` and `build/mindspore/CMakeFiles/`, then rebuild |
| Missing PyTorch (tests) | Many tests skipped/not collected | `pip install torch torchvision torchaudio` |

## General Debugging Steps

When build fails:

1. Check `troubleshooting.md` for matching error pattern
2. Verify all environment variables are set
3. Confirm you're in the correct directory
4. Check disk space: `df -h .`
5. Verify compiler version meets requirements (GCC 7.3.0+)
6. Verify LLVM is installed (llvm-config --version)

## Critical Mistakes (Most Common)

### 1. Conda Environment Not Activated

**Symptom:** Dependencies install to system Python instead of conda environment

**Fix:**
```bash
# Verify conda environment is active
which python  # Should point to conda env
python --version  # Should be 3.9-3.12
```

### 2. Old GCC Version

**Symptom:** C++17 compilation errors, unsupported compiler flags

**Fix:**
```bash
# Check GCC version
gcc --version  # Should be 7.3.0 or higher

# Install newer GCC if needed
sudo apt install -y gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
```

### 3. Stale CMake Cache

**Symptom:** Same error persists after installing missing dependencies

**Fix:**
```bash
# Clean CMake cache
rm -rf build/mindspore/CMakeCache.txt build/mindspore/CMakeFiles/
# Rebuild
bash build.sh -e cpu -j8 -S on
```
