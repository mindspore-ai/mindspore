# Common Mistakes in MindSpore macOS Compilation

This reference lists common mistakes encountered during MindSpore compilation on macOS Apple Silicon, along with symptoms and fixes.

## Mistake Reference Table

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Wrong Python version | Import errors, ABI mismatch | Use Python 3.9-3.12 |
| Missing Xcode tools | `clang: command not found` | Run `xcode-select --install` |
| Insufficient disk space | Build fails mid-compilation | Free up 20GB+ before starting |
| Skipping environment variables | Linker errors, missing symbols | Set CC, CXX, LIBRARY_PATH, LDFLAGS |
| Not in source directory | `build.sh: No such file` | Verify `pwd` shows mindspore/ |
| Reusing old cache with new source | Cryptic build failures | Clear `.mslib/` and rebuild |
| Installing without uninstalling old version | Version conflicts | `pip uninstall mindspore -y` first |
| Missing pybind11 | `'pybind11/pybind11.h' file not found` | Install: `pip install pybind11`, then clean CMake cache and rebuild |
| Stale CMake cache | Same error persists after installing missing package | Delete `build/mindspore/CMakeCache.txt` and `build/mindspore/CMakeFiles/`, then rebuild |
| Missing PyTorch (tests) | Many tests skipped/not collected | `pip install torch torchvision torchaudio` |
| OpenMP conflict (tests) | `OMP: Error #15: Initializing libiomp5.dylib` | `export KMP_DUPLICATE_LIB_OK=TRUE` |

## General Debugging Steps

When build fails:

1. Check `troubleshooting.md` for matching error pattern
2. Verify all environment variables are set
3. Confirm you're in the correct directory
4. Check disk space: `df -h .`

## Critical Mistakes (Most Common)

### 1. Conda Environment Not Activated

**Symptom:** Dependencies install to system Python instead of conda environment

**Fix:**
```bash
# Verify conda environment is active
which python  # Should point to conda env
python --version  # Should be 3.9-3.12
```

### 2. Missing Xcode Command Line Tools

**Symptom:** `clang: command not found` or compiler errors

**Fix:**
```bash
xcode-select -p  # Check if installed
xcode-select --install  # Install if missing
```

### 3. Stale CMake Cache

**Symptom:** Same error persists after installing missing dependencies

**Fix:**
```bash
# Clean CMake cache
rm -rf build/mindspore/CMakeCache.txt build/mindspore/CMakeFiles/
# Rebuild
bash build.sh -e cpu -S on -j8
```
