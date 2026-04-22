# MindSpore macOS Compilation Error Reference

Historical compilation errors and solutions for macOS Apple Silicon with Clang 17.0+.

---

## Error 0: Network Timeout When Downloading Dependencies

**Error Message**:
```
Failed to download [package_name]
Connection timeout
curl: (28) Operation timed out
fatal: unable to access 'https://github.com/...': Failed to connect
```

**Root Cause**: MindSpore build system downloads third-party dependencies from GitHub during compilation. GitHub may be inaccessible or slow due to network restrictions, firewall policies, or regional connectivity issues.

**Solution** (try in priority order):

### Priority 1: Retry with Automatic Retry Logic
```bash
# The build system will automatically retry failed downloads up to 3 times
# Simply re-run the build command:
bash build.sh -e cpu -S on -j4
```

**Why this works**: Transient network issues often resolve themselves. Automatic retries handle temporary connection failures without manual intervention.

### Priority 2: Use Gitee Mirror Sources
Check if the dependency has a Gitee (gitee.com) mirror and configure the build to use it:

```bash
# Check cmake/external_libs/*.cmake files for the failing dependency
# Look for URL definitions like:
# set(PACKAGE_URL "https://github.com/...")

# Modify to use Gitee mirror if available:
# set(PACKAGE_URL "https://gitee.com/mirrors/...")
```

Common Gitee mirrors for MindSpore dependencies:
- `https://gitee.com/mirrors/flatbuffers`
- `https://gitee.com/mirrors/protobuf`
- `https://gitee.com/mirrors/grpc`

**Why this works**: Gitee mirrors are often more accessible in regions where GitHub is restricted or slow. Many popular open-source projects maintain official Gitee mirrors.

### Priority 3: Configure Network Proxy
If retries and mirrors fail, check network proxy availability:

```bash
# Set proxy environment variables before compilation:
export http_proxy="http://proxy-server:port"
export https_proxy="http://proxy-server:port"
export HTTP_PROXY="http://proxy-server:port"
export HTTPS_PROXY="http://proxy-server:port"

# Then run build:
bash build.sh -e cpu -S on -j4
```

**Warning to user**: "Network connectivity issue detected. GitHub is not accessible. Do you have a proxy server available? Please configure proxy settings or check your network connection."

**Why this works**: Corporate or institutional networks often require proxy configuration. Setting proxy environment variables allows curl/git to route through the proxy.

### Priority 4: Manual Download and Local Installation (Last Resort)
If all above methods fail, manually download dependencies:

```bash
# 1. Identify the failing package and URL from error message
# 2. Download manually using browser or alternative tool
# 3. Place downloaded file in MSLIBS_CACHE_PATH:
mkdir -p $(pwd)/.mslib
cp /path/to/downloaded/package.tar.gz $(pwd)/.mslib/

# 4. Modify corresponding cmake/external_libs/*.cmake to use local file:
# set(PACKAGE_URL "file://$(pwd)/.mslib/package.tar.gz")
```

**Critical Warning**: "Manual download may cause hash mismatch errors if file integrity is compromised. The build system verifies package checksums for security. If you encounter hash mismatch errors, the downloaded file may be corrupted or tampered with. **Strongly recommend fixing network issues first** rather than bypassing security checks."

**Why this approach is risky**:
- Downloaded files may not match expected checksums (MD5/SHA256)
- Build will fail with "hash mismatch" error if checksums don't match
- Bypassing checksum verification compromises build security
- Network issues are the root cause and should be addressed properly

---

## Error 1: SDK Version Not Supported

**Error Message**:
```
CMake Error: Could not find appropriate macOS SDK
```

**Root Cause**: The regex in `cmake/check_requirements.cmake` only matches `MacOSX11.x` format, but newer macOS versions use `MacOSX26.x` or higher.

**Solution**: Modify `cmake/check_requirements.cmake` line 89:
```cmake
# Change from:
set(MACOSX_SDK_REGEX "MacOSX11(\\.\\d+)?")

# To:
set(MACOSX_SDK_REGEX "MacOSX(1[1-9]|[2-9][0-9])(\\.\\d+)?")
```

**Why this works**: New regex matches macOS 11-99, supporting current and future SDK versions.

---

## Error 2: flatbuffers Unused Variable Warning

**Error Message**:
```
/path/to/flatbuffers-src/src/idl_gen_rust.cpp:499:12: error: variable 'i' set but not used [-Werror,-Wunused-but-set-variable]
  499 |     size_t i = 0;
      |            ^
1 error generated.
```

**Root Cause**: Apple Clang 17.0 introduced stricter warnings. flatbuffers v2.0.0 third-party library triggers this warning in `idl_gen_rust.cpp`, and since flatbuffers compiles with `-Werror` hardcoded in its own CMakeLists.txt (line 225), warnings become errors.

**Important**: The `cmake/external_libs/flatbuffers.cmake` already contains `-Wno-unused-but-set-variable` in `flatbuffers_CXXFLAGS` and `-DFLATBUFFERS_STRICT_MODE=OFF` in CMAKE_OPTION, but these **do not work** because:
1. `flatbuffers_CXXFLAGS` is only used for the library build, not the `flatc` compiler tool
2. `FLATBUFFERS_STRICT_MODE` is not recognized by flatbuffers v2.0.0 (CMake warns "Manually-specified variables were not used")

**Solution**: The downloaded flatbuffers source must be patched after extraction. Two approaches:

### Approach 1: Manual Patch (Quick Fix)
After the first build failure, manually patch the downloaded source:

```bash
# Edit the downloaded flatbuffers CMakeLists.txt
vim build/mindspore/_deps/flatbuffers-src/CMakeLists.txt

# Find line 225 (in the APPLE section):
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Werror -Wextra -Wno-unused-parameter")

# Change to:
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Werror -Wextra -Wno-unused-parameter -Wno-unused-but-set-variable")

# Then rebuild:
bash build.sh -e cpu -S on -j4
```

### Approach 2: Permanent Patch (Recommended)
Create a patch file and modify `cmake/external_libs/flatbuffers.cmake` to apply it automatically:

1. Create patch file `cmake/external_libs/flatbuffers_clang17.patch`:
```diff
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -222,7 +222,7 @@ elseif(CMAKE_TOOLCHAIN_FILE)
   message(STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}.")
 elseif(APPLE)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++")
-  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Werror -Wextra -Wno-unused-parameter")
+  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Werror -Wextra -Wno-unused-parameter -Wno-unused-but-set-variable")
   set(FLATBUFFERS_PRIVATE_CXX_FLAGS "-Wold-style-cast")
 elseif(CMAKE_COMPILER_IS_GNUCXX)
   if(CYGWIN)
```

2. Modify `cmake/external_libs/flatbuffers.cmake` line 60-66 to add PATCHES parameter:
```cmake
if(APPLE)
    mindspore_add_pkg(flatbuffers
            VER 2.0.0
            LIBS flatbuffers
            EXE flatc
            URL ${REQ_URL}
            SHA256 ${SHA256}
            PATCHES ${CMAKE_CURRENT_LIST_DIR}/flatbuffers_clang17.patch
            CMAKE_OPTION -DFLATBUFFERS_BUILD_TESTS=OFF -DCMAKE_INSTALL_LIBDIR=lib)
```

**Why this works**: Directly patching flatbuffers' CMakeLists.txt adds the warning suppression flag where it's actually used (for both library and flatc tool compilation). The patch is applied after download but before build.

**Verification**: After applying the fix, check the build log for:
```
-- CMAKE_CXX_FLAGS: -fPIC -fPIE ... -Wno-unused-but-set-variable -std=c++11 -stdlib=libc++ -Wall -pedantic -Werror -Wextra -Wno-unused-parameter -Wno-unused-but-set-variable
```

---

## Error 3: sentencepiece Cannot Find libatomic

**Error Message**:
```
ld: library 'atomic' not found
```

**Root Cause**: sentencepiece depends on libatomic library. The library exists in conda environment (`$CONDA_PREFIX/lib/libatomic.dylib`) but linker cannot find it because conda lib path is not in default search paths.

**Solution**: Set environment variable before compilation:
```bash
export LIBRARY_PATH=$CONDA_PREFIX/lib
```

**Why this works**: `LIBRARY_PATH` tells the linker where to search for libraries during link time. This adds the conda lib directory to the search path.

---

## Error 4: jemalloc Configure Failure

**Error Message**:
```
Unsupported pointer size: 0
```

**Root Cause**: jemalloc's configure script runs a test program to detect pointer size. The test program links against libc++.1.dylib but cannot find it at runtime because the system library path `/usr/lib` is not in the runtime search path (rpath).

**Solution**: Set environment variable before compilation:
```bash
export LDFLAGS="-Wl,-rpath,/usr/lib -Wl,-rpath,$CONDA_PREFIX/lib"
```

**Why this works**: `-Wl,-rpath,/usr/lib` adds system library path to runtime search path. `-Wl,-rpath,$CONDA_PREFIX/lib` adds conda library path. This allows configure scripts and compiled binaries to find required dynamic libraries at runtime.

---

## Error 5: robin_hood_hashing Deprecated Builtin

**Error Message**:
```
error: builtin __has_trivial_copy is deprecated; use __is_trivially_copyable instead [-Werror,-Wdeprecated-builtins]
```

**Root Cause**: Apple Clang 17.0 deprecated `__has_trivial_copy` builtin in favor of `__is_trivially_copyable`. The third-party library robin_hood_hashing uses the old builtin, triggering this error when compiled with `-Werror`.

**Solution**: Modify `CMakeLists.txt` line 69, add to CMAKE_CXX_FLAGS:
```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I/usr/local/include -std=c++17 \
    -Werror -Wall -Wno-deprecated-declarations -Wno-deprecated-builtins -fPIC")
```

**Why this works**: `-Wno-deprecated-builtins` suppresses warnings about deprecated compiler builtins, allowing third-party code to compile without modification.

---

## Error 6: Variable Length Array Extension

**Error Message**:
```
error: variable length arrays in C++ are a Clang extension [-Werror,-Wvla-cxx-extension]
```

**Root Cause**: Code uses Variable Length Arrays (VLA), a C99 feature not part of C++ standard. Apple Clang 17.0 strictly enforces C++ standard compliance and treats VLA usage as an error when `-Werror` is enabled.

**Solution**: Modify `CMakeLists.txt` line 69, add to CMAKE_CXX_FLAGS:
```cmake
-Wno-vla-cxx-extension
```

**Why this works**: Suppresses the VLA extension warning, allowing code that uses VLAs (like crypto.cc) to compile. This is acceptable for platform-specific code where VLA support is guaranteed.

---

## Error 7: grpc Cannot Find libz.1.dylib

**Error Message**:
```
dyld: Library not loaded: @rpath/libz.1.dylib
```

**Root Cause**: grpc depends on zlib (libz.1.dylib). The library exists in conda environment but the runtime linker (dyld) cannot find it because `@rpath` doesn't include the conda lib directory.

**Solution**: Set environment variable before compilation (same as Error 4):
```bash
export LDFLAGS="-Wl,-rpath,/usr/lib -Wl,-rpath,$CONDA_PREFIX/lib"
```

**Why this works**: Adding conda lib path to rpath ensures that when grpc (or any binary) is executed, dyld can find libz.1.dylib at runtime.

---

## Error 8: BFloat16 Power Template Instantiation Failure

**Error Message**:
```
error: no matching function for call to '__test'
note: in instantiation of template class 'std::promote<BFloat16, BFloat16>' requested here
```

**Root Cause**: BFloat16 is a custom type that doesn't satisfy `std::promote` type trait requirements (not an arithmetic type). When `Power<BFloat16>` template is instantiated, the compiler cannot deduce the return type because BFloat16 doesn't work with standard type promotion rules.

**Solution**: Add explicit template specialization to `mindspore/ccsrc/utils/np_dtypes.cc` after line 408:
```cpp
template <>
struct Power<BFloat16> {
  BFloat16 operator()(BFloat16 a, BFloat16 b) {
    return BFloat16(std::pow(static_cast<float>(a), static_cast<float>(b)));
  }
};
```

**Why this works**: Provides explicit implementation for BFloat16 power operation by converting to float, performing the operation, then converting back. This bypasses the type trait requirements.

---

## Error 9: Unqualified std::move Call

**Error Message**:
```
error: unqualified call to 'std::move' [-Werror,-Wunqualified-std-cast-call]
```

**Root Cause**: Apple Clang 17.0 requires explicit `std::` namespace prefix when calling standard library functions. Code contains calls like `move(x)` instead of `std::move(x)`.

**Solution**: Modify `CMakeLists.txt` line 69, add to CMAKE_CXX_FLAGS:
```cmake
-Wno-unqualified-std-cast-call
```

**Why this works**: Suppresses the warning about unqualified std calls. Alternative would be to fix all call sites, but suppressing is acceptable for third-party code.

---

## Error 10: Linker Does Not Support -noall_load

**Error Message**:
```
ld: unknown options: -noall_load
```

**Root Cause**: Modern macOS linker removed support for `-noall_load` option. This option was used to cancel the effect of `-all_load` (which forces loading of all symbols from static libraries). The old pattern was: `-Wl,-all_load lib1 -Wl,-noall_load lib2 lib3`.

**Solution**: Modify `mindspore/ccsrc/CMakeLists.txt` lines 100-106:
```cmake
# Change from single call with -noall_load:
target_link_libraries(_c_expression PRIVATE
    -Wl,-all_load mindspore_ops_fallback proto_input
    -Wl,-noall_load mindspore_core mindspore_ops ...)

# To two separate calls:
target_link_libraries(_c_expression PRIVATE
    -Wl,-all_load mindspore_ops_fallback proto_input)
target_link_libraries(_c_expression PRIVATE
    mindspore_core mindspore_ops ...)
```

**Why this works**: Splitting into two calls achieves the same effect: first call applies `-all_load` only to specified libraries, second call links remaining libraries normally without `-all_load`.

---

## Error 11: wheel Packaging Failure

**Error Message**:
```
TypeError: WheelFile.__init__() takes from 2 to 3 positional arguments but 4 were given
```

**Root Cause**: Using outdated `wheel==0.32.0` which has incompatible API with current setuptools version. The WheelFile class signature changed in newer versions. This error occurs during CPack packaging phase when calling `setup.py bdist_wheel`.

**Solution**: Upgrade wheel package before compilation:
```bash
pip uninstall wheel -y
pip install wheel>=0.46.3 -i https://repo.huaweicloud.com/repository/pypi/simple/
```

Then rebuild:
```bash
bash build.sh -e cpu -S on -j4
```

**Why this works**: Newer wheel versions (0.46.3+) have compatible API with current setuptools. Note: C++ compilation completes successfully; only Python packaging fails, so you must rebuild after upgrading wheel.

---

## Error 12: Missing pybind11 Header

**Error Message**:
```
mindspore/ops/infer/ops_func_impl/py_func.cc:19:10: fatal error: 'pybind11/pybind11.h' file not found
   19 | #include <pybind11/pybind11.h>
      |          ^~~~~~~~~~~~~~~~~~~~~
1 error generated.
```

**Root Cause**: pybind11 is required for compiling MindSpore's Python bindings layer (py_func.cc, etc.), but it's not listed in the official build prerequisites. The header file `pybind11/pybind11.h` must be available in the system or conda include paths.

**Solution**: Install pybind11 before compilation:
```bash
pip install pybind11 -i https://repo.huaweicloud.com/repository/pypi/simple/
```

**Important**: If pybind11 is installed **after** CMake has already been configured, you must clean the CMake cache for the new include paths to be detected:
```bash
rm -rf build/mindspore/CMakeCache.txt build/mindspore/CMakeFiles
bash build.sh -e cpu -S on -j4
```

Simply re-running `build.sh` without clearing the cache will NOT pick up the new pybind11 installation, because CMake reuses the cached configuration.

**Why this works**: pybind11 provides C++ headers that enable seamless Python/C++ interoperability. Installing via pip places the headers in a path that CMake's `FindPython` or pybind11 config module can discover during the configure phase.

---

## Summary of Required Changes

### Source Code Files (5 files):
1. `cmake/check_requirements.cmake` - Update SDK version regex
2. `CMakeLists.txt` - Add 3 warning suppressions: `-Wno-deprecated-builtins -Wno-vla-cxx-extension -Wno-unqualified-std-cast-call`
3. `cmake/external_libs/flatbuffers.cmake` - Disable strict mode, add warning flags
4. `mindspore/ccsrc/CMakeLists.txt` - Remove `-noall_load`, split into two link calls
5. `mindspore/ccsrc/utils/np_dtypes.cc` - Add BFloat16 Power template specialization

### Environment Variables (required before compilation):
```bash
export LIBRARY_PATH=$CONDA_PREFIX/lib
export LDFLAGS="-Wl,-rpath,/usr/lib -Wl,-rpath,$CONDA_PREFIX/lib"
```

### Python Dependencies:
```bash
pip install wheel>=0.46.3 pybind11
```

---

## Quick Diagnosis Guide

**Download fails or network timeout** -> Error 0 (retry, use Gitee mirrors, configure proxy)

**CMake fails to find SDK** -> Error 1 (SDK version regex)

**Linker cannot find library** (`ld: library 'X' not found`) -> Errors 3, 7 (set LIBRARY_PATH)

**Runtime library not loaded** (`dyld: Library not loaded`) -> Errors 4, 7 (set LDFLAGS with rpath)

**Compiler warning treated as error** (`[-Werror,-W...]`) -> Errors 2, 5, 6, 9 (add warning suppression)

**Template instantiation fails** -> Error 8 (add template specialization)

**Linker option not recognized** -> Error 10 (update linker flags)

**Python packaging fails** -> Error 11 (upgrade wheel package)

**Missing header file** (`fatal error: 'X.h' file not found`) -> Error 12 (install pybind11, clean CMake cache)
