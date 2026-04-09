# MindSpore Linux CPU Compilation Error Reference

Historical compilation errors and solutions for Linux x86_64 with GCC/Clang.

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
bash build.sh -e cpu -S on -j8
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

**Why this works**: Gitee mirrors are often more accessible in regions where GitHub is restricted or slow.

### Priority 3: Configure Network Proxy
If retries and mirrors fail, check network proxy availability:

```bash
# Set proxy environment variables before compilation:
export http_proxy="http://proxy-server:port"
export https_proxy="http://proxy-server:port"
export HTTP_PROXY="http://proxy-server:port"
export HTTPS_PROXY="http://proxy-server:port"

# Then run build:
bash build.sh -e cpu -S on -j8
```

**Warning to user**: "Network connectivity issue detected. GitHub is not accessible. Do you have a proxy server available? Please configure proxy settings or check your network connection."

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

---

## Error 1: GCC Version Too Old

**Error Message**:
```
error: #error This file requires compiler and library support for the ISO C++ 2017 standard
CMake Error: C++17 support is required
```

**Root Cause**: MindSpore requires C++17 support, which needs GCC 7.3+ or Clang 9.0+. Older compilers don't support C++17 features.

**Solution**: Install newer GCC version:

**For Ubuntu/Debian:**
```bash
sudo apt install -y gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90

# Verify installation
gcc --version
g++ --version
```

**For CentOS/RHEL:**
```bash
sudo yum install -y centos-release-scl
sudo yum install -y devtoolset-9-gcc devtoolset-9-gcc-c++
scl enable devtoolset-9 bash

# Verify installation
gcc --version
g++ --version
```

**Why this works**: GCC 9 fully supports C++17 standard required by MindSpore.

---

## Error 2: Missing System Headers

**Error Message**:
```
fatal error: openssl/ssl.h: No such file or directory
fatal error: zlib.h: No such file or directory
fatal error: ffi.h: No such file or directory
```

**Root Cause**: System development headers not installed. These are required for building third-party dependencies.

**Solution**: Install development packages:

**For Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y libssl-dev libffi-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev \
    libncurses5-dev libncursesw5-dev xz-utils \
    tk-dev libxml2-dev libxmlsec1-dev liblzma-dev
```

**For CentOS/RHEL:**
```bash
sudo yum install -y openssl-devel libffi-devel zlib-devel \
    bzip2-devel readline-devel sqlite-devel \
    ncurses-devel xz-devel tk-devel \
    libxml2-devel xmlsec1-devel liblzma-devel
```

**Why this works**: Development packages provide header files and libraries needed for compilation.

---

## Error 3: CMake Version Too Old

**Error Message**:
```
CMake 3.18 or higher is required. You are running version 3.10.2
```

**Root Cause**: System CMake version is too old. MindSpore requires CMake 3.18+.

**Solution**: Install newer CMake via conda:

```bash
conda install cmake=3.22.3 -y

# Verify installation
cmake --version
```

**Why this works**: Conda provides newer CMake versions than system package managers.

---

## Error 4: pybind11 Header Not Found

**Error Message**:
```
mindspore/ops/infer/ops_func_impl/py_func.cc:19:10: fatal error: 'pybind11/pybind11.h' file not found
   19 | #include <pybind11/pybind11.h>
      |          ^~~~~~~~~~~~~~~~~~~~~
```

**Root Cause**: pybind11 is required for compiling MindSpore's Python bindings layer but not installed.

**Solution**: Install pybind11 before compilation:

```bash
pip install pybind11 -i https://repo.huaweicloud.com/repository/pypi/simple/
```

**Important**: If pybind11 is installed **after** CMake has already been configured, you must clean the CMake cache:

```bash
rm -rf build/mindspore/CMakeCache.txt build/mindspore/CMakeFiles
bash build.sh -e cpu -S on -j8
```

**Why this works**: pybind11 provides C++ headers for Python/C++ interoperability. Cleaning CMake cache ensures new include paths are detected.

---

## Error 5: wheel Packaging Failure

**Error Message**:
```
TypeError: WheelFile.__init__() takes from 2 to 3 positional arguments but 4 were given
```

**Root Cause**: Using outdated `wheel==0.32.0` which has incompatible API with current setuptools version.

**Solution**: Upgrade wheel package before compilation:

```bash
pip uninstall wheel -y
pip install wheel>=0.46.3 -i https://repo.huaweicloud.com/repository/pypi/simple/
```

Then rebuild:
```bash
bash build.sh -e cpu -S on -j8
```

**Why this works**: Newer wheel versions (0.46.3+) have compatible API with current setuptools.

---

## Error 6: Linker Cannot Find Library

**Error Message**:
```
/usr/bin/ld: cannot find -latomic
/usr/bin/ld: cannot find -lstdc++
```

**Root Cause**: Required libraries not in linker search path or not installed.

**Solution**: Install missing libraries and set library path:

**For Ubuntu/Debian:**
```bash
sudo apt install -y libatomic1 libstdc++-9-dev
```

**For CentOS/RHEL:**
```bash
sudo yum install -y libatomic libstdc++-devel
```

Set library path:
```bash
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

**Why this works**: Ensures linker can find required libraries during build and runtime.

---

## Error 7: Out of Memory During Compilation

**Error Message**:
```
c++: fatal error: Killed signal terminated program cc1plus
virtual memory exhausted: Cannot allocate memory
```

**Root Cause**: Insufficient RAM for parallel compilation. Each compilation thread consumes significant memory.

**Solution**: Reduce parallel jobs:

```bash
# Instead of -j8, use fewer threads:
bash build.sh -e cpu -S on -j2

# Or use -j1 for single-threaded build:
bash build.sh -e cpu -S on -j1
```

**Alternative**: Add swap space:

```bash
# Create 8GB swap file
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verify swap is active
free -h
```

**Why this works**: Reducing parallelism decreases memory usage. Swap provides additional virtual memory.

---

## Error 8: Permission Denied When Installing System Packages

**Error Message**:
```
E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)
```

**Root Cause**: Installing system packages requires root privileges.

**Solution**: Use sudo:

```bash
sudo apt install -y <package-name>
```

**If sudo not available**: Ask system administrator to install required packages or use conda alternatives where possible.

---

## Error 9: Git Clone Fails with SSL Certificate Error

**Error Message**:
```
fatal: unable to access 'https://gitcode.com/mindspore/mindspore.git/':
SSL certificate problem: unable to get local issuer certificate
```

**Root Cause**: System CA certificates outdated or missing.

**Solution**: Update CA certificates:

**For Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y ca-certificates
sudo update-ca-certificates
```

**For CentOS/RHEL:**
```bash
sudo yum install -y ca-certificates
sudo update-ca-trust
```

**Temporary workaround** (not recommended for production):
```bash
git config --global http.sslVerify false
```

**Why this works**: Updated CA certificates allow Git to verify SSL connections.

---

## Error 10: Disk Space Exhausted

**Error Message**:
```
No space left on device
write error: No space left on device
```

**Root Cause**: Insufficient disk space for build artifacts and dependencies.

**Solution**: Free up disk space:

```bash
# Check disk usage
df -h .

# Clean package manager cache
sudo apt clean  # Ubuntu/Debian
sudo yum clean all  # CentOS/RHEL

# Remove old build artifacts
rm -rf build/ output/ .mslib/

# Check for large files
du -sh * | sort -h
```

**Minimum required**: 20GB free space before starting compilation.

---

## Summary of Required Changes

### Environment Variables (required before compilation):
```bash
export MSLIBS_CACHE_PATH=$(pwd)/.mslib
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
# Or for Clang:
# export CC=/usr/bin/clang
# export CXX=/usr/bin/clang++
```

### Python Dependencies:
```bash
pip install wheel>=0.46.3 PyYAML==6.0.2 numpy==1.26.4 pybind11
```

### System Dependencies (Ubuntu/Debian):
```bash
sudo apt install -y build-essential cmake git wget \
    libssl-dev libffi-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev \
    autoconf libtool pkg-config gcc-9 g++-9
```

---

## Quick Diagnosis Guide

**Download fails or network timeout** -> Error 0 (retry, use Gitee mirrors, configure proxy)

**C++17 errors or compiler too old** -> Error 1 (install GCC 9+ or Clang 9+)

**Missing header files** (`fatal error: X.h not found`) -> Error 2 (install development packages)

**CMake version error** -> Error 3 (install CMake 3.22.3 via conda)

**pybind11 header not found** -> Error 4 (install pybind11, clean CMake cache)

**Python packaging fails** -> Error 5 (upgrade wheel package)

**Linker cannot find library** -> Error 6 (install libraries, set LIBRARY_PATH)

**Out of memory / Killed** -> Error 7 (reduce parallel jobs, add swap)

**Permission denied** -> Error 8 (use sudo for system packages)

**SSL certificate error** -> Error 9 (update CA certificates)

**No space left on device** -> Error 10 (free up disk space, need 20GB+)
