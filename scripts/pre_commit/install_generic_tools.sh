#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Remove strict error handling to prevent script from stopping on minor errors
# set -euo pipefail

# Color output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if running on Debian/Ubuntu
is_debian_based() {
    command -v apt-get >/dev/null 2>&1 && \
    ([[ -f /etc/debian_version ]] || [[ -f /etc/lsb-release ]])
}

# Check network connectivity to PyPI
check_network() {
    log_info "Checking network connectivity to PyPI..."
    if python -c "import urllib.request; urllib.request.urlopen('https://pypi.org', timeout=10)" 2>/dev/null; then
        log_success "Network connectivity is OK"
        return 0
    else
        log_warning "Cannot reach PyPI directly. This may be due to proxy settings or network restrictions."
        return 1
    fi
}

# Check Python installation and version
check_python() {
    local available_py_versions=("3.9" "3.10" "3.11")
    
    if ! command_exists python; then
        log_error "Python is not installed. Please install Python ${available_py_versions[*]}"
        return 1
    fi

    local python_version
    python_version=$(python -c "import sys; print('.'.join(str(x) for x in sys.version_info[:2]))" 2>/dev/null)
    
    if [[ -z "$python_version" ]]; then
        log_error "Failed to determine Python version"
        return 1
    fi

    local version_ok=false
    for version in "${available_py_versions[@]}"; do
        if [[ "$python_version" == "$version" ]]; then
            version_ok=true
            break
        fi
    done

    if ! $version_ok; then
        log_error "Python version '$python_version' is not supported. Available versions: [${available_py_versions[*]}]"
        return 1
    fi

    log_info "Found Python version: $python_version"
    return 0
}

# Check pip installation
check_pip() {
    if ! command_exists pip && ! python -c "import pip" 2>/dev/null; then
        log_error "pip is not installed. Please install pip first"
        return 1
    fi
    
    # Use Python to call pip to avoid potential path issues
    if ! python -m pip --version >/dev/null 2>&1; then
        log_error "pip is not properly installed or not in PATH"
        return 1
    fi
    
    log_info "Found pip: $(python -m pip --version 2>/dev/null | head -n1 || echo "pip available")"
    return 0
}

# Special handling for Debian/Ubuntu systems
debian_safe_install() {
    local package_name="$1"
    local package_spec="$2"
    
    if is_debian_based; then
        log_info "Debian/Ubuntu system detected, using --break-system-packages to avoid conflicts with system packages"
        if python -m pip install --break-system-packages --upgrade --force-reinstall "$package_spec" 2>/dev/null; then
            return 0
        else
            log_warning "Failed with --break-system-packages, trying with --user install"
            if python -m pip install --user --upgrade --force-reinstall "$package_spec" 2>/dev/null; then
                return 0
            fi
        fi
    fi
    return 1
}

# Try multiple installation methods
install_with_retry() {
    local package_name="$1"
    local package_spec="$2"
    local max_retries=2
    local retry_count=0
    
    while [[ $retry_count -le $max_retries ]]; do
        case $retry_count in
            0)
                # Method 1: Direct installation
                log_info "Attempting to install $package_name directly..."
                if python -m pip install --upgrade --force-reinstall "$package_spec" 2>/dev/null; then
                    return 0
                fi
                ;;
            1)
                # Method 2: Debian-safe installation
                if is_debian_based; then
                    if debian_safe_install "$package_name" "$package_spec"; then
                        return 0
                    fi
                fi
                ;;
            2)
                # Method 3: Using mirror
                log_info "Trying with Tsinghua mirror..."
                if python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn --upgrade --force-reinstall "$package_spec" 2>/dev/null; then
                    return 0
                fi
                ;;
        esac
        
        ((retry_count++))
        if [[ $retry_count -le $max_retries ]]; then
            log_warning "Installation attempt $retry_count failed for $package_name, retrying..."
            sleep 2
        fi
    done
    
    return 1
}

# Install package with comprehensive error handling
install_package() {
    local package_name="$1"
    local package_spec="$2"
    
    log_info "Installing $package_name..."
    
    # Check if already installed
    if python -c "import ${package_name%%==*}" 2>/dev/null; then
        log_info "$package_name is already installed, reinstalling..."
    fi
    
    if install_with_retry "$package_name" "$package_spec"; then
        log_success "Successfully installed $package_name"
        return 0
    else
        log_error "All installation attempts failed for $package_name"
        
        # Provide specific advice for Debian systems
        if is_debian_based; then
            log_info "On Debian/Ubuntu systems, you can try:"
            log_info "1. Install using apt: sudo apt-get install python3-${package_name}"
            log_info "2. Use Python virtual environment: python -m venv venv && source venv/bin/activate"
            log_info "3. Use conda: conda install ${package_name}"
        else
            log_info "You may need to:"
            log_info "1. Check your network connection and proxy settings"
            log_info "2. Configure pip proxy: pip config set global.proxy http://proxy:port"
            log_info "3. Use Python virtual environment to avoid system package conflicts"
        fi
        return 1
    fi
}

# Check tool version
check_tool_version() {
    local tool_name="$1"
    local check_cmd="$2"
    
    log_info "Checking $tool_name version..."
    if eval "$check_cmd" 2>/dev/null; then
        log_success "$tool_name is working properly"
        return 0
    else
        # Try to find the command in Python user directory
        local user_bin_dir
        user_bin_dir="$(python -m site --user-base 2>/dev/null)/bin"
        if [[ -d "$user_bin_dir" && -x "$user_bin_dir/$tool_name" ]]; then
            log_info "Found $tool_name in user bin directory, you may need to add to PATH:"
            log_info "export PATH=\$PATH:$user_bin_dir"
            return 0
        else
            log_warning "$tool_name not installed or not in PATH!"
            return 1
        fi
    fi
}

# Provide manual installation instructions
show_manual_instructions() {
    log_info "=== Manual Installation Instructions ==="
    log_info "If automatic installation continues to fail, you can:"
    log_info ""
    
    if is_debian_based; then
        log_info "For Debian/Ubuntu systems:"
        log_info "1. Try system packages:"
        log_info "   sudo apt-get update"
        log_info "   sudo apt-get install python3-pylint python3-cpplint codespell"
        log_info ""
        log_info "2. Use Python virtual environment:"
        log_info "   python -m venv codecheck-env"
        log_info "   source codecheck-env/bin/activate"
        log_info "   pip install pylint cpplint codespell cmakelint lizard clang-format"
        log_info ""
    fi
    
    log_info "3. Configure pip proxy (if behind corporate firewall):"
    log_info "   pip config set global.proxy http://your-proxy:port"
    log_info ""
    log_info "4. Download wheels manually and install offline:"
    log_info "   pip download package_name -d ./packages/"
    log_info "   pip install --no-index --find-links ./packages/ package_name"
    log_info ""
}

# Main installation function
main() {
    log_info "Starting code check tools installation..."
    
    # Detect system type
    if is_debian_based; then
        log_info "Detected Debian/Ubuntu based system"
    fi
    
    # Pre-flight checks
    log_info "Checking system requirements..."
    if ! check_python; then
        exit 1
    fi
    if ! check_pip; then
        exit 1
    fi
    if ! check_network; then
        log_warning "Network issues detected, installation may fail"
    fi
    
    # Define tools to install (name package_spec check_command)
    declare -a tools=(
        "cmakelint|cmakelint|cmakelint --version"
        "codespell|codespell|codespell --version"
        "cpplint|cpplint|cpplint --version"
        "lizard|lizard|lizard --version"
        "pylint|pylint==3.3.7|pylint --version"
        "clang-format|clang-format==18.1.8|clang-format --version"
    )
    
    local failed_installations=0
    local successful_installations=0
    
    # Install tools
    for tool_info in "${tools[@]}"; do
        IFS='|' read -r tool_name package_spec check_cmd <<< "$tool_info"
        
        log_info "Processing tool: $tool_name"
        if install_package "$tool_name" "$package_spec"; then
            ((successful_installations++))
        else
            ((failed_installations++))
        fi
        echo "----------------------------------------"
    done
    
    # Verify installations
    log_info "Verifying installations..."
    local failed_checks=0
    
    for tool_info in "${tools[@]}"; do
        IFS='|' read -r tool_name package_spec check_cmd <<< "$tool_info"
        
        if ! check_tool_version "$tool_name" "$check_cmd"; then
            ((failed_checks++))
        fi
    done
    
    # Summary
    echo
    log_info "=== Installation Summary ==="
    if [[ $failed_installations -eq 0 && $failed_checks -eq 0 ]]; then
        log_success "All code check tools installed successfully!"
    else
        log_warning "Installation completed with some issues:"
        log_warning "- $successful_installations tool(s) installed successfully"
        log_warning "- $failed_installations installation(s) failed"
        log_warning "- $failed_checks tool(s) failed version check"
        
        # Show PATH instructions
        local user_bin_dir
        user_bin_dir="$(python -m site --user-base 2>/dev/null)/bin"
        if [[ -d "$user_bin_dir" ]]; then
            log_info "You may need to add Python user binaries to your PATH:"
            log_info "export PATH=\$PATH:$user_bin_dir"
            log_info "Add this to your ~/.bashrc or ~/.profile for permanent effect"
        fi
        
        if [[ $failed_installations -gt 0 ]]; then
            echo
            show_manual_instructions
        fi
    fi
}

# Run main function
main "$@"