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

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Get OS information
readonly OS_NAME=$(uname -s)
readonly ARCH=$(uname -m)

# Default shellcheck version
SHELLCHECK_VERSION="v0.7.1"

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

# Show help information
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Install codecheck tools for different platforms.

On Windows: Only install Ruby gems (chef-utils, mdl) - ShellCheck is skipped
On Linux/macOS: Install shellcheck and optionally Ruby gems

Options:
  -v, --version VERSION    Specify shellcheck version (default: $SHELLCHECK_VERSION) - Linux/macOS only
  -h, --help               Show this help message
  --install-dir DIR        Custom installation directory - Linux only
  --install-gems           Install Ruby gems (chef-utils, mdl)

Examples:
  $0                          # Install with default settings
  $0 --install-gems           # Install Ruby gems on all platforms
  $0 -v v0.9.0                # Install specific shellcheck version (Linux/macOS)
  $0 --install-dir /usr/local # Install to custom directory (Linux)

Platform-specific behavior:
  - Windows: Only installs Ruby gems (ShellCheck is intentionally skipped as it has no reference value on Windows)
  - Linux: Installs shellcheck and optionally Ruby gems
  - macOS: Installs shellcheck via Homebrew and optionally Ruby gems

EOF
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites for Linux installation
check_linux_prerequisites() {
    local missing_tools=()
    
    if ! command_exists xz && ! command_exists tar; then
        missing_tools+=("xz or tar")
    fi
    
    if ! command_exists curl && ! command_exists wget; then
        missing_tools+=("curl or wget")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install them using your package manager:"
        log_info "  Ubuntu/Debian: sudo apt-get install ${missing_tools[*]}"
        log_info "  CentOS/RHEL: sudo yum install ${missing_tools[*]}"
        exit 1
    fi
}

# Download file with curl or wget
download_file() {
    local url="$1"
    local output="$2"
    
    if command_exists wget; then
        wget -O "$output" "$url" "--no-check-certificate"
    elif command_exists curl; then
        curl --ssl-no-revoke -k -f -L -o "$output" "$url"
    else
        log_error "Neither curl nor wget is available"
        return 1
    fi
}

install_linux_codecheck_tool() {
    # Check system architecture
    case "$ARCH" in
        x86_64|aarch64)
            log_info "Supported architecture detected: $ARCH"
            ;;
        *)
            log_error "Unsupported architecture: $ARCH"
            log_info "Supported architectures: x86_64 aarch64"
            log_info "For other architectures, please install shellcheck manually from your package manager"
            exit 1
            ;;
    esac
    
    log_info "System architecture: $ARCH, starting installation..."
    
    # Check sudo privileges for regular users
    if [[ $EUID -ne 0 ]]; then
        log_info "Checking sudo privileges..."
        if ! command_exists sudo; then
            log_error "sudo command not found. Regular user installation requires sudo privileges."
            exit 1
        fi
    fi
    
    check_linux_prerequisites
    
    local package_shellcheck_name="shellcheck-${SHELLCHECK_VERSION}"
    local temp_dir="/tmp"
    
    log_info "Installing shellcheck version: $SHELLCHECK_VERSION"
    
    # Determine installation directory
    local install_dir="${INSTALL_DIR:-/usr/bin}"
    if [[ ! -w "$install_dir" && $EUID -ne 0 ]]; then
        log_info "Installation directory $install_dir requires root access"
        if ! sudo -v >/dev/null 2>&1; then
            log_error "Regular user installation requires sudo privileges. Please run with a user that has sudo access."
            exit 1
        fi
    fi
    
    # Unified URL building logic
    local filename
    local download_url
    
    case "$ARCH" in
        x86_64)
            filename="${package_shellcheck_name}.linux.x86_64.tar.xz"
            download_url="https://tools.mindspore.cn/tools/check/shellcheck/${package_shellcheck_name}/$filename"
            ;;
        aarch64)
            filename="${package_shellcheck_name}.linux.aarch64.tar.xz"
            download_url="https://tools.mindspore.cn/tools/check/shellcheck/${package_shellcheck_name}/$filename"
            ;;
        *)
            log_error "Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac

    # Download
    cd "$temp_dir" || {
        log_error "Cannot change to temp directory: $temp_dir"
        exit 1
    }
    
    log_info "Downloading shellcheck from: $download_url"
    if ! download_file "$download_url" "$filename"; then
        log_error "Failed to download shellcheck"
        log_info "Please check:"
        log_info "1. Network connectivity"
        log_info "2. Version $SHELLCHECK_VERSION availability"
        log_info "3. URL: $download_url"
        exit 1
    fi
    
    # Verify downloaded file
    if [[ ! -f "$filename" ]]; then
        log_error "Downloaded file not found: $filename"
        exit 1
    fi
    
    # Extract
    log_info "Extracting shellcheck package..."
    if command_exists tar; then
        if ! tar -xf "$filename"; then
            log_error "Failed to extract shellcheck package"
            exit 1
        fi
    else
        log_error "tar command not found, cannot extract package"
        exit 1
    fi
    
    # Find the shellcheck binary in the extracted directory
    local shellcheck_binary
    if [[ -f "${package_shellcheck_name}/shellcheck" ]]; then
        shellcheck_binary="${package_shellcheck_name}/shellcheck"
    elif [[ -f "shellcheck" ]]; then
        shellcheck_binary="shellcheck"
    else
        # Try to find it in any subdirectory
        shellcheck_binary=$(find "/tmp/${package_shellcheck_name}" -name "shellcheck" -type f 2>/dev/null | head -1)
        if [[ -z "$shellcheck_binary" ]]; then
            log_error "Could not find shellcheck binary in extracted files"
            exit 1
        fi
    fi
    
    # Install
    log_info "Installing shellcheck to $install_dir/shellcheck"
    if [[ $EUID -eq 0 ]]; then
        # Running as root
        mv "$shellcheck_binary" "$install_dir/shellcheck"
        chmod 755 "$install_dir/shellcheck"
    else
        # Running as regular user, use sudo
        sudo mv "$shellcheck_binary" "$install_dir/shellcheck"
        sudo chmod 755 "$install_dir/shellcheck"
    fi
    
    # Cleanup
    log_info "Cleaning up temporary files..."
    rm -rf "/tmp/${package_shellcheck_name}" 2>/dev/null || true
    rm -f "/tmp/${filename}" 2>/dev/null || true
    
    log_success "Shellcheck installed successfully to $install_dir/shellcheck"
}

install_windows_codecheck_tools() {
    log_info "Installing Windows codecheck tools..."
    log_warning "================================================================"
    log_warning "ShellCheck intentionally skipped on Windows"
    log_warning "Reason: ShellCheck has no reference value in Windows environment"
    log_warning "Windows shell environments differ significantly from Linux/macOS"
    log_warning "================================================================"
    
    # Install Ruby gems if available and requested
    if command_exists gem && [[ "$INSTALL_GEMS" == "true" ]]; then
        log_info "Installing Ruby gems..."
        gem sources --add https://gems.ruby-china.com/ --remove https://rubygems.org/
        gem install chef-utils -v 16.6.14
        gem install mdl
        log_success "Ruby gems installed successfully"
    else
        log_info "Skipping Ruby gems installation (gem not found or not requested)"
        log_info "To install Ruby gems, use the --install-gems flag"
    fi
    
    log_success "Windows codecheck tools installation completed"
    log_info "Note: ShellCheck was intentionally not installed as it has no reference value on Windows"
}

install_mac_codecheck_tools() {
    log_info "Installing macOS codecheck tools..."
    
    if ! command_exists brew; then
        log_error "Homebrew is required but not installed. Please install Homebrew first."
        log_info "You can install Homebrew from: https://brew.sh/"
        log_info "Install command:"
        log_info '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        exit 1
    fi
    
    log_info "Installing shellcheck via Homebrew..."
    brew install shellcheck
    brew link --overwrite shellcheck
    
    # Install Ruby gems if needed
    if command_exists gem && [[ "$INSTALL_GEMS" == "true" ]]; then
        log_info "Installing Ruby gems..."
        gem install chef-utils -v 16.6.14
        gem install mdl
    fi
    
    log_success "macOS codecheck tools installed successfully"
}

check_shellcheck_version() {
    # Skip version check on Windows
    if [[ "$OS_NAME" == MINGW* || "$OS_NAME" == MSYS* || "$OS_NAME" == CYGWIN* ]]; then
        log_info "Skipping shellcheck version check on Windows (ShellCheck has no reference value on Windows)"
        return 0
    fi
    
    log_info "Checking shellcheck version..."
    
    if ! command_exists shellcheck; then
        log_warning "shellcheck is not installed or not in PATH"
        return 1
    fi
    
    local version_output
    if ! version_output=$(shellcheck -V 2>/dev/null); then
        log_warning "Failed to get shellcheck version"
        return 1
    fi
    
    # Robust version parsing
    local version_str
    if [[ "$version_output" =~ [Vv]ersion[[:space:]]*([0-9]+\.[0-9]+\.[0-9]+) ]]; then
        version_str="${BASH_REMATCH[1]}"
    else
        version_str=$(echo "$version_output" | grep -i version | head -1 | awk '{print $2}')
    fi
    
    if [[ -z "$version_str" ]]; then
        log_warning "Could not parse shellcheck version"
        return 1
    fi
    
    log_info "Installed shellcheck version: $version_str"
    
    # Version comparison (extract version number from SHELLCHECK_VERSION if it has 'v' prefix)
    local required_version="${SHELLCHECK_VERSION#v}"
    
    # Use sort -V for version comparison
    if printf '%s\n' "$required_version" "$version_str" | sort -V -C ; then
        log_success "shellcheck version meets requirement (>= $required_version)"
        return 0
    else
        log_warning "shellcheck version $version_str is less than $required_version"
        return 1
    fi
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--version)
                if [[ -n "${2:-}" ]]; then
                    SHELLCHECK_VERSION="$2"
                    shift 2
                else
                    log_error "'--version' requires a non-empty argument"
                    exit 1
                fi
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            --install-dir)
                if [[ -n "${2:-}" ]]; then
                    INSTALL_DIR="$2"
                    shift 2
                else
                    log_error "'--install-dir' requires a non-empty argument"
                    exit 1
                fi
                ;;
            --install-gems)
                INSTALL_GEMS="true"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

main() {
    log_info "Detected OS: $OS_NAME, Architecture: $ARCH"
    
    parse_arguments "$@"
    
    case "$OS_NAME" in
        MINGW*|MSYS*|CYGWIN*)
            log_info "Windows environment detected"
            install_windows_codecheck_tools
            ;;
        Linux)
            log_info "GNU/Linux detected"
            install_linux_codecheck_tool
            ;;
        Darwin)
            log_info "macOS detected"
            install_mac_codecheck_tools
            ;;
        *)
            log_error "Unsupported operating system: $OS_NAME"
            exit 1
            ;;
    esac
    
    if check_shellcheck_version; then
        log_success "Codecheck tools installation completed successfully"
    else
        log_warning "Installation completed with some warnings"
        if [[ "$OS_NAME" == MINGW* || "$OS_NAME" == MSYS* || "$OS_NAME" == CYGWIN* ]]; then
            log_info "On Windows, only Ruby gems were installed (ShellCheck was intentionally skipped as it has no reference value on Windows)"
        else
            log_info "Please verify the installation manually"
        fi
    fi
}

# Initialize default values
INSTALL_GEMS="false"

main "$@"