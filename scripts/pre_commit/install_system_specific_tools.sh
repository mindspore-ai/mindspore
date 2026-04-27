#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd
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

set -euo pipefail

readonly OS_NAME=$(uname -s)
readonly ARCH=$(uname -m)
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'
SHELLCHECK_VERSION="v0.7.1"

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }

# Returns 0 if version1 >= version2, 1 otherwise
compare_versions() {
    local v1="${1#v}" v2="${2#v}"
    if printf '%s\n' "$v1" "$v2" | sort -V -C 2>/dev/null; then return 0; else return 1; fi
}

version_lt() {
    local v1="${1#v}" v2="${2#v}"
    if printf '%s\n' "$v1" "$v2" | sort -V -C 2>/dev/null; then
        if [[ "$v1" != "$v2" ]]; then return 0; else return 1; fi
    else return 1; fi
}

command_exists() { command -v "$1" >/dev/null 2>&1; }

check_prerequisites() {
    local missing_tools=()
    if ! command_exists tar; then missing_tools+=("tar"); fi
    if ! command_exists curl && ! command_exists wget; then missing_tools+=("curl or wget"); fi
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        if [[ "$OS_NAME" == "Linux" ]]; then
            log_info "Please install them using your package manager:"
            log_info "  Ubuntu/Debian: sudo apt-get install ${missing_tools[*]}"
            log_info "  CentOS/RHEL: sudo yum install ${missing_tools[*]}"
        elif [[ "$OS_NAME" == "Darwin" ]]; then
            log_info "Please install them using Homebrew: brew install ${missing_tools[*]}"
        fi
        exit 1
    fi
}

download_file() {
    local url="$1" output="$2"
    if command_exists wget; then
        wget --progress=dot:giga --no-check-certificate -O "$output" "$url" 2>&1 && return 0 || return 1
    elif command_exists curl; then
        curl --ssl-no-revoke -k -f -L -o "$output" "$url" 2>&1 && return 0 || return 1
    else
        log_error "Neither curl nor wget is available"
        return 1
    fi
}

normalize_architecture() {
    local arch="$1" os_name="$2"
    if [[ "$os_name" == "Darwin" ]]; then
        if [[ "$arch" == "i386" || "$arch" == "x86_64" || "$arch" == "x64" ]]; then echo "x86_64"; return; fi
        if [[ "$arch" == "arm64" || "$arch" == "aarch64" ]]; then echo "arm64"; return; fi
    fi
    if [[ "$os_name" == "Linux" ]]; then
        if [[ "$arch" == "x86_64" || "$arch" == "amd64" || "$arch" == "x64" ]]; then echo "x86_64"; return; fi
        if [[ "$arch" == "aarch64" || "$arch" == "arm64" ]]; then echo "aarch64"; return; fi
        if [[ "$arch" == "armv6hf" || "$arch" == "armv7l" || "$arch" == "armv6l" ]]; then echo "armv6hf"; return; fi
    fi
    echo "$arch"
}

get_shellcheck_filename() {
    local version="$1" os_name="$2" arch="$3"
    local normalized_arch
    normalized_arch=$(normalize_architecture "$arch" "$os_name")
    local filename="shellcheck-${version}"
    if [[ "$os_name" == "Linux" ]]; then
        case "$normalized_arch" in
            x86_64)  filename="${filename}.linux.x86_64.tar.xz" ;;
            aarch64) filename="${filename}.linux.aarch64.tar.xz" ;;
            armv6hf) filename="${filename}.linux.armv6hf.tar.xz" ;;
            *) log_error "Unsupported Linux architecture: $arch"; exit 1 ;;
        esac
    elif [[ "$os_name" == "Darwin" ]]; then
        case "$normalized_arch" in
            x86_64) filename="${filename}.darwin.x86_64.tar.xz" ;;
            arm64)
                if version_lt "$version" "v0.10.0"; then
                    log_error "ShellCheck does not provide macOS ARM64 binaries before v0.10.0"; exit 1
                else
                    filename="${filename}.darwin.arm64.tar.xz"
                fi ;;
            *) log_error "Unsupported macOS architecture: $arch"; exit 1 ;;
        esac
    fi
    echo "$filename"
}

download_shellcheck_package() {
    local version="$1" filename="$2" output="$3"
    local github_base="https://github.com/koalaman/shellcheck/releases"
    local github_url="${github_base}/download/${version}/${filename}"
    local internal_base="http://tools.mindspore.cn/tools/check/shellcheck"
    local internal_url="${internal_base}/shellcheck-${version}/${filename}"

    log_info "Trying to download from official GitHub repository..."
    if download_file "$github_url" "$output"; then log_success "Successfully downloaded from official GitHub repository"; return 0; fi

    log_warning "Failed to download from official GitHub repository, trying internal mirror..."
    if download_file "$internal_url" "$output"; then log_success "Successfully downloaded from internal mirror"; return 0; fi

    log_error "Failed to download from both sources"; return 1
}

install_shellcheck() {
    local version="$1" install_dir="${2:-/usr/bin}"
    log_info "Installing shellcheck version: $version"
    if [[ $EUID -ne 0 ]]; then
        if ! command_exists sudo; then log_error "sudo command not found"; exit 1; fi
    fi
    check_prerequisites
    local temp_dir="/tmp"
    local filename
    filename=$(get_shellcheck_filename "$version" "$OS_NAME" "$ARCH")
    if [[ ! -w "$install_dir" && $EUID -ne 0 ]]; then
        if ! sudo -v >/dev/null 2>&1; then log_error "sudo required"; exit 1; fi
    fi
    cd "$temp_dir" || { log_error "Cannot change to temp directory"; exit 1; }
    if ! download_shellcheck_package "$version" "$filename" "$filename"; then
        log_error "Failed to download shellcheck"; exit 1
    fi
    if [[ ! -f "$filename" ]]; then log_error "Downloaded file not found"; exit 1; fi
    log_info "Extracting shellcheck package..."
    if ! tar -xf "$filename"; then log_error "Failed to extract"; exit 1; fi
    local base_name="${filename%.tar.xz}"
    local shellcheck_binary
    if [[ -f "${base_name}/shellcheck" ]]; then
        shellcheck_binary="${base_name}/shellcheck"
    elif [[ -f "shellcheck" ]]; then
        shellcheck_binary="shellcheck"
    else
        local extracted_dir
        extracted_dir=$(find . -maxdepth 1 -type d -name "shellcheck*" | head -1)
        if [[ -n "$extracted_dir" ]] && [[ -f "${extracted_dir}/shellcheck" ]]; then
            shellcheck_binary="${extracted_dir}/shellcheck"
        else
            log_error "Could not find shellcheck binary in extracted files"; exit 1
        fi
    fi
    if [[ ! -f "$shellcheck_binary" ]]; then log_error "ShellCheck binary not found"; exit 1; fi
    log_info "Installing shellcheck to $install_dir/shellcheck"
    if [[ $EUID -eq 0 ]]; then
        rm -f "$install_dir/shellcheck" 2>/dev/null || true
        mv "$shellcheck_binary" "$install_dir/shellcheck"
        chmod 755 "$install_dir/shellcheck"
    else
        sudo rm -f "$install_dir/shellcheck" 2>/dev/null || true
        sudo mv "$shellcheck_binary" "$install_dir/shellcheck"
        sudo chmod 755 "$install_dir/shellcheck"
    fi
    log_info "Cleaning up temporary files..."
    rm -rf "$base_name" 2>/dev/null || true
    rm -f "$filename" 2>/dev/null || true
    find . -maxdepth 1 -type d -name "shellcheck*" -exec rm -rf {} + 2>/dev/null || true
    log_success "Shellcheck installed successfully to $install_dir/shellcheck"
}

install_linux_codecheck_tool() {
    log_info "Installing Linux codecheck tools..."
    local install_dir="${INSTALL_DIR:-/usr/bin}"
    install_shellcheck "$SHELLCHECK_VERSION" "$install_dir"
    if command_exists gem && [[ "$INSTALL_GEMS" == "true" ]]; then
        log_info "Installing Ruby gems..."
        gem sources --add https://gems.ruby-china.com/ --remove https://rubygems.org/
        gem install chef-utils -v 16.6.14
        gem install mdl
        log_success "Ruby gems installed successfully"
    fi
    log_success "Linux codecheck tools installation completed"
}

install_windows_codecheck_tools() {
    log_info "Installing Windows codecheck tools..."
    log_warning "================================================================"
    log_warning "ShellCheck intentionally skipped on Windows"
    log_warning "Reason: ShellCheck has no reference value in Windows environment"
    log_warning "================================================================"
    if command_exists gem && [[ "$INSTALL_GEMS" == "true" ]]; then
        log_info "Installing Ruby gems..."
        gem sources --add https://gems.ruby-china.com/ --remove https://rubygems.org/
        gem install chef-utils -v 16.6.14
        gem install mdl
        log_success "Ruby gems installed successfully"
    else
        log_info "Skipping Ruby gems installation (gem not found or not requested)"
    fi
    log_success "Windows codecheck tools installation completed"
    log_info "Note: ShellCheck was intentionally not installed as it has no reference value on Windows"
}

install_mac_codecheck_tools() {
    log_info "Installing macOS codecheck tools..."
    log_warning "================================================================"
    log_warning "ShellCheck intentionally skipped on macOS"
    log_warning "Reason: ShellCheck has no reference value in macOS environment"
    log_warning "================================================================"
    if command_exists gem && [[ "$INSTALL_GEMS" == "true" ]]; then
        log_info "Installing Ruby gems..."
        gem install chef-utils -v 16.6.14
        gem install mdl
        log_success "Ruby gems installed successfully"
    else
        log_info "Skipping Ruby gems installation (gem not found or not requested)"
    fi
    log_success "macOS codecheck tools installation completed"
    log_info "Note: ShellCheck was intentionally not installed as it has no reference value on macOS"
}

check_shellcheck_version() {
    if [[ "$OS_NAME" == MINGW* || "$OS_NAME" == MSYS* || "$OS_NAME" == CYGWIN* ]]; then
        log_info "Skipping shellcheck version check on Windows"
        return 0
    fi
    if [[ "$OS_NAME" == "Darwin" ]]; then
        log_info "Skipping shellcheck version check on macOS"
        return 0
    fi
    log_info "Checking shellcheck version..."
    if ! command_exists shellcheck; then log_warning "shellcheck is not installed"; return 1; fi
    local version_output
    if ! version_output=$(shellcheck -V 2>/dev/null); then log_warning "Failed to get shellcheck version"; return 1; fi
    local version_str
    if [[ "$version_output" =~ [Vv]ersion[[:space:]]*([0-9]+\.[0-9]+\.[0-9]+) ]]; then
        version_str="${BASH_REMATCH[1]}"
    else
        version_str=$(echo "$version_output" | grep -i version | head -1 | awk '{print $2}')
    fi
    if [[ -z "$version_str" ]]; then log_warning "Could not parse shellcheck version"; return 1; fi
    log_info "Installed shellcheck version: $version_str"
    local required_version="${SHELLCHECK_VERSION#v}"
    if printf '%s\n' "$required_version" "$version_str" | sort -V -C ; then
        log_success "shellcheck version meets requirement (>= $required_version)"
        return 0
    else
        log_warning "shellcheck version $version_str is less than $required_version"
        return 1
    fi
}

main() {
    log_info "Detected OS: $OS_NAME, Architecture: $ARCH"
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
            log_info "On Windows, only Ruby gems were installed (ShellCheck was intentionally skipped)"
        elif [[ "$OS_NAME" == "Darwin" ]]; then
            log_info "On macOS, ShellCheck was intentionally not installed"
        else
            log_info "Please verify the installation manually"
        fi
    fi
}

INSTALL_GEMS="false"
main "$@"