#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'
readonly SUPPORTED_PYTHON_VERSIONS=("3.9" "3.10" "3.11" "3.12")
readonly PYPI_MIRRORS=(
    "https://repo.huaweicloud.com/repository/pypi/simple"
)

detect_architecture() {
    local arch
    arch=$(uname -m)
    case "$arch" in
        arm64|aarch64) echo "arm64" ;;
        x86_64) echo "x86_64" ;;
        *) echo "$arch" ;;
    esac
}

detect_os() {
    local os_name
    os_name=$(uname -s)
    case "$os_name" in
        Darwin) echo "macos" ;;
        Linux)
            if [[ -f /etc/debian_version ]] || [[ -f /etc/lsb-release ]]; then echo "debian"
            elif [[ -f /etc/redhat-release ]]; then echo "redhat"
            else echo "linux"
            fi ;;
        MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
        *) echo "$os_name" ;;
    esac
}

is_arm64_mac() { [[ "$(detect_os)" == "macos" ]] && [[ "$(detect_architecture)" == "arm64" ]]; }
is_debian_based() { command -v apt-get >/dev/null 2>&1 && ([[ -f /etc/debian_version ]] || [[ -f /etc/lsb-release ]]); }

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }

command_exists() { command -v "$1" >/dev/null 2>&1; }

find_python() {
    for cmd in python python3 py; do
        if command_exists "$cmd"; then
            if $cmd -c "import sys; print('.'.join(str(x) for x in sys.version_info[:2]))" >/dev/null 2>&1; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    log_warning "No working Python command found (tried: python, python3, py)"
    return 1
}

find_pip() {
    local python_cmd="$1"
    if $python_cmd -m pip --version >/dev/null 2>&1; then
        echo "$python_cmd -m pip"
        return 0
    fi
    if command_exists pip3; then echo "pip3"; return 0; fi
    if command_exists pip; then echo "pip"; return 0; fi
    if $python_cmd -c "import sys; print(f'pip{sys.version_info[0]}.{sys.version_info[1]}')" >/dev/null 2>&1; then
        local pip_cmd
        pip_cmd=$($python_cmd -c "import sys; print(f'pip{sys.version_info[0]}.{sys.version_info[1]}')" 2>/dev/null)
        if command_exists "$pip_cmd"; then echo "$pip_cmd"; return 0; fi
    fi
    log_warning "No working pip command found for $python_cmd"
    return 1
}

compare_versions() {
    local version1="$1" version2="$2"
    $PYTHON_CMD -c "import sys; v1=tuple(map(int,'$version1'.split('.'))); v2=tuple(map(int,'$version2'.split('.'))); sys.exit(0 if v1>=v2 else 1)" 2>/dev/null
    return $?
}

check_network() {
    log_info "Checking network connectivity to PyPI..."
    local endpoints=(
        "https://pypi.org"
        "https://repo.huaweicloud.com"
    )
    for endpoint in "${endpoints[@]}"; do
        if $PYTHON_CMD -c "
import urllib.request
import ssl
context = ssl.create_default_context()
try:
    urllib.request.urlopen('$endpoint', timeout=10, context=context)
    print('SUCCESS')
except Exception:
    pass
" 2>/dev/null | grep -q "SUCCESS"; then
            log_success "Network connectivity confirmed"
            return 0
        fi
    done
    log_warning "Cannot reach external servers. This may be due to proxy settings or network restrictions."
    return 1
}

check_python() {
    PYTHON_CMD=$(find_python)
    if [[ -z "$PYTHON_CMD" ]]; then
        log_error "Python is not installed. Please install Python 3.9 or higher"
        log_error "Tried commands: python, python3, py"
        return 1
    fi
    log_info "Using Python command: $PYTHON_CMD"
    local python_version=""
    python_version=$($PYTHON_CMD -c "import sys; print('.'.join(str(x) for x in sys.version_info[:2]))" 2>/dev/null || echo "")
    if [[ -z "$python_version" ]]; then
        python_version=$($PYTHON_CMD --version 2>&1 | grep -oP '\d+\.\d+' | head -1 || echo "")
    fi
    if [[ -z "$python_version" ]]; then
        log_error "Failed to determine Python version using $PYTHON_CMD"
        log_error "Please ensure Python is properly installed"
        return 1
    fi
    log_info "Found Python version: $python_version"
    if ! compare_versions "$python_version" "3.9"; then
        log_error "Python version '$python_version' is not supported"
        log_error "This script requires Python 3.9 or higher"
        log_error "Please upgrade your Python installation and try again"
        return 1
    fi
    local version_ok=false
    for version in "${SUPPORTED_PYTHON_VERSIONS[@]}"; do
        if [[ "$python_version" == "$version" ]]; then version_ok=true; break; fi
    done
    if ! $version_ok; then
        log_warning "Python version '$python_version' is not in the officially supported list"
        log_warning "Supported versions: ${SUPPORTED_PYTHON_VERSIONS[*]}"
        log_warning "Continuing anyway, but may encounter compatibility issues..."
    else
        log_success "Python version $python_version is supported"
    fi
    return 0
}

check_pip() {
    PIP_CMD=$(find_pip "$PYTHON_CMD")
    if [[ -z "$PIP_CMD" ]]; then
        log_error "pip is not installed. Please install pip first"
        log_info "Please install pip from the official Python packaging documentation."
        return 1
    fi
    if [[ "$PIP_CMD" == "pip3" ]] || [[ "$PIP_CMD" == "pip" ]] || [[ "$PIP_CMD" =~ ^pip[0-9.]+$ ]]; then
        if ! $PIP_CMD --version >/dev/null 2>&1; then
            log_error "pip command '$PIP_CMD' is not working properly"
            return 1
        fi
        log_info "Found pip: $($PIP_CMD --version 2>/dev/null | head -n1 || echo "pip available")"
    else
        if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
            log_error "pip is not properly installed or not accessible"
            return 1
        fi
        log_info "Found pip: $($PYTHON_CMD -m pip --version 2>/dev/null | head -n1 || echo "pip available")"
    fi
    return 0
}

execute_pip() {
    local args="$*"
    if [[ "$PIP_CMD" == "pip3" ]] || [[ "$PIP_CMD" == "pip" ]]; then
        $PIP_CMD $args
    else
        $PYTHON_CMD -m pip $args
    fi
}

check_package_installed() {
    local package_name="$1" package_spec="$2" version=""
    if [[ "$package_spec" == *"=="* ]]; then
        version="${package_spec#*==}"
        package_name="${package_spec%%==*}"
    fi
    local installed_version=""
    installed_version=$($PYTHON_CMD -c "
try:
    import importlib.metadata; print(importlib.metadata.version('$package_name'))
except ImportError:
    try:
        import pkg_resources; print(pkg_resources.get_distribution('$package_name').version)
    except: print('')
except: print('')" 2>/dev/null || echo "")
    if [[ -n "$installed_version" ]]; then
        if [[ -z "$version" ]] || [[ "$installed_version" == "$version" ]]; then
            log_info "$package_name $installed_version is already installed"
            return 0
        else
            log_info "$package_name version $installed_version installed, but requested $version"
            return 1
        fi
    fi
    return 1
}

arm64_mac_safe_install() {
    local package_name="$1" package_spec="$2"
    log_info "ARM64 Mac detected, using special handling for $package_name..."
    if [[ "$package_name" == "pylint" ]]; then
        log_info "Special handling for pylint on ARM64 Mac..."
        execute_pip install --upgrade "importlib-metadata<7.0" 2>/dev/null || true
        if execute_pip install "$package_spec" --no-deps 2>/dev/null; then
            execute_pip install "$package_spec" 2>/dev/null && return 0
        fi
        if [[ "$package_spec" == "pylint==3.3.7" ]]; then
            log_info "Trying alternative installation method for pylint 3.3.7..."
            execute_pip install "tomli>=1.1" "astroid<=3.4.0.dev0,>=3.3.8" \
                "typing-extensions>=3.10" "tomlkit>=0.10.1" "isort!=5.13,<7,>=4.2.5" \
                "platformdirs>=2.2" "dill>=0.2" "mccabe<0.8,>=0.6" \
                "importlib-metadata>=4.6.0,<7.0" 2>/dev/null || true
            if execute_pip install --no-deps "$package_spec" 2>/dev/null; then return 0; fi
        fi
    fi
    log_info "Trying standard installation for $package_name on ARM64 Mac..."
    if execute_pip install "$package_spec" 2>/dev/null; then return 0; fi
    return 1
}

debian_safe_install() {
    local package_name="$1" package_spec="$2"
    log_info "Debian/Ubuntu system detected, using safe installation method..."
    if execute_pip install --break-system-packages --upgrade --force-reinstall "$package_spec" 2>/dev/null; then
        return 0
    else
        log_warning "Failed with --break-system-packages, trying --user install"
        if execute_pip install --user --upgrade --force-reinstall "$package_spec" 2>/dev/null; then return 0; fi
    fi
    return 1
}

install_with_retry() {
    local package_name="$1" package_spec="$2" max_retries=3 retry_count=0
    while [[ $retry_count -le $max_retries ]]; do
        case $retry_count in
            0)
                if is_arm64_mac; then
                    log_info "ARM64 Mac: Attempting direct installation..."
                    if execute_pip install "$package_spec" 2>/dev/null; then return 0; fi
                else
                    log_info "Attempting direct installation of $package_name..."
                    if execute_pip install --upgrade --force-reinstall "$package_spec" 2>/dev/null; then return 0; fi
                fi ;;
            1)
                if is_arm64_mac; then
                    if arm64_mac_safe_install "$package_name" "$package_spec"; then return 0; fi
                elif is_debian_based; then
                    if debian_safe_install "$package_name" "$package_spec"; then return 0; fi
                fi ;;
            2)
                log_info "Trying installation with PyPI mirrors..."
                for mirror in "${PYPI_MIRRORS[@]}"; do
                    log_info "Attempting with mirror: $(echo "$mirror" | cut -d'/' -f3)"
                    if execute_pip install -i "$mirror" "$package_spec" 2>/dev/null; then return 0; fi
                done ;;
            3)
                log_info "Trying installation without dependencies..."
                if execute_pip install --no-deps "$package_spec" 2>/dev/null; then
                    log_warning "Installed without dependencies, may have limited functionality"
                    return 0
                fi ;;
        esac
        ((retry_count++))
        if [[ $retry_count -le $max_retries ]]; then
            log_warning "Installation attempt $retry_count failed for $package_name, retrying..."
            sleep 2
        fi
    done
    return 1
}

install_package() {
    local package_name="$1" package_spec="$2"
    if is_arm64_mac && [[ "$package_name" == "clang-format" ]]; then
        log_info "ARM64 Mac detected, adjusting clang-format version for compatibility..."
        package_spec="clang-format==18.1.4"
    fi
    log_info "Processing tool: $package_name"
    if check_package_installed "$package_name" "$package_spec"; then
        log_success "$package_name is already installed with correct version"
        return 0
    fi
    log_info "Installing $package_name..."
    if install_with_retry "$package_name" "$package_spec"; then
        log_success "Successfully installed $package_name"
        return 0
    else
        log_error "All installation attempts failed for $package_name"
        if is_arm64_mac; then
            log_info "=== ARM64 Mac Troubleshooting ==="
            log_info "Try these manual steps:"
            log_info "1. Update pip: $PYTHON_CMD -m pip install --upgrade pip"
            log_info "2. Use virtual environment: $PYTHON_CMD -m venv venv && source venv/bin/activate && pip install $package_spec"
            log_info "3. Try newer version: pip install ${package_name%%==*}"
        elif is_debian_based; then
            log_info "=== Debian/Ubuntu Troubleshooting ==="
            log_info "1. Install using system package manager: sudo apt-get install python3-${package_name%%==*}"
            log_info "2. Use virtual environment as shown above"
        else
            log_info "=== General Troubleshooting ==="
            log_info "1. Check network/proxy settings"
            log_info "2. Configure pip proxy according to your network environment."
            log_info "3. Use virtual environment"
        fi
        return 1
    fi
}

convert_windows_path_to_git_bash() {
    local windows_path="$1"
    local unix_path=$(echo "$windows_path" | sed 's/\\/\//g')
    if [[ "$unix_path" =~ ^[A-Za-z]: ]]; then
        local drive_letter=$(echo "$unix_path" | cut -c1 | tr '[:upper:]' '[:lower:]')
        unix_path="/$drive_letter$(echo "$unix_path" | cut -c3-)"
    fi
    echo "$unix_path"
}

check_path_contains() {
    local target_dir="$1"
    target_dir=$(echo "$target_dir" | sed 's|/*$||')
    IFS=':' read -ra path_dirs <<< "$PATH"
    for dir in "${path_dirs[@]}"; do
        dir=$(echo "$dir" | sed 's|/*$||')
        if [[ "$dir" == "$target_dir" ]]; then return 0; fi
        local os_type=$(detect_os)
        if [[ "$os_type" == "windows" ]]; then
            local windows_dir=$(echo "$dir" | sed 's/^\///' | sed 's/^c\//C:\//i' | sed 's/\//\\/g')
            if [[ "$windows_dir" == "$target_dir" ]]; then return 0; fi
            local git_bash_target=$(convert_windows_path_to_git_bash "$target_dir")
            if [[ "$dir" == "$git_bash_target" ]]; then return 0; fi
        fi
    done
    return 1
}

get_python_user_scripts_dir() {
    local python_cmd="$1" scripts_dir=""
    scripts_dir=$($python_cmd -c "
import sys, os, site
def get_user_scripts_dir():
    try:
        if hasattr(site, 'getuserbase'): user_base = site.getuserbase()
        elif hasattr(site, 'USER_BASE'): user_base = site.USER_BASE
        else:
            if sys.platform == 'win32':
                appdata = os.environ.get('APPDATA', '')
                user_base = os.path.join(appdata, 'Python') if appdata else os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'Python')
            else:
                user_base = os.path.join(os.path.expanduser('~'), '.local')
        scripts_name = 'Scripts' if sys.platform == 'win32' else 'bin'
        if sys.platform == 'win32':
            python_version = f'Python{sys.version_info[0]}{sys.version_info[1]}'
            return os.path.join(user_base, python_version, scripts_name)
        else:
            return os.path.join(user_base, scripts_name)
    except: return ''
try:
    d = get_user_scripts_dir()
    if d and (os.path.exists(d) or os.path.exists(os.path.dirname(d))):
        print(d)
    else:
        if sys.platform == 'win32':
            appdata = os.environ.get('APPDATA', '')
            python_version = f'Python{sys.version_info[0]}{sys.version_info[1]}'
            if appdata:
                print(os.path.join(appdata, 'Python', python_version, 'Scripts'))
            else:
                print(os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'Python', python_version, 'Scripts'))
        else:
            print(os.path.join(os.path.expanduser('~'), '.local', 'bin'))
except:
    if sys.platform == 'win32':
        python_version = f'Python{sys.version_info[0]}{sys.version_info[1]}'
        print(os.path.join(os.environ.get('APPDATA', ''), 'Python', python_version, 'Scripts'))
    else:
        print(os.path.join(os.path.expanduser('~'), '.local', 'bin'))" 2>/dev/null || echo "")
    echo "$scripts_dir"
}

check_and_configure_path() {
    local python_cmd="$1" scripts_dir=""
    log_info "Checking PATH environment variable configuration..."
    scripts_dir=$(get_python_user_scripts_dir "$python_cmd")
    if [[ -z "$scripts_dir" ]]; then
        log_warning "Cannot determine Python user scripts directory"
        return 1
    fi
    log_info "Python scripts directory: $scripts_dir"
    if [[ ! -d "$scripts_dir" ]]; then
        log_info "Directory does not exist yet (will be created during installation)"
    fi
    if check_path_contains "$scripts_dir"; then
        log_success "PATH is correctly configured"
        return 0
    else
        log_warning "PATH does not contain Python scripts directory"
        show_path_configuration_guide "$scripts_dir"
        return 1
    fi
}

show_path_configuration_guide() {
    local scripts_dir="$1"
    log_info "=== PATH Configuration Guide ==="
    log_info ""
    log_info "Add the following directory to your PATH:"
    log_info "  $scripts_dir"
    log_info ""
    local os_type=$(detect_os)
    case "$os_type" in
        windows|MINGW*|MSYS*|CYGWIN*)
            local git_bash_path=$(convert_windows_path_to_git_bash "$scripts_dir")
            log_info "For Windows (Git Bash/MINGW64):"
            log_info "  1. Edit ~/.bashrc: nano ~/.bashrc"
            log_info "  2. Add: export PATH=\"\$PATH:$git_bash_path\""
            log_info "  3. Reload: source ~/.bashrc"
            log_info ""
            log_info "For Windows System Properties (permanent):"
            log_info "  1. Right-click 'This PC' > Properties > Advanced system settings"
            log_info "  2. Click 'Environment Variables'"
            log_info "  3. Edit 'Path' under 'User variables'"
            log_info "  4. Add: $scripts_dir" ;;
        macos)
            log_info "For macOS (Bash):"
            log_info "  1. Edit ~/.bash_profile: nano ~/.bash_profile"
            log_info "  2. Add: export PATH=\"\$PATH:$scripts_dir\""
            log_info "  3. Reload: source ~/.bash_profile"
            log_info ""
            log_info "For macOS (Zsh):"
            log_info "  1. Edit ~/.zshrc: nano ~/.zshrc"
            log_info "  2. Add: export PATH=\"\$PATH:$scripts_dir\""
            log_info "  3. Reload: source ~/.zshrc" ;;
        linux|debian|redhat)
            log_info "For Linux:"
            log_info "  1. Edit ~/.bashrc: nano ~/.bashrc"
            log_info "  2. Add: export PATH=\"\$PATH:$scripts_dir\""
            log_info "  3. Reload: source ~/.bashrc" ;;
        *)
            log_info "General method:"
            log_info "  Add to your shell configuration file:"
            log_info "  export PATH=\"\$PATH:$scripts_dir\"" ;;
    esac
    log_info ""
    log_info "After configuration, restart your terminal and verify with:"
    log_info "  echo \$PATH | grep -i \"$(basename "$scripts_dir")\""
    log_info "========================================"
}

ask_temporary_path_config() {
    local scripts_dir="$1"
    log_info "Would you like to temporarily add to PATH for this session? [y/N]"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        log_info "Temporarily adding directory to PATH..."
        local os_type=$(detect_os)
        local path_to_add="$scripts_dir"
        if [[ "$os_type" == "windows" ]]; then
            path_to_add=$(convert_windows_path_to_git_bash "$scripts_dir")
            log_info "Windows environment detected, using Git Bash format path: $path_to_add"
        fi
        if check_path_contains "$scripts_dir"; then
            log_success "Directory already in PATH"
            return 0
        fi
        export PATH="$PATH:$path_to_add"
        if check_path_contains "$scripts_dir"; then
            log_success "Temporarily added to PATH (current session only)"
            log_info "Added path: $path_to_add"
            return 0
        else
            log_error "Failed to add to PATH"
            log_info "Debug: PATH is now: $PATH"
            return 1
        fi
    else
        log_info "Please configure PATH manually as shown above."
        return 1
    fi
}

check_tool_version() {
    local tool_name="$1" check_cmd="$2"
    log_info "Verifying $tool_name installation..."
    if eval "$check_cmd" 2>/dev/null; then
        log_success "$tool_name is working properly"
        return 0
    else
        log_warning "$tool_name not accessible or not in PATH!"
        return 1
    fi
}

verify_installations() {
    local tools=("$@") failed_checks=0
    log_info "Verifying tool installations..."
    for tool_info in "${tools[@]}"; do
        IFS='|' read -r tool_name package_spec check_cmd <<< "$tool_info"
        if ! check_tool_version "$tool_name" "$check_cmd"; then ((failed_checks++)); fi
    done
    return $failed_checks
}

main() {
    log_info "Starting Code Check Tools Installation..."
    log_info "Version: 2.0.0"
    log_info "========================================"
    local os_type=$(detect_os)
    local arch=$(detect_architecture)
    log_info "Detected OS: $os_type, Architecture: $arch"
    if [[ "$os_type" == "windows" ]]; then
        log_info "Windows Git Bash environment detected, applying compatibility fixes..."
        if [[ -n "$USERNAME" ]]; then local win_user="$USERNAME"
        elif [[ -n "$USER" ]]; then local win_user="$USER"
        else local win_user=$(echo "$HOME" | grep -oP '/home/\K[^/]+' || echo "")
        fi
        if [[ -n "$win_user" ]]; then
            local possible_paths=(
                "/c/Users/$win_user/AppData/Roaming/Python/Python39/Scripts"
                "/c/Users/$win_user/AppData/Roaming/Python/Python310/Scripts"
                "/c/Users/$win_user/AppData/Roaming/Python/Python311/Scripts"
                "/c/Users/$win_user/AppData/Roaming/Python/Python312/Scripts"
            )
            for test_path in "${possible_paths[@]}"; do
                if [[ -d "$test_path" ]] && ! check_path_contains "$test_path"; then
                    export PATH="$PATH:$test_path"
                    log_info "Auto-added Git Bash path to PATH: $test_path"
                fi
            done
        fi
    fi
    log_info "Performing system checks..."
    if ! check_python; then exit 1; fi
    if ! check_pip; then exit 1; fi
    if ! check_and_configure_path "$PYTHON_CMD"; then
        local scripts_dir=$(get_python_user_scripts_dir "$PYTHON_CMD")
        if [[ -n "$scripts_dir" ]]; then
            if ask_temporary_path_config "$scripts_dir"; then
                log_info "Continuing with temporary PATH configuration..."
            else
                log_warning "Tools may not be accessible after installation"
                log_info "Continue anyway? [y/N]"
                read -r continue_response
                if [[ ! "$continue_response" =~ ^[Yy]$ ]]; then
                    log_info "Please configure PATH first, then run this script again."
                    exit 1
                fi
            fi
        fi
    fi
    if ! check_network; then
        log_warning "Network issues detected, using alternative installation methods"
    fi
    declare -a tools=(
        "cmakelint|cmakelint==1.4.1|cmakelint --version"
        "codespell|codespell==2.0.0|codespell --version"
        "cpplint|cpplint==2.0.2|cpplint --version"
        "lizard|lizard==1.17.19|lizard --version"
        "pylint|pylint==3.3.7|pylint --version"
        "clang-format|clang-format==18.1.8|clang-format --version"
    )
    local failed_installations=0 successful_installations=0
    log_info "Installing code check tools..."
    log_info "========================================"
    for tool_info in "${tools[@]}"; do
        IFS='|' read -r tool_name package_spec check_cmd <<< "$tool_info"
        if install_package "$tool_name" "$package_spec"; then
            ((successful_installations++))
        else
            ((failed_installations++))
        fi
        echo "----------------------------------------"
    done
    local failed_checks=0
    if verify_installations "${tools[@]}"; then failed_checks=$?; fi
    echo
    log_info "=== Installation Summary ==="
    if [[ $failed_installations -eq 0 && $failed_checks -eq 0 ]]; then
        log_success "All code check tools installed successfully!"
        log_info ""
        log_info "Tools available:"
        log_info "  • cmakelint    - CMake file linter"
        log_info "  • codespell    - Spelling checker for code"
        log_info "  • cpplint      - C++ style checker"
        log_info "  • lizard       - Cyclomatic complexity analyzer"
        log_info "  • pylint       - Python code analyzer"
        log_info "  • clang-format - C/C++ code formatter"
    else
        log_warning "Installation completed with issues:"
        log_warning "  • Successfully installed: $successful_installations"
        log_warning "  • Failed installations: $failed_installations"
        log_warning "  • Failed verifications: $failed_checks"
        if [[ $failed_checks -gt 0 ]]; then
            local scripts_dir=$(get_python_user_scripts_dir "$PYTHON_CMD")
            if [[ -n "$scripts_dir" ]]; then
                log_info ""
                log_info "=== IMPORTANT: PATH Configuration Required ==="
                log_info "Add to PATH: $scripts_dir"
                log_info "Then restart your terminal or run: source ~/.bashrc"
                if [[ "$(detect_os)" == "windows" ]]; then
                    local git_bash_path=$(convert_windows_path_to_git_bash "$scripts_dir")
                    log_info "For Git Bash, add this to ~/.bashrc:"
                    log_info "  export PATH=\"\$PATH:$git_bash_path\""
                fi
            fi
        fi
    fi
    log_info ""
    log_info "=== Next Steps ==="
    log_info "1. For Windows environment, please execute commands in the \`git bash\` window."
    log_info "2. Configure the \`core.hooksPath\` parameter: git config core.hooksPath scripts/pre_commit/githooks"
    log_info "3. pre-push does not need to be executed manually. Each \`git push\` operation will automatically trigger pre-push to scan the code being pushed."
    log_info ""
    log_info "For technical support, please refer to the documentation or contact the technical support team."
    log_info "Please refer to scripts/pre_commit/README_CN.md for technical support."
    log_info "========================================"
    return $failed_installations
}

PYTHON_CMD=""
PIP_CMD=""

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
    exit $?
fi