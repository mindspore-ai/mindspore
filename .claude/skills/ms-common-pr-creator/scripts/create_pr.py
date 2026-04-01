#!/usr/bin/env python3
"""
GitCode PR Creation Script

Creates a pull request on GitCode using the REST API.
Automatically detects repository info and branch names.
The PR body is generated externally (by LLM) and passed via --body-file.
"""

import json
import os
import re
import subprocess
import sys
from typing import Optional, Tuple


# ============================================================================
# Configuration Loading
# ============================================================================

# All gitcode-skills share the same config file and env var names so users
# only configure credentials once.
#
# Resolution order (first non-empty wins):
#   1. Environment variables  (CI-friendly, ephemeral)
#   2. Config file             (interactive-friendly, persistent)
#
# Config file path: ~/.config/gitcode-skills/config.json
# Expected keys:    user_id, access_token
#
# Environment variables:
#   GITCODE_USER_ID
#   GITCODE_ACCESS_TOKEN

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SKILL_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
_README_FILE = os.path.join(_SKILL_DIR, "README.md")

_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", "gitcode-skills")
_CONFIG_FILE = os.path.join(_CONFIG_DIR, "config.json")

_SETUP_GUIDE = """\
[ERROR] GitCode credentials not configured.

Configure credentials locally, then retry.

Local setup guide:
  {readme_file}

Supported local configuration sources:
  Config file: {config_file}
  Environment variables: GITCODE_USER_ID, GITCODE_ACCESS_TOKEN

Do not paste tokens, environment-variable values, or config contents into chat.
""".format(readme_file=_README_FILE, config_file=_CONFIG_FILE)


def load_config():
    """Load GitCode credentials from environment variables or config file.

    Returns:
        dict with keys: user_id, access_token.

    Raises:
        SystemExit(4) if credentials are not configured.
    """
    user_id = os.environ.get("GITCODE_USER_ID", "").strip()
    access_token = os.environ.get("GITCODE_ACCESS_TOKEN", "").strip()

    if user_id and access_token:
        return {"user_id": user_id, "access_token": access_token}

    # Fall through to config file
    if os.path.isfile(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE, encoding="utf-8") as f:
                data = json.load(f)
            file_owner = str(data.get("user_id", "")).strip()
            file_token = str(data.get("access_token", "")).strip()
            if file_owner and file_token:
                return {"user_id": file_owner, "access_token": file_token}
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Failed to read {_CONFIG_FILE}: {e}", file=sys.stderr)

    # Nothing configured — print guide and exit
    print(_SETUP_GUIDE, file=sys.stderr)
    output_error(
        "GitCode credentials not configured",
        "AUTH_NOT_CONFIGURED",
        f"No credentials found in environment variables or {_CONFIG_FILE}. "
        "See stderr for setup instructions.",
    )
    sys.exit(4)


def output_error(error, error_code, details=None):
    """Print error JSON to stdout and message to stderr."""
    result = {
        "success": False,
        "error": error,
        "error_code": error_code,
        "details": details,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[ERROR] {error}: {details}", file=sys.stderr)


API_BASE = "https://api.gitcode.com/api/v5"


# ============================================================================
# Body Post-Processing
# ============================================================================

def sanitize_pr_body(body: str) -> str:
    """Post-process LLM-generated PR body to strip leaked YAML frontmatter.

    Safety net applied before the API call — if the LLM accidentally
    includes the template's YAML frontmatter (--- block at the start),
    this strips it so the API receives a clean PR body.
    """
    stripped = body.lstrip()
    if stripped.startswith('---'):
        # Find the closing --- on a subsequent line
        end = stripped.find('\n---', 3)
        if end != -1:
            body = stripped[end + 4:].lstrip('\n')

    return body


def build_pr_web_url(result: dict, upstream_owner: str, repo: str) -> str:
    """Build a canonical GitCode PR URL for user-facing output."""
    number = result.get("number")
    if number not in (None, ""):
        return f"https://gitcode.com/{upstream_owner}/{repo}/pull/{number}"

    html_url = str(result.get("html_url", "")).strip()
    if html_url:
        return html_url.replace("/pulls/", "/pull/")

    return "N/A"


def _tokenize_name(name: str) -> list[str]:
    """Split a repository or project name into comparable lowercase tokens."""
    return [token for token in re.split(r'[^a-z0-9]+', name.lower()) if token]


def _project_matches_repo_name(project_name: str, repo_name: str) -> bool:
    """Match project names on token boundaries instead of raw substring checks."""
    project_tokens = _tokenize_name(project_name)
    repo_tokens = _tokenize_name(repo_name)
    if not project_tokens or not repo_tokens:
        return False
    window = len(project_tokens)
    return any(repo_tokens[i:i + window] == project_tokens for i in range(len(repo_tokens) - window + 1))


# ============================================================================
# Template Resolution (auto-discovery)
# ============================================================================

# Templates are discovered from two directories (scanned in order):
#
#   1. User templates:  ~/.config/gitcode-skills/pr_templates/   (persistent)
#   2. Skill templates: <skill_dir>/templates/                    (bundled)
#
# User templates take precedence: if the same project name appears in both
# directories, the user template wins. This way skill upgrades never overwrite
# user-added templates.
#
# Each .md file declares its scope via YAML frontmatter:
#
#   ---
#   projects:
#     - mindspore
#     - docs
#   default: true
#   ---
#   (template body follows)

_BUNDLED_TEMPLATE_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "templates"))
_USER_TEMPLATE_DIR = os.path.join(_CONFIG_DIR, "pr_templates")


def _parse_frontmatter(content: str) -> Tuple[dict, str]:
    """Parse YAML frontmatter from template content.

    Supports a minimal subset: simple key-value pairs and string lists.
    Does not require the pyyaml package.

    Returns:
        (metadata_dict, content_without_frontmatter).
        If no frontmatter found, returns ({}, original_content).
    """
    if not content.startswith('---'):
        return {}, content

    end = content.find('\n---', 3)
    if end == -1:
        return {}, content

    fm_text = content[3:end].strip()
    body = content[end + 4:]
    if body.startswith('\n'):
        body = body[1:]

    metadata = {}
    current_key = None
    current_list = None

    for line in fm_text.split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue

        if stripped.startswith('- '):
            if current_list is not None:
                current_list.append(stripped[2:].strip())
            continue

        if ':' in stripped:
            # Flush previous list
            if current_key is not None and current_list is not None:
                metadata[current_key] = current_list

            key, _, value = stripped.partition(':')
            key = key.strip()
            value = value.strip()

            if not value:
                current_key = key
                current_list = []
            else:
                if value.lower() == 'true':
                    metadata[key] = True
                elif value.lower() == 'false':
                    metadata[key] = False
                else:
                    metadata[key] = value
                current_key = None
                current_list = None

    # Flush last list
    if current_key is not None and current_list is not None:
        metadata[current_key] = current_list

    return metadata, body


def _scan_template_dir(directory: str, mapping: dict, default_path: Optional[str]) -> Tuple[dict, Optional[str]]:
    """Scan a single directory for template files and merge into mapping.

    Existing entries in mapping are NOT overwritten — this gives earlier
    directories higher priority.

    Returns:
        Updated (mapping, default_path).
    """
    if not os.path.isdir(directory):
        return mapping, default_path

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith('.md'):
            continue

        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, encoding='utf-8') as f:
                content = f.read()
        except OSError:
            continue

        metadata, _ = _parse_frontmatter(content)

        for project in metadata.get('projects', []):
            key = project.lower()
            if key not in mapping:  # earlier directory wins
                mapping[key] = filepath

        if metadata.get('default', False) and default_path is None:
            default_path = filepath

    return mapping, default_path


def _discover_templates() -> Tuple[dict, Optional[str]]:
    """Scan user and bundled template directories, build project-to-path mapping.

    Scan order (first match wins):
      1. User templates:   ~/.config/gitcode-skills/pr_templates/
      2. Bundled templates: <skill_dir>/templates/

    Returns:
        (mapping, default_path)
        - mapping: dict of lowercase project name → absolute template file path
        - default_path: path of the template marked default: true, or None
    """
    mapping = {}
    default_path = None

    # User templates first — they take precedence over bundled ones
    mapping, default_path = _scan_template_dir(_USER_TEMPLATE_DIR, mapping, default_path)
    mapping, default_path = _scan_template_dir(_BUNDLED_TEMPLATE_DIR, mapping, default_path)

    return mapping, default_path


def get_template_for_project(repo_name: str) -> Tuple[str, bool]:
    """Get the PR template file path for a project via auto-discovery.

    Scans templates/ directory, reads YAML frontmatter from each .md file,
    and matches by project name.

    Matching order:
    1. Exact match on repo name against frontmatter 'projects' lists
    2. Partial match (e.g., "mindspore-akg" contains "akg")
    3. Template marked as default: true

    Args:
        repo_name: The repository name (from upstream remote URL).

    Returns:
        Tuple of (template_file_path, is_configured)
        - template_file_path: Absolute path to the template .md file
        - is_configured: True if project has a specific template, False if using default
    """
    mapping, default_path = _discover_templates()
    repo_lower = repo_name.lower()

    # Exact match
    if repo_lower in mapping:
        return mapping[repo_lower], True

    # Token-boundary partial match (e.g., "mindspore-akg" matches "akg")
    for project_name, path in mapping.items():
        if _project_matches_repo_name(project_name, repo_lower):
            return path, True

    # Default
    if default_path:
        return default_path, False

    # Absolute fallback — first template found, or error
    if mapping:
        return next(iter(mapping.values())), False

    print("[ERROR] No template files found in templates/ directory.", file=sys.stderr)
    sys.exit(1)


# ============================================================================
# Git Helpers
# ============================================================================

def run_git_command(cmd: list, cwd: Optional[str] = None, error_context: str = "") -> str:
    """Run a git command and return stdout.

    Args:
        cmd: The git command and arguments.
        cwd: Working directory for the command.
        error_context: Optional context string for error messages (e.g., "fetching upstream branch").

    Returns:
        Command stdout stripped of whitespace, or empty string on failure.
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        context = f" ({error_context})" if error_context else ""
        print(f"[ERROR] Git command failed{context}: {' '.join(cmd)}", file=sys.stderr)
        stderr_lower = (e.stderr or "").lower()
        if e.stderr and e.stderr.strip():
            print(f"  Details: {e.stderr.strip()}", file=sys.stderr)
        if "could not resolve" in stderr_lower or "unable to access" in stderr_lower:
            print("  Hint: Check your network connection.", file=sys.stderr)
        elif "not a git repository" in stderr_lower:
            print("  Hint: Run this command from within a git repository.", file=sys.stderr)
        elif "unknown revision" in stderr_lower or "bad revision" in stderr_lower:
            print(f"  Hint: The specified branch or revision may not exist. Try: git fetch upstream", file=sys.stderr)
        return ""
    except FileNotFoundError:
        print("[ERROR] 'git' command not found. Ensure git is installed and in PATH.", file=sys.stderr)
        return ""


def get_upstream_info() -> Tuple[Optional[str], Optional[str]]:
    """Get upstream repository owner and name from git remote configuration.

    Extracts both owner and repo name from upstream remote URL.
    Supports both HTTPS and SSH URL formats.

    Returns:
        A tuple of (owner, repo_name), or (None, None) if detection fails.
    """
    upstream_url = run_git_command(["git", "remote", "get-url", "upstream"])
    if not upstream_url:
        return None, None

    # Matches: https://gitcode.com/owner/repo.git or git@gitcode.com:owner/repo.git
    match = re.search(r'[:/]([^/]+)/([^/]+?)(?:\.git)?$', upstream_url)
    if match:
        return match.group(1), match.group(2)

    return None, None


def get_current_branch() -> Optional[str]:
    """Get current git branch name."""
    return run_git_command(["git", "branch", "--show-current"])


def ensure_upstream_branch_fetched(base_branch: str) -> bool:
    """Ensure the upstream branch exists locally, fetch if not.

    Args:
        base_branch: The target branch name (e.g., master, r2.8)

    Returns:
        True if the branch is available, False otherwise
    """
    upstream_ref = f"upstream/{base_branch}"

    # Check if the upstream branch ref exists locally
    result = run_git_command(
        ["git", "rev-parse", "--verify", upstream_ref],
        error_context=f"checking if {upstream_ref} exists locally"
    )

    if result:
        return True

    # Branch ref doesn't exist, need to fetch it
    print(f"Fetching upstream/{base_branch}...")
    run_git_command(
        ["git", "fetch", "upstream", base_branch],
        error_context=f"fetching {upstream_ref} from remote"
    )

    # Verify again after fetch
    result = run_git_command(
        ["git", "rev-parse", "--verify", upstream_ref],
        error_context=f"verifying {upstream_ref} after fetch"
    )

    if not result:
        print(f"[ERROR] Branch upstream/{base_branch} is not available after fetch.", file=sys.stderr)
        print(f"  Possible causes:", file=sys.stderr)
        print(f"    - Branch '{base_branch}' does not exist in the upstream repository", file=sys.stderr)
        print(f"    - Network connectivity issue during fetch", file=sys.stderr)
        print(f"    - Upstream remote URL is incorrect", file=sys.stderr)
        print(f"  To see available upstream branches:", file=sys.stderr)
        print(f"    git branch -r --list 'upstream/*'", file=sys.stderr)
        return False

    return True


def get_pr_title(base_branch: str) -> Optional[str]:
    """Get the PR title from the earliest commit subject on the current branch.

    Uses --format=%s to get only the subject line of the first commit,
    avoiding issues with multi-commit branches where full bodies would
    be concatenated together.
    """
    # Get the first commit's subject line (chronological order)
    title = run_git_command([
        "git", "log", f"upstream/{base_branch}..HEAD",
        "--reverse", "--format=%s", "--max-count=1"
    ])

    if not title:
        # Fallback to latest commit if no divergence from base
        title = run_git_command(["git", "log", "-1", "--format=%s"])

    return title if title else None


# ============================================================================
# API
# ============================================================================

def create_gitcode_pr(title: str, head: str, base: str, body: str, repo: str, upstream_owner: str, access_token: str) -> Optional[dict]:
    """Create a pull request on GitCode.

    Uses access_token as URL parameter per GitCode API specification.
    Reference: https://docs.gitcode.com/en/docs/apis/post-api-v-5-repos-owner-repo-pulls

    Args:
        title: PR title
        head: Source branch in format "user_id:branch_name"
        base: Target branch name in upstream repository
        body: PR description body
        repo: Repository name
        upstream_owner: Owner of the upstream repository where PR will be created
        access_token: GitCode API access token
    """
    import urllib.request
    import urllib.error

    # Create PR in the upstream repository (not the fork)
    url = f"{API_BASE}/repos/{upstream_owner}/{repo}/pulls?access_token={access_token}"

    payload = {
        "title": title,
        "head": head,
        "base": base,
        "body": body
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        with urllib.request.urlopen(req) as response:
            response_data = json.loads(response.read().decode("utf-8"))
            return response_data

    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}", file=sys.stderr)
        try:
            error_body = json.loads(e.read().decode("utf-8"))
            print(f"Error details: {error_body}", file=sys.stderr)
        except Exception:
            pass
        return None
    except Exception as e:
        print(f"Error creating PR: {e}", file=sys.stderr)
        return None


# ============================================================================
# Main
# ============================================================================

def main():
    """Main function to create PR."""
    import argparse

    parser = argparse.ArgumentParser(description='Create a GitCode pull request')
    parser.add_argument('--base', type=str, default='',
                       help='Target branch for the PR (e.g., master, main, br_mfusion).')
    parser.add_argument('--body-file', type=str, default='',
                       help='Path to a file containing the complete PR body (markdown).')
    parser.add_argument('--body-stdin', action='store_true',
                       help='Read the complete PR body from stdin instead of a file.')
    parser.add_argument('--template-status', action='store_true',
                       help='Print template file path for the current project and exit.')
    args = parser.parse_args()

    # Get upstream info (owner and repo name) from git remote URL
    upstream_owner, repo = get_upstream_info()
    if not upstream_owner or not repo:
        print("\n[ERROR] Failed to detect upstream repository info.", file=sys.stderr)
        print("\nThis script requires an 'upstream' remote to create PRs in the upstream repository.", file=sys.stderr)
        print("\nPlease ensure:", file=sys.stderr)
        print("  1. You have an 'upstream' remote configured", file=sys.stderr)
        print("  2. Run: git remote add upstream <upstream_repo_url>", file=sys.stderr)
        print("  3. Run: git fetch upstream", file=sys.stderr)
        print("\nExample:", file=sys.stderr)
        print("  git remote add upstream https://gitcode.com/mindspore/akg.git", file=sys.stderr)
        print("  git fetch upstream", file=sys.stderr)
        sys.exit(1)

    # Handle --template-status flag
    if args.template_status:
        template_path, is_configured = get_template_for_project(repo)
        if is_configured:
            print(f"[CONFIGURED] Project '{repo}' has a configured PR template.")
        else:
            print(f"[DEFAULT] Project '{repo}' is using the default template.")
        # Show which directory the template comes from
        norm_path = os.path.normpath(template_path)
        if norm_path.startswith(os.path.normpath(_USER_TEMPLATE_DIR)):
            print(f"  Source: user ({_USER_TEMPLATE_DIR})")
        else:
            print(f"  Source: bundled ({_BUNDLED_TEMPLATE_DIR})")
        print(f"  Template file: {template_path}")
        sys.exit(0)

    # Validate required parameters
    if not args.base or not args.base.strip():
        print("\n[ERROR] Target branch not specified.", file=sys.stderr)
        print("\nUsage: python create_pr.py --base <branch> --body-file <path>", file=sys.stderr)
        print("\nTo see available branches:", file=sys.stderr)
        print("  git branch -r --list 'upstream/*'", file=sys.stderr)
        sys.exit(2)

    if args.body_stdin and args.body_file.strip():
        print("\n[ERROR] Use either --body-file or --body-stdin, not both.", file=sys.stderr)
        sys.exit(3)

    if not args.body_stdin and (not args.body_file or not args.body_file.strip()):
        print("\n[ERROR] Either --body-file or --body-stdin is required.", file=sys.stderr)
        print("\nProvide a file containing the complete PR description body, or pipe it via stdin.", file=sys.stderr)
        sys.exit(3)

    # Read body from stdin or file
    if args.body_stdin:
        body = sys.stdin.read()
    else:
        body_path = args.body_file.strip()
        if not os.path.isfile(body_path):
            print(f"\n[ERROR] Body file not found: {body_path}", file=sys.stderr)
            sys.exit(3)

        try:
            with open(body_path, encoding="utf-8") as f:
                body = f.read()
        except OSError as e:
            print(f"\n[ERROR] Failed to read body file: {e}", file=sys.stderr)
            sys.exit(3)

    # Post-process: strip leaked frontmatter
    body = sanitize_pr_body(body)

    if not body.strip():
        print("\n[ERROR] Body file is empty after sanitization.", file=sys.stderr)
        sys.exit(3)

    config = load_config()

    print("GitCode PR Creator")
    print("=" * 50)

    print(f"Repository: {upstream_owner}/{repo}")
    print(f"User ID: {config['user_id']}")

    # Get branch information
    current_branch = get_current_branch()
    if not current_branch:
        print("Error: Could not determine current branch.", file=sys.stderr)
        sys.exit(1)

    print(f"Current branch: {current_branch}")

    base_branch = args.base
    print(f"Base branch: {base_branch}")

    # Ensure upstream branch is fetched
    if not ensure_upstream_branch_fetched(base_branch):
        print(f"Error: Could not fetch upstream/{base_branch}", file=sys.stderr)
        sys.exit(1)

    # Get PR title from the earliest commit on this branch
    title = get_pr_title(base_branch)
    if not title:
        print("Error: Could not get commit message for PR title.", file=sys.stderr)
        sys.exit(1)

    print(f"PR title: {title}")

    # For GitCode API, head should be in "user_id:branch" format
    head_branch = f"{config['user_id']}:{current_branch}"

    print("\n" + "=" * 50)
    print("Creating PR...")
    print(f"  Target repository: {upstream_owner}/{repo}")
    print(f"  From: {head_branch}")
    print(f"  To: {base_branch}")
    print("=" * 50 + "\n")

    # Create PR in upstream repository
    result = create_gitcode_pr(title, head_branch, base_branch, body, repo, upstream_owner, config["access_token"])

    if result:
        pr_url = build_pr_web_url(result, upstream_owner, repo)
        print("[OK] PR created successfully!")
        print(f"  URL: {pr_url}")
        print(f"  Number: {result.get('number', 'N/A')}")
        print(f"  State: {result.get('state', 'N/A')}")
        return 0
    else:
        print("[ERROR] Failed to create PR.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
