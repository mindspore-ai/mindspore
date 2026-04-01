"""Local unit tests for the ms-common-pr-creator script."""

# pylint: disable=missing-docstring,bad-continuation

import importlib.util
import io
import json
import os
import tempfile
import textwrap
import unittest
from unittest import mock
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = SKILL_DIR / "scripts" / "create_pr.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "gitcode_pr_create_pr", MODULE_PATH
)
create_pr = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(create_pr)


class CreatePrLocalTests(unittest.TestCase):  # pylint: disable=missing-docstring
    """Hermetic unit tests for create_pr.py behavior and edge cases."""

    def run_main(self, argv, stdin_text=None, stdin_bytes=None):
        """Run create_pr.main() with patched argv/stdin and capture outputs."""
        stdout = io.StringIO()
        stderr = io.StringIO()
        if stdin_bytes is None:
            stdin_bytes = ("" if stdin_text is None else stdin_text).encode("utf-8")
        stdin = io.TextIOWrapper(io.BytesIO(stdin_bytes), encoding="utf-8")

        with mock.patch("sys.argv", ["create_pr.py", *argv]):
            with mock.patch("sys.stdin", stdin):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    try:
                        result = create_pr.main()
                    except SystemExit as exc:
                        code = exc.code
                    else:
                        code = result if isinstance(result, int) else 0

        return code, stdout.getvalue(), stderr.getvalue()

    def write_template(self, directory, filename, content):
        """Write a template file into a temporary template directory."""
        path = Path(directory) / filename
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        return str(path)

    def test_load_config_prefers_environment_over_file(self):
        with tempfile.TemporaryDirectory() as config_dir, mock.patch.dict(
            os.environ,
            {"GITCODE_USER_ID": "env-user", "GITCODE_ACCESS_TOKEN": "env-token"},
            clear=False,
        ), mock.patch.object(create_pr, "_CONFIG_DIR", config_dir), mock.patch.object(
            create_pr, "_CONFIG_FILE", os.path.join(config_dir, "config.json")
        ):
            config_file = getattr(create_pr, "_CONFIG_FILE")
            Path(config_file).write_text(
                json.dumps({"user_id": "file-user", "access_token": "file-token"}),
                encoding="utf-8",
            )

            config = create_pr.load_config()

        self.assertEqual(config, {"user_id": "env-user", "access_token": "env-token"})

    def test_load_config_reads_file_when_environment_missing(self):
        with tempfile.TemporaryDirectory() as config_dir, mock.patch.dict(
            os.environ, {"GITCODE_USER_ID": "", "GITCODE_ACCESS_TOKEN": ""}, clear=False
        ), mock.patch.object(create_pr, "_CONFIG_DIR", config_dir), mock.patch.object(
            create_pr, "_CONFIG_FILE", os.path.join(config_dir, "config.json")
        ):
            config_file = getattr(create_pr, "_CONFIG_FILE")
            Path(config_file).write_text(
                json.dumps({"user_id": "file-user", "access_token": "file-token"}),
                encoding="utf-8",
            )

            config = create_pr.load_config()

        self.assertEqual(config, {"user_id": "file-user", "access_token": "file-token"})

    def test_load_config_exits_with_setup_error_when_missing(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with tempfile.TemporaryDirectory() as config_dir, mock.patch.dict(
            os.environ, {"GITCODE_USER_ID": "", "GITCODE_ACCESS_TOKEN": ""}, clear=False
        ), mock.patch.object(create_pr, "_CONFIG_DIR", config_dir), mock.patch.object(
            create_pr, "_CONFIG_FILE", os.path.join(config_dir, "config.json")
        ), redirect_stdout(
            stdout
        ), redirect_stderr(
            stderr
        ):
            with self.assertRaises(SystemExit) as ctx:
                create_pr.load_config()

        self.assertEqual(ctx.exception.code, 4)
        self.assertIn('"error_code": "AUTH_NOT_CONFIGURED"', stdout.getvalue())
        self.assertIn("GitCode credentials not configured", stderr.getvalue())

    def test_sanitize_pr_body_strips_leading_frontmatter(self):
        body = textwrap.dedent(
            """\
            ---
            projects:
              - mindspore
            default: true
            ---

            **What does this PR do / why do we need it**:
            修复了前置 metadata 泄漏问题。
            """
        )

        sanitized = create_pr.sanitize_pr_body(body)

        self.assertNotIn("projects:", sanitized)
        self.assertTrue(
            sanitized.startswith("**What does this PR do / why do we need it**:")
        )

    def test_project_match_uses_token_boundaries(self):
        project_matches_repo_name = getattr(create_pr, "_project_matches_repo_name")
        self.assertTrue(project_matches_repo_name("akg", "mindspore-akg"))
        self.assertTrue(project_matches_repo_name("mindspore", "mindspore-docs"))
        self.assertFalse(project_matches_repo_name("mind", "mindspore"))
        self.assertFalse(project_matches_repo_name("doc", "docs"))

    def test_user_template_takes_precedence_over_bundled(self):
        with tempfile.TemporaryDirectory() as bundled_dir, tempfile.TemporaryDirectory() as user_dir:
            bundled_path = self.write_template(
                bundled_dir,
                "akg.md",
                """\
                ---
                projects:
                  - akg
                ---
                bundled template
                """,
            )
            user_path = self.write_template(
                user_dir,
                "akg.md",
                """\
                ---
                projects:
                  - akg
                ---
                user template
                """,
            )

            with mock.patch.object(
                create_pr, "_BUNDLED_TEMPLATE_DIR", bundled_dir
            ), mock.patch.object(create_pr, "_USER_TEMPLATE_DIR", user_dir):
                template_path, is_configured = create_pr.get_template_for_project("akg")

            self.assertEqual(template_path, user_path)
            self.assertTrue(is_configured)
            self.assertNotEqual(template_path, bundled_path)

    def test_main_template_status_reports_docs_template_mapping(self):
        with tempfile.TemporaryDirectory() as bundled_dir, tempfile.TemporaryDirectory() as user_dir:
            mindspore_path = self.write_template(
                bundled_dir,
                "mindspore.md",
                """\
                ---
                projects:
                  - mindspore
                  - docs
                default: true
                ---
                bundled mindspore template
                """,
            )

            with mock.patch.object(
                create_pr, "_BUNDLED_TEMPLATE_DIR", bundled_dir
            ), mock.patch.object(
                create_pr, "_USER_TEMPLATE_DIR", user_dir
            ), mock.patch.object(
                create_pr, "get_upstream_info", return_value=("mindspore", "docs")
            ):
                code, stdout, stderr = self.run_main(["--template-status"])

            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            self.assertIn(
                "[CONFIGURED] Project 'docs' has a configured PR template.", stdout
            )
            self.assertIn(mindspore_path, stdout)

    def test_main_exits_when_upstream_cannot_be_detected(self):
        with mock.patch.object(
            create_pr, "get_upstream_info", return_value=(None, None)
        ):
            code, stdout, stderr = self.run_main(
                ["--base", "master", "--body-file", "ignored.md"]
            )

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("Failed to detect upstream repository info", stderr)
        self.assertIn("git remote add upstream", stderr)

    def test_main_requires_base_branch(self):
        with mock.patch.object(
            create_pr, "get_upstream_info", return_value=("mindspore", "akg")
        ):
            code, stdout, stderr = self.run_main([])

        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("Target branch not specified", stderr)
        self.assertIn("--base <branch>", stderr)

    def test_main_requires_body_source(self):
        with mock.patch.object(
            create_pr, "get_upstream_info", return_value=("mindspore", "akg")
        ):
            code, stdout, stderr = self.run_main(["--base", "master"])

        self.assertEqual(code, 3)
        self.assertEqual(stdout, "")
        self.assertIn("Either --body-file or --body-stdin is required", stderr)

    def test_main_rejects_body_file_and_body_stdin_together(self):
        with tempfile.NamedTemporaryFile(
            "w", delete=False, encoding="utf-8", suffix=".md"
        ) as body_file:
            body_file.write("这是一个有效的 PR 正文。\n")
            body_path = body_file.name

        try:
            with mock.patch.object(
                create_pr, "get_upstream_info", return_value=("mindspore", "akg")
            ):
                code, stdout, stderr = self.run_main(
                    ["--base", "master", "--body-file", body_path, "--body-stdin"],
                    stdin_text="stdin body",
                )
        finally:
            if os.path.exists(body_path):
                os.unlink(body_path)

        self.assertEqual(code, 3)
        self.assertEqual(stdout, "")
        self.assertIn("Use either --body-file or --body-stdin, not both", stderr)

    def test_main_rejects_empty_body_after_sanitization(self):
        with tempfile.NamedTemporaryFile(
            "w", delete=False, encoding="utf-8", suffix=".md"
        ) as body_file:
            body_file.write(
                textwrap.dedent(
                    """\
                    ---
                    projects:
                      - akg
                    ---
                    """
                )
            )
            body_path = body_file.name

        try:
            with mock.patch.object(
                create_pr, "get_upstream_info", return_value=("mindspore", "akg")
            ):
                code, stdout, stderr = self.run_main(
                    ["--base", "master", "--body-file", body_path]
                )
        finally:
            if os.path.exists(body_path):
                os.unlink(body_path)

        self.assertEqual(code, 3)
        self.assertEqual(stdout, "")
        self.assertIn("Body file is empty after sanitization", stderr)

    def test_main_happy_path_creates_pr_with_expected_arguments(self):
        with tempfile.NamedTemporaryFile(
            "w", delete=False, encoding="utf-8", suffix=".md"
        ) as body_file:
            body_file.write(
                textwrap.dedent(
                    """\
                    ---
                    projects:
                      - akg
                    ---

                    这是清理后的 PR 正文。
                    """
                )
            )
            body_path = body_file.name

        try:
            with mock.patch.object(
                create_pr, "get_upstream_info", return_value=("mindspore", "akg")
            ), mock.patch.object(
                create_pr,
                "load_config",
                return_value={"user_id": "alice", "access_token": "secret-token"},
            ), mock.patch.object(
                create_pr, "get_current_branch", return_value="feature-branch"
            ), mock.patch.object(
                create_pr, "ensure_upstream_branch_fetched", return_value=True
            ), mock.patch.object(
                create_pr, "get_pr_title", return_value="feat(runtime): add validator"
            ), mock.patch.object(
                create_pr,
                "create_gitcode_pr",
                return_value={
                    "html_url": "https://gitcode.com/mindspore/akg/pulls/123",
                    "number": 123,
                    "state": "open",
                },
            ) as create_mock:
                code, stdout, stderr = self.run_main(
                    ["--base", "master", "--body-file", body_path]
                )
        finally:
            if os.path.exists(body_path):
                os.unlink(body_path)

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        create_mock.assert_called_once_with(
            "feat(runtime): add validator",
            "alice:feature-branch",
            "master",
            "这是清理后的 PR 正文。\n",
            "akg",
            "mindspore",
            "secret-token",
        )
        self.assertIn("GitCode PR Creator", stdout)
        self.assertIn("Repository: mindspore/akg", stdout)
        self.assertIn("Current branch: feature-branch", stdout)
        self.assertIn("[OK] PR created successfully!", stdout)
        self.assertIn("URL: https://gitcode.com/mindspore/akg/pull/123", stdout)
        self.assertIn("Number: 123", stdout)

    def test_main_reads_body_from_stdin(self):
        body = textwrap.dedent(
            """\
            ---
            projects:
              - akg
            ---

            这是从 stdin 读取的 PR 正文。
            """
        )

        with mock.patch.object(
            create_pr, "get_upstream_info", return_value=("mindspore", "akg")
        ), mock.patch.object(
            create_pr,
            "load_config",
            return_value={"user_id": "alice", "access_token": "secret-token"},
        ), mock.patch.object(
            create_pr, "get_current_branch", return_value="feature-branch"
        ), mock.patch.object(
            create_pr, "ensure_upstream_branch_fetched", return_value=True
        ), mock.patch.object(
            create_pr, "get_pr_title", return_value="feat(runtime): add validator"
        ), mock.patch.object(
            create_pr,
            "create_gitcode_pr",
            return_value={
                "html_url": "https://gitcode.com/mindspore/akg/pulls/123",
                "number": 123,
                "state": "open",
            },
        ) as create_mock:
            code, stdout, stderr = self.run_main(
                ["--base", "master", "--body-stdin"], stdin_text=body
            )

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        create_mock.assert_called_once_with(
            "feat(runtime): add validator",
            "alice:feature-branch",
            "master",
            "这是从 stdin 读取的 PR 正文。\n",
            "akg",
            "mindspore",
            "secret-token",
        )
        self.assertIn("URL: https://gitcode.com/mindspore/akg/pull/123", stdout)

    def test_main_rejects_non_utf8_stdin_body(self):
        with mock.patch.object(
            create_pr, "get_upstream_info", return_value=("mindspore", "akg")
        ):
            code, stdout, stderr = self.run_main(
                ["--base", "master", "--body-stdin"], stdin_bytes=b"\xff\xfe\xfd"
            )

        self.assertEqual(code, 3)
        self.assertEqual(stdout, "")
        self.assertIn("PR body from stdin must be UTF-8 encoded", stderr)
        self.assertIn("Use --body-file with a UTF-8 temporary file", stderr)

    def test_main_fails_when_current_branch_cannot_be_determined(self):
        with tempfile.NamedTemporaryFile(
            "w", delete=False, encoding="utf-8", suffix=".md"
        ) as body_file:
            body_file.write("这是一个有效的 PR 正文。\n")
            body_path = body_file.name

        try:
            with mock.patch.object(
                create_pr, "get_upstream_info", return_value=("mindspore", "akg")
            ), mock.patch.object(
                create_pr,
                "load_config",
                return_value={"user_id": "alice", "access_token": "secret-token"},
            ), mock.patch.object(
                create_pr, "get_current_branch", return_value=""
            ), mock.patch.object(
                create_pr, "create_gitcode_pr"
            ) as create_mock:
                code, stdout, stderr = self.run_main(
                    ["--base", "master", "--body-file", body_path]
                )
        finally:
            if os.path.exists(body_path):
                os.unlink(body_path)

        self.assertEqual(code, 1)
        self.assertIn("GitCode PR Creator", stdout)
        self.assertIn("Repository: mindspore/akg", stdout)
        self.assertIn("Could not determine current branch", stderr)
        create_mock.assert_not_called()

    def test_main_fails_when_upstream_branch_cannot_be_fetched(self):
        with tempfile.NamedTemporaryFile(
            "w", delete=False, encoding="utf-8", suffix=".md"
        ) as body_file:
            body_file.write("这是一个有效的 PR 正文。\n")
            body_path = body_file.name

        try:
            with mock.patch.object(
                create_pr, "get_upstream_info", return_value=("mindspore", "akg")
            ), mock.patch.object(
                create_pr,
                "load_config",
                return_value={"user_id": "alice", "access_token": "secret-token"},
            ), mock.patch.object(
                create_pr, "get_current_branch", return_value="feature-branch"
            ), mock.patch.object(
                create_pr, "ensure_upstream_branch_fetched", return_value=False
            ), mock.patch.object(
                create_pr, "get_pr_title"
            ) as title_mock, mock.patch.object(
                create_pr, "create_gitcode_pr"
            ) as create_mock:
                code, stdout, stderr = self.run_main(
                    ["--base", "master", "--body-file", body_path]
                )
        finally:
            if os.path.exists(body_path):
                os.unlink(body_path)

        self.assertEqual(code, 1)
        self.assertIn("Current branch: feature-branch", stdout)
        self.assertIn("Base branch: master", stdout)
        self.assertIn("Could not fetch upstream/master", stderr)
        title_mock.assert_not_called()
        create_mock.assert_not_called()

    def test_main_fails_when_pr_title_cannot_be_resolved(self):
        with tempfile.NamedTemporaryFile(
            "w", delete=False, encoding="utf-8", suffix=".md"
        ) as body_file:
            body_file.write("这是一个有效的 PR 正文。\n")
            body_path = body_file.name

        try:
            with mock.patch.object(
                create_pr, "get_upstream_info", return_value=("mindspore", "akg")
            ), mock.patch.object(
                create_pr,
                "load_config",
                return_value={"user_id": "alice", "access_token": "secret-token"},
            ), mock.patch.object(
                create_pr, "get_current_branch", return_value="feature-branch"
            ), mock.patch.object(
                create_pr, "ensure_upstream_branch_fetched", return_value=True
            ), mock.patch.object(
                create_pr, "get_pr_title", return_value=""
            ), mock.patch.object(
                create_pr, "create_gitcode_pr"
            ) as create_mock:
                code, stdout, stderr = self.run_main(
                    ["--base", "master", "--body-file", body_path]
                )
        finally:
            if os.path.exists(body_path):
                os.unlink(body_path)

        self.assertEqual(code, 1)
        self.assertIn("Current branch: feature-branch", stdout)
        self.assertIn("Base branch: master", stdout)
        self.assertIn("Could not get commit message for PR title", stderr)
        create_mock.assert_not_called()

    def test_main_fails_when_create_gitcode_pr_returns_none(self):
        with tempfile.NamedTemporaryFile(
            "w", delete=False, encoding="utf-8", suffix=".md"
        ) as body_file:
            body_file.write("这是一个有效的 PR 正文。\n")
            body_path = body_file.name

        try:
            with mock.patch.object(
                create_pr, "get_upstream_info", return_value=("mindspore", "akg")
            ), mock.patch.object(
                create_pr,
                "load_config",
                return_value={"user_id": "alice", "access_token": "secret-token"},
            ), mock.patch.object(
                create_pr, "get_current_branch", return_value="feature-branch"
            ), mock.patch.object(
                create_pr, "ensure_upstream_branch_fetched", return_value=True
            ), mock.patch.object(
                create_pr, "get_pr_title", return_value="feat(runtime): add validator"
            ), mock.patch.object(
                create_pr, "create_gitcode_pr", return_value=None
            ):
                code, stdout, stderr = self.run_main(
                    ["--base", "master", "--body-file", body_path]
                )
        finally:
            if os.path.exists(body_path):
                os.unlink(body_path)

        self.assertEqual(code, 1)
        self.assertIn("Creating PR...", stdout)
        self.assertIn("PR title: feat(runtime): add validator", stdout)
        self.assertIn("Failed to create PR", stderr)
        self.assertNotIn("[OK] PR created successfully!", stdout)

    def test_get_pr_title_prefers_earliest_divergent_commit(self):
        with mock.patch.object(
            create_pr,
            "run_git_command",
            side_effect=[
                "feat(runtime): first commit\nrefactor(runtime): second commit\nfix(runtime): latest commit"
            ],
        ):
            title = create_pr.get_pr_title("master")

        self.assertEqual(title, "feat(runtime): first commit")

    def test_get_pr_title_falls_back_to_latest_commit(self):
        with mock.patch.object(
            create_pr,
            "run_git_command",
            side_effect=["", "fix(runtime): latest commit"],
        ):
            title = create_pr.get_pr_title("master")

        self.assertEqual(title, "fix(runtime): latest commit")

    def test_ensure_upstream_branch_fetched_returns_true_when_branch_exists_locally(
        self,
    ):
        with mock.patch.object(create_pr, "run_git_command", side_effect=["deadbeef"]):
            result = create_pr.ensure_upstream_branch_fetched("master")

        self.assertTrue(result)

    def test_ensure_upstream_branch_fetched_fetches_missing_branch_successfully(self):
        with mock.patch.object(
            create_pr, "run_git_command", side_effect=["", "", "deadbeef"]
        ):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = create_pr.ensure_upstream_branch_fetched("r2.5")

        self.assertTrue(result)
        self.assertIn("Fetching upstream/r2.5...", stdout.getvalue())

    def test_ensure_upstream_branch_fetched_returns_false_after_failed_fetch(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with mock.patch.object(
            create_pr, "run_git_command", side_effect=["", "", ""]
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            result = create_pr.ensure_upstream_branch_fetched("does-not-exist")

        self.assertFalse(result)
        self.assertIn("Fetching upstream/does-not-exist...", stdout.getvalue())
        self.assertIn(
            "Branch upstream/does-not-exist is not available after fetch",
            stderr.getvalue(),
        )

    def test_create_gitcode_pr_sends_expected_payload(self):
        captured = {}

        class FakeResponse:  # pylint: disable=missing-docstring
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {"html_url": "https://gitcode.com/pr/1", "number": 1}
                ).encode("utf-8")

        def fake_urlopen(req):
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            captured["body"] = json.loads(req.data.decode("utf-8"))
            captured["content_type"] = req.headers.get("Content-type")
            return FakeResponse()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = create_pr.create_gitcode_pr(
                title="feat(runtime): add validator",
                head="alice:feature-branch",
                base="master",
                body="PR body",
                repo="akg",
                upstream_owner="mindspore",
                access_token="secret-token",
            )

        self.assertEqual(result["number"], 1)
        self.assertEqual(
            captured["url"],
            "https://api.gitcode.com/api/v5/repos/mindspore/akg/pulls?access_token=secret-token",
        )
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(
            captured["body"],
            {
                "title": "feat(runtime): add validator",
                "head": "alice:feature-branch",
                "base": "master",
                "body": "PR body",
            },
        )
        self.assertEqual(captured["content_type"], "application/json")

    def test_create_gitcode_pr_returns_none_on_http_error(self):
        import urllib.error

        stdout = io.StringIO()
        stderr = io.StringIO()
        error = urllib.error.HTTPError(
            url="https://api.gitcode.com/api/v5/repos/mindspore/akg/pulls?access_token=bad-token",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(json.dumps({"message": "Bad credentials"}).encode("utf-8")),
        )

        with mock.patch("urllib.request.urlopen", side_effect=error), redirect_stdout(
            stdout
        ), redirect_stderr(stderr):
            result = create_pr.create_gitcode_pr(
                title="feat(runtime): add validator",
                head="alice:feature-branch",
                base="master",
                body="PR body",
                repo="akg",
                upstream_owner="mindspore",
                access_token="bad-token",
            )

        self.assertIsNone(result)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("HTTP Error 401: Unauthorized", stderr.getvalue())
        self.assertIn("Bad credentials", stderr.getvalue())

    def test_build_pr_web_url_prefers_canonical_pull_path_from_number(self):
        url = create_pr.build_pr_web_url(
            {"html_url": "https://gitcode.com/mindspore/akg/pulls/123", "number": 123},
            "mindspore",
            "akg",
        )

        self.assertEqual(url, "https://gitcode.com/mindspore/akg/pull/123")

    def test_build_pr_web_url_normalizes_api_html_url_when_number_missing(self):
        url = create_pr.build_pr_web_url(
            {"html_url": "https://gitcode.com/mindspore/akg/pulls/123"},
            "mindspore",
            "akg",
        )

        self.assertEqual(url, "https://gitcode.com/mindspore/akg/pull/123")


if __name__ == "__main__":
    unittest.main()
