# GitCode PR API Reference

Complete GitCode Pull Request API documentation for creating PRs.

## API Endpoint

```
POST https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls?access_token={token}
```

**Important**: `access_token` must be passed as a URL parameter, not in the Authorization header.

Reference: https://docs.gitcode.com/en/docs/apis/post-api-v-5-repos-owner-repo-pulls

## Required Credentials

The API call needs:

- **User ID**: your GitCode user ID
- **Access Token**: your GitCode access token
- **Repo**: auto-detected from the current git repository

Credentials are not hardcoded in the skill. They are loaded from the shared gitcode-skills
config file or environment variables at runtime. If credentials are missing, use
[CREDENTIALS_SETUP.md](CREDENTIALS_SETUP.md) for the agent-side safety workflow, and point the user to
[../README.md](../README.md) for local credential setup guidance.

## Authentication

GitCode API uses `access_token` URL parameter for authentication:

```bash
# Correct - access_token as URL parameter
POST https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls?access_token=YOUR_TOKEN

# Incorrect - Authorization header is NOT supported by GitCode
Authorization: token YOUR_TOKEN  # ❌ This will fail with 404
```

Example with curl:

```bash
curl --location --request POST 'https://api.gitcode.com/api/v5/repos/xiaogang_test/test222/pulls?access_token=xxxx' \
--header 'Content-Type: application/json' \
--data-raw '{
    "title": "title",
    "head": "dev",
    "base": "main"
}'
```

## Request Headers

```http
Content-Type: application/json
```

**Note**: Do NOT include `Authorization` header - GitCode uses `access_token` URL parameter instead.

## Request Body Parameters

| Parameter | Type    | Required | Description                      |
|-----------|---------|----------|----------------------------------|
| title     | string  | Yes      | PR title (from commit message)   |
| head      | string  | Yes      | Source branch: `{user_id}:<branch>` |
| base      | string  | Yes      | Target branch (from upstream)    |
| body      | string  | Yes      | PR description in markdown       |

## Example Request

```json
POST https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls?access_token=YOUR_ACCESS_TOKEN
Content-Type: application/json

{
  "title": "refactor(module-x): unify attribute naming",
  "head": "{user_id}:feat-my-branch",
  "base": "br_mfusion",
  "body": "**What type of PR is this?**\n> `/kind feature`</br>\n\n**What does this PR do**:\nUnify attribute naming conventions..."
}
```

## Response

Success (200 OK):

```json
{
  "id": 12345,
  "number": 42,
  "state": "open",
  "title": "PR title",
  "body": "PR description",
  "html_url": "https://gitcode.com/{owner}/{repo}/pulls/42",
  "head": {
    "ref": "feature-branch",
    "repo": {
      "full_name": "{user_id}/{repo}"
    }
  },
  "base": {
    "ref": "br_mfusion",
    "repo": {
      "full_name": "mindspore/akg"
    }
  }
}
```

## Error Responses

### 404 Not Found

Common causes:
- **Invalid token**: Token passed as Authorization header instead of URL parameter
- **Wrong repository**: Repository doesn't exist or user doesn't have access
- **Malformed URL**: Incorrect API endpoint format

Solution: Ensure `access_token` is a URL parameter: `?access_token=YOUR_TOKEN`

### 401 Unauthorized

Access token is invalid, expired, or missing from the user's local credential setup
(shared config file or environment variables).

Solution:
1. Ask the user to re-check their locally configured `user_id` and `access_token`
2. Point the user to [../README.md](../README.md) for the local-only credential setup guidance
3. Keep the reply limited to minimal setup reminders and avoid asking the user to paste any secrets

### 422 Unprocessable Entity

Validation error (missing required fields, invalid branch names, etc.).

For PR description template details, see [IMPLEMENTATION.md](IMPLEMENTATION.md).
