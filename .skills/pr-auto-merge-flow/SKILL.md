---
name: pr-auto-merge-flow
description: Record and execute the project workflow for PR-based delivery: create a branch, open a PR, run automated inspection, add an inspection comment instead of self-approval, then merge the PR when inspection passes.
---

# PR Auto Merge Flow

Use this skill when a change should be delivered through a Pull Request and the repository is operated by a single GitHub account that cannot approve its own PR.

## Purpose

This project uses Pull Requests instead of direct pushes to `main`. When automated inspection passes, do not attempt to approve the PR with the same GitHub account. Instead, add a PR comment documenting the inspection result, then merge the PR.

## Workflow

1. Sync local `main` with `origin/main` when network access is available.
2. Create a focused branch for the change.
3. Make the minimal scoped change.
4. Verify the local diff and status.
5. Commit with the Relay Agent co-author trailer:

   ```text
   Co-Authored-By: RelayAgent <noreply@relayagent.local>
   ```

6. Push the branch to `origin`.
7. Create a Pull Request targeting `main`.
8. Run automated read-only inspection of the PR diff.
9. If inspection finds issues:
   - fix them on the same branch;
   - push the update;
   - inspect again.
10. If inspection passes:
    - add a PR comment that records the automated inspection result;
    - merge the PR;
    - sync local `main` after merge.

## PR Comment Template

```markdown
Automated inspection completed.

Result: passed.

Checks performed:
- Confirmed the PR diff is limited to the intended scope.
- Confirmed generated or local-only artifacts are not submitted.
- Confirmed there are no blocking review findings.

This comment replaces self-approval for single-account repositories. Proceeding to merge.
```

## Important Rules

- Do not push directly to `main` for normal changes.
- Do not force-push unless explicitly requested and confirmed.
- Do not attempt to approve a PR authored by the same GitHub account; GitHub may reject self-approval through the API.
- Prefer a PR comment documenting automated inspection when only one account is available.
- Keep each PR focused and easy to review.

## Validation Commands

Use commands appropriate to the change. Typical checks include:

```bash
git status --short --ignored
git diff --stat main..HEAD
git diff main..HEAD
```

For GitHub PR operations with GitHub CLI installed in the default Windows path:

```bash
"/c/Program Files/GitHub CLI/gh.exe" pr create --repo OWNER/REPO --base main --head BRANCH --title "TITLE" --body-file BODY.md
"/c/Program Files/GitHub CLI/gh.exe" pr comment PR_NUMBER --repo OWNER/REPO --body-file COMMENT.md
"/c/Program Files/GitHub CLI/gh.exe" pr merge PR_NUMBER --repo OWNER/REPO --merge --delete-branch
```
