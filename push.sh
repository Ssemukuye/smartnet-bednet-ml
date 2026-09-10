#!/usr/bin/env bash
#
# Stage, verify and commit the project — refusing to proceed if anything that
# must not be published has been staged.
#
#   bash push.sh            stage, check, commit  (does NOT push)
#   bash push.sh --push     the above, then push to origin/main
#
# The checks are the point. Run this instead of git add/commit by hand so the
# study data and model binaries cannot be committed by accident.

set -euo pipefail
cd "$(dirname "$0")"

BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; OFF=$'\033[0m'
say()  { printf '%s\n' "$*"; }
ok()   { printf '%s  ✓ %s%s\n' "$GREEN" "$*" "$OFF"; }
warn() { printf '%s  ! %s%s\n' "$YELLOW" "$*" "$OFF"; }
die()  { printf '%s\n  ✗ %s%s\n' "$RED" "$*" "$OFF" >&2; exit 1; }

say ""
say "${BOLD}smartnet-bednet-ml — staged commit with safety checks${OFF}"
say "─────────────────────────────────────────────────────────"

# --- 0. stale lock -----------------------------------------------------------
if [ -f .git/index.lock ]; then
  rm -f .git/index.lock && ok "removed stale .git/index.lock"
fi

[ -d .git ] || die "No git repository here. Run: git init && git branch -M main"

# --- 1. stage ----------------------------------------------------------------
git add -A
STAGED=$(git diff --cached --name-only | wc -l | tr -d ' ')
[ "$STAGED" -gt 0 ] || { warn "nothing new to commit"; exit 0; }
ok "staged $STAGED files"

# --- 2. refuse to publish what must not be published -------------------------
say ""
say "${BOLD}Pre-flight checks${OFF}"

check_none() {  # description, regex
  local desc="$1" pattern="$2" hits
  hits=$(git diff --cached --name-only | grep -Ei "$pattern" || true)
  if [ -n "$hits" ]; then
    say ""
    say "$RED  BLOCKED — $desc$OFF"
    # One line per path; filenames contain spaces, so never word-split.
    while IFS= read -r f; do printf '      %s\n' "$f"; done <<<"$hits"
    say ""
    git reset -q
    die "Nothing committed and everything unstaged. Fix .gitignore, then re-run."
  fi
  ok "$desc"
}

check_none "no study data staged"      '\.dta$|^data/raw/|^data/processed/|^data/interim/'
check_none "no model binaries staged"  '\.joblib$|\.pkl$'
check_none "no credentials staged"     '(^|/)\.env$|credentials\.json$|\.pem$'

# Belt and braces: grep staged file *contents* for an obvious secret.
if git diff --cached -U0 | grep -Eiq '^\+.*(password|passwd|secret|api[_-]?key)[[:space:]]*=[[:space:]]*["'\''][^"'\'']{4,}'; then
  say ""
  warn "a staged line looks like a hardcoded secret — review before continuing:"
  git diff --cached -U0 | grep -Ein '^\+.*(password|passwd|secret|api[_-]?key)[[:space:]]*=' | head -5
  say ""
  read -r -p "  Continue anyway? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || die "Aborted. Nothing committed."
else
  ok "no hardcoded secrets detected in staged changes"
fi

# --- 3. tests ----------------------------------------------------------------
say ""
if command -v python3 >/dev/null && [ -d tests ]; then
  say "${BOLD}Running tests${OFF}"
  if PYTHONPATH=src python3 -m pytest tests/ -q >/tmp/pytest_push.log 2>&1; then
    ok "$(tail -1 /tmp/pytest_push.log | tr -d '\n')"
  else
    tail -15 /tmp/pytest_push.log
    say ""
    read -r -p "  Tests failed. Commit anyway? [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]] || die "Aborted. Nothing committed."
  fi
fi

# --- 4. commit ---------------------------------------------------------------
say ""
MSG="${COMMIT_MSG:-Add full project: pipeline, MLOps governance, DHIS2 anomaly detection

- src/smartnet: leakage-aware ML pipeline over tagged accelerometer data
- src/smartnet/mlops: model registry, versioning, promotion gates, PSI/KS drift
  detection, retraining policy, rollback, append-only audit trail
- src/smartnet/dhis2: data-entry anomaly detection against the DHIS2 aggregate
  data model, plus a Web API client that imports DHIS2 validation rules
- 86 tests passing on synthetic fixtures; no study data required

Study data and trained model binaries are excluded by .gitignore.}"

git commit -q -m "$MSG"
ok "committed $STAGED files"

# --- 5. push (only when asked) -----------------------------------------------
say ""
if [ "${1:-}" = "--push" ]; then
  if ! git remote get-url origin >/dev/null 2>&1; then
    die "No 'origin' remote. Add one:
     git remote add origin https://github.com/Ssemukuye/smartnet-bednet-ml.git"
  fi
  say "${BOLD}Pushing to origin/main${OFF}"
  if git push origin main; then
    ok "pushed"
    say ""
    say "  Repository: $(git remote get-url origin | sed 's/\.git$//')"
  else
    say ""
    warn "push rejected — histories have probably diverged. Try:"
    say "     git pull --rebase origin main"
    say "     git push origin main"
  fi
else
  say "  Committed locally. Nothing pushed."
  say "  To push:  ${BOLD}bash push.sh --push${OFF}"
fi

say ""
say "  Reminder: keep the repository ${BOLD}private${OFF} until your PI confirms"
say "  in writing that publishing derived study results is acceptable."
say ""
