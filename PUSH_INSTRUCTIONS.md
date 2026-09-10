# Pushing the rest of the project

Your repo currently contains **one commit with one notebook**. Everything else — the
package, the tests, the README, the MLOps lifecycle, the DHIS2 work — is still only
on your Mac.

Run these yourself. I can't: I don't handle credentials, my shell is a different
machine from yours, and the mounted folder blocks git's lock files.

---

## Step 0 — Remove the stale lock file (do this first)

My dry-run left a lock behind that my sandbox can't delete. Git will refuse to do
anything until it's gone:

```bash
cd ~/Documents/"Job Applications"/"Data engineering DFCU"/smartnet-bednet-ml
rm -f .git/index.lock
```

---

## Step 1 — Stage everything

```bash
git add -A
```

## Step 2 — Pre-flight check (do not skip)

```bash
# Study data — must print NOTHING:
git ls-files --cached | grep -Ei '\.dta$|^data/raw/|^data/processed/'

# Model binaries — must print NOTHING (13 MB forests; git keeps every version forever):
git ls-files --cached | grep -Ei '\.joblib$|\.pkl$'

# Sanity: roughly 100 files
git ls-files --cached | wc -l
```

If either grep prints anything, stop — `.gitignore` isn't being applied and you'd be
committing something that shouldn't leave your machine.

## Step 3 — Commit

```bash
git commit -m "Add full project: pipeline, MLOps governance, DHIS2 anomaly detection

- src/smartnet: leakage-aware ML pipeline over tagged accelerometer data
- src/smartnet/mlops: model registry, versioning, promotion gates, PSI/KS drift
  detection, retraining policy, rollback, append-only audit trail
- src/smartnet/dhis2: data-entry anomaly detection against the DHIS2 aggregate
  data model, plus a Web API client that imports DHIS2 validation rules
- 86 tests, all passing on synthetic fixtures; no study data required
- Notebooks, figures, aggregate results and documentation

Study data and trained model binaries are excluded by .gitignore."
```

## Step 4 — Push

```bash
git push origin main
```

If it rejects because the remote has your earlier notebook commit and histories
have diverged:

```bash
git pull --rebase origin main    # then resolve if needed
git push origin main
```

---

## What gets published, and what does not

| Committed | Excluded |
|---|---|
| All source code and tests | `data/raw/*.dta` — study data |
| README, notes, documentation | `models/registry/artefacts/*.joblib` — 28 MB of binaries |
| Notebooks | `data/processed/`, `data/interim/` |
| Figures and aggregate results (CSV, JSON) | Any `.pkl` |
| **Registry index and audit trail** | |

The registry index and audit log **are** tracked deliberately — they are the
governance record, they are small, and they diff readably. They reference artefact
files that aren't in the repo; that's intentional and noted in `.gitignore`. Anyone
can regenerate them:

```bash
PYTHONPATH=src python scripts/run_mlops_lifecycle.py
```

---

## Still outstanding

**PI permission.** The repo publishes derived results from the bed-net study —
confusion matrices, class counts, recording dates, figures. The raw data is excluded
and I verified that, but the derived results still come from someone else's study.
Keep the repository **private** until your PI confirms in writing that publishing
those is fine. A private repo you can screen-share is worth more than a public one
that shouldn't exist.

**Tidy the registry.** Six versions accumulated across development runs
(v1–v6, two of them `staging` leftovers). For a clean demonstration, delete
`models/registry/index.json` and `audit_log.jsonl` and re-run the lifecycle once —
it produces a tidy v1/v2/v3 story: blocked promotion, successful promotion, rollback.

**Stray test artefacts.** `models/registry/artefacts/m_v*.joblib` are leftovers from
before I fixed the test suite writing into the real registry. They're git-ignored
now, but you can delete them:

```bash
rm -f models/registry/artefacts/m_v*.joblib
```

---

## After the push

1. **Set the repository description and topics** — text is in
   `../Palladium_AI_ML_Specialist_Application/PROJECT_DESCRIPTIONS.md`, section 3.
2. **Check the rendered README** — confirm all 14 figures display. GitHub is
   case-sensitive about paths in a way macOS is not.
3. **Verify from a clean clone** that someone else can actually run it:

   ```bash
   git clone <url> /tmp/check && cd /tmp/check
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   PYTHONPATH=src python scripts/make_synthetic_sample.py
   PYTHONPATH=src pytest tests/ -q        # expect: 86 passed
   ```

   This step matters. A repository that only runs on the author's machine is a
   liability in a technical interview.
