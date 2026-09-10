# Understanding DHIS2

Notes taken from a live instance — the DHIS2 **2.43.1 Sierra Leone demo** at
`play.im.dhis2.org/stable-2-43-1` — not from documentation. Everything below was
read from the API, and the numbers are what that instance actually contains.

---

## 1. The one idea that explains the rest

DHIS2 separates **metadata** (what can be recorded) from **data** (what was recorded).
Almost every confusion about DHIS2 comes from mixing the two up.

A single aggregate data value is identified by a four-part key:

```
dataElement  ×  period  ×  orgUnit  ×  categoryOptionCombo   →   value
```

- **dataElement** — what is counted ("ANC 1st visit")
- **period** — when ("202601" = January 2026)
- **orgUnit** — where ("Ngelehun CHC")
- **categoryOptionCombo** — the disaggregation ("Fixed" vs "Outreach")

Miss the fourth dimension and your numbers will silently disagree with the platform's,
because DHIS2 stores ANC 1 *twice* — once for Fixed, once for Outreach — and the
headline figure is their sum.

---

## 2. What is in this instance

| Metadata object | Count |
|---|---|
| Data elements | 1,209 |
| Data sets | 33 |
| Organisation units | 1,332 |
| Organisation unit levels | 4 |
| Indicators | 77 |
| Validation rules | 37 |
| Category combinations | 26 |
| Period types | 24 |

**Data coverage:** September 2025 – August 2026. (Demo databases roll their data
forward, so never hard-code a date range — ask analytics with `LAST_12_MONTHS`.)

### Organisation hierarchy

```
1  National
2  District      e.g. Bo, Bombali, Kailahun
3  Chiefdom      e.g. Badjia, Baoma, Bargbe
4  Facility      e.g. Ngelehun CHC, Kigbai MCHP
```

Uganda's HMIS uses a similar shape (National → Region → District → HSD → Facility),
so the pattern transfers; the level count does not. **Read
`/api/organisationUnitLevels` rather than assuming four.**

The `LEVEL-4` shorthand in an analytics request means "all facilities beneath the
org units I named", which is how you get facility-level data without listing
1,161 UIDs.

### Data sets

Data sets group data elements into a reporting form with a period type:

```
Monthly     | Reproductive Health, Child Health, HIV Care Monthly, EPI Stock
Weekly      | IDSR Weekly
Quarterly   | FBP qualité technique (score)
Yearly      | Facility Assessment
SixMonthly  | Clinical Monitoring Checklist
```

`Reproductive Health` (`QX4ZTUbOt3a`) is monthly and assigned to **1,161 org units** —
that assignment is what determines who is *expected* to report, which is the basis
of any completeness or missing-report check.

---

## 3. Validation rules — DHIS2 already does consistency checking

This was the most useful discovery. DHIS2 ships a validation-rule engine, and the
rules look exactly like the consistency checks a data-quality tool would write:

| Rule | Operator | Importance |
|---|---|---|
| ANC 2 <= ANC 1 | `less_than_or_equal_to` | MEDIUM |
| ANC 3 <= ANC 2 | `less_than_or_equal_to` | MEDIUM |
| Cholera outbreak with 50% or more increase above average | `less_than_or_equal_to` | HIGH |
| Commodities Amoxicillin (balance + ordered >= consumption) | `greater_than_or_equal_to` | MEDIUM |

### Expression grammar

Each side of a rule is an expression over data elements:

```
#{cYeuwXTCPkU.pq2XI5kz2BY} + #{cYeuwXTCPkU.PT59n8BQbqM}
  └─ dataElement UID ──┘   └─ categoryOptionCombo UID ─┘
```

That example is ANC 2 = Fixed + Outreach. The grammar is
`#{dataElementUid.categoryOptionComboUid}` joined by arithmetic operators;
the category combo part is optional.

The ANC UIDs in this instance:

| Data element | UID |
|---|---|
| ANC 1st visit | `fbfJHSPpUQD` |
| ANC 2nd visit | `cYeuwXTCPkU` |
| ANC 3rd visit | `Jtf34kNZhzP` |

**Implication for our detector.** Hand-coding consistency rules would guarantee
drift between the platform and the tool. `smartnet.dhis2.client.fetch_validation_rules()`
therefore parses these expressions and returns `ConsistencyRule` objects, so the
detector inherits whatever the Ministry has configured. Compound expressions
(two or more data elements on one side) are **skipped rather than approximated** —
silently mis-translating a rule is worse than not translating it.

---

## 4. Analytics vs data value sets — the distinction that matters

| | `/api/analytics` | `/api/dataValueSets` |
|---|---|---|
| Returns | Aggregated, resolved | Raw stored values |
| Category combos | Summed away | Preserved |
| Org units | Aggregates up the hierarchy | Exactly as stored |
| Needs analytics tables generated | Yes | No |
| Right for | Indicators, dashboards, trends | **Data-entry auditing** |

**For anomaly detection, prefer `dataValueSets`.** Analytics shows you a rolled-up
number; a data clerk typed the raw one. If ANC 1 Fixed was mis-keyed and ANC 1
Outreach was fine, the aggregate may look plausible while the entry is wrong.

Two practical traps:

- **`skipRounding=true`** on analytics, or rounding destroys digit-preference detection.
- **Empty results are usually a period problem, not a permissions problem.** Querying
  2024 on this instance returns zero rows because the data sits in 2025–2026.

---

## 5. Endpoints worth knowing

```
/api/system/info                     version, revision, calendar
/api/organisationUnitLevels          hierarchy depth and names
/api/organisationUnits?filter=level:eq:2
/api/dataSets?fields=id,name,periodType,dataSetElements[dataElement[id,name]]
/api/dataElements?filter=name:like:ANC
/api/validationRules?fields=name,operator,leftSide[expression],rightSide[expression]
/api/dataValueSets?dataSet=&orgUnit=&startDate=&endDate=&children=true
/api/analytics?dimension=dx:UID;UID&dimension=pe:LAST_12_MONTHS&dimension=ou:UID;LEVEL-4
/api/outlierDetection?ds=&startDate=&endDate=&ou=&algorithm=MOD_Z_SCORE
```

`filter` uses `property:operator:value` (`like`, `eq`, `in:[a,b]`), and `fields`
takes a nested selector — `dataSetElements[dataElement[id,name]]`. Both are worth
learning; they turn three round trips into one.

DHIS2's own outlier endpoint supports `Z_SCORE`, `MOD_Z_SCORE` and `MIN_MAX`.
**It uses modified z-score (median/MAD) for the same reason we did** — a single
mis-keyed extreme inflates the standard deviation enough to conceal itself. That
convergence is worth noting: the platform's designers reached the same conclusion
independently.

---

## 6. Running our detectors on the real instance

Scope: Bo district, four chiefdoms, all facilities beneath them.

| | |
|---|---|
| Values | 971 |
| Facilities | 28 |
| Periods | 12 (Sep 2025 – Aug 2026) |
| Data elements | ANC 1st / 2nd / 3rd visit |

### MAD outliers — 23 flagged

| Facility | Period | Element | Value | Facility median | Modified z |
|---|---|---|---|---|---|
| Kasse MCHP | 202511 | ANC 3rd visit | **1,935** | 24.5 | 135.6 |
| Ngelehun CHC | 202511 | ANC 1st visit | **720** | 20 | 118.0 |
| Ngelehun CHC | 202607 | ANC 1st visit | 283 | 20 | 44.3 |
| Ngelehun CHC | 202511 | ANC 2nd visit | 701 | 30.5 | 31.2 |

1,935 against a facility median of 24.5 is the classic extra-digit signature —
exactly the failure mode the detector was built for, appearing unprompted in real data.

### Consistency violations — 236 flagged

Against DHIS2's own rules (ANC 2 ≤ ANC 1, ANC 3 ≤ ANC 2). Example: Kigbai MCHP,
May 2026 — ANC 1 = 1, ANC 2 = 6, ANC 3 = 7.

**Read this number with care.** 236 violations across ~336 facility-months is a ~70%
rate, which no functioning HMIS would produce. The demo database is randomly
generated, so its values do not respect its own validation rules. This says the
detector fires correctly on rule violations; it says nothing about real Sierra
Leone or Uganda data quality. Reporting it as a real-world finding would be wrong.

---

## 7. What to look at next

- **Min-max values** (`/api/dataEntry/minMaxValues`) — DHIS2 stores per-facility,
  per-element expected ranges. Free, curated bounds our `impossible_value` detector
  could consume instead of a global ceiling.
- **Data approval workflows** — the app this exploration started in. Approval state
  determines whether data is still editable, which affects whether flagging it is
  actionable at all.
- **Data Quality app** — the built-in tool for validation-rule analysis, outlier
  detection and follow-up. Anything new must justify itself against this.
- **Completeness reporting** — `reportingRateSummary` gives expected-vs-actual
  reports per data set, a better basis for missing-report detection than inferring
  from gaps.
- **Uganda's HMIS specifics** — different hierarchy depth, different data sets,
  DHIS2 version may differ. The API grammar transfers; the metadata does not.

---

## 8. Reproducing this

The exploration above was done interactively against the live demo. To re-derive it
yourself rather than take it on trust:

```bash
export DHIS2_BASE_URL="https://play.im.dhis2.org/stable-2-43-1"
export DHIS2_USERNAME="admin"
export DHIS2_PASSWORD="..."          # never commit this

PYTHONPATH=src python scripts/explore_dhis2.py
```

That connects, reports what the instance contains, imports its validation rules,
pulls real facility data and runs the detectors — writing
`dhis2_live_validation_rules.csv`, `dhis2_live_anomalies.csv` and
`dhis2_live_manifest.json` to `reports/model_results/`.

`--no-detect` stops after the metadata step. `--district <UID>` points it at a
different organisation unit. Everything is read-only; nothing is written back.

**Note on provenance.** `smartnet/dhis2/client.py` is unit-tested against response
shapes captured verbatim from this instance, but the exploration that produced the
findings above was run interactively. Running the script is what confirms the Python
client works end to end against a live server.

---

## 9. Honest position

I have now read a live DHIS2 instance, mapped its data model, imported its
validation rules programmatically, and run detectors against its real data. That is
meaningfully more than course completion.

It is not the same as having administered a national instance. I have not
configured metadata, managed a data-entry rollout, tuned analytics table
generation, or dealt with the operational realities of district-level reporting.
The DHIS2 Academy course is in progress and this exploration sits alongside it.
