# Data directory — read before adding anything here

## The study data is not in this repository, and must not be committed

The analysis in this project uses a tagged accelerometer extract from a bed-net
monitoring study. That file is **human-subjects research data belonging to the
study investigators and their institution**, not to the author of this
repository. It is excluded by `.gitignore` and is not distributed here.

This is not a formality. A repository that demonstrates responsible handling of
health data while publishing someone else's participant data would refute its
own argument.

**Before publishing this repository anywhere public, the author must obtain
written permission from the study Principal Investigator** covering:

1. Publishing derived results, figures and performance metrics.
2. Naming the study in a public portfolio context.
3. If applicable, whether any data may be redistributed at all.

Until that permission exists, keep the repository private.

---

## Expected layout

```
data/
├── raw/         tagged extract, as supplied  (git-ignored)
├── interim/     intermediate artefacts       (git-ignored)
└── processed/   modelling frames             (git-ignored)
```

Place the tagged Stata extract at:

```
data/raw/tagged train data.dta
```

The filename is configurable via `smartnet.config.RAW_STATA_FILENAME`.

---

## Running without the study data

The pipeline is fully reproducible on synthetic data with an identical schema:

```bash
python scripts/make_synthetic_sample.py     # writes data/raw/synthetic_sample.dta
PYTHONPATH=src python scripts/run_pipeline.py
```

The synthetic generator produces the same 53 columns, the same nested label
schemes and the same contiguous-event structure. Numbers produced from it are
**not** study results and are labelled as such. The unit test suite runs
entirely on synthetic fixtures and needs no real data.

---

## Schema

| Group | Columns | Description |
|---|---|---|
| Timestamp | `date_str` | Second-resolution timestamp of the tagged epoch |
| Labels | `motion_something`, `motion_3cat`, `motion_4cat`, `motion_5cat` | Nested classification schemes, coarsest to finest |
| Current epoch | `sum_vectormagnitudes`, `stdvz` | Summary statistics for the labelled second |
| Backward lags | `back_stdvz1`–`back_stdvz10`, `back_sumvector1`–`back_sumvector10` | The 10 preceding epochs |
| Forward leads | `forward_stdvz1`–`forward_stdvz10`, `forward_sumvector1`–`forward_sumvector10` | The 10 following epochs |
| Window aggregates | `sumover10vector_back`, `meanzover10vector_back`, `stdvzover10vector_back`, and the three `_forward` equivalents | Statistics over the 10-epoch windows |

### Label schemes

| Scheme | Classes |
|---|---|
| `motion_something` | Nothing, Something |
| `motion_3cat` | Nothing, Down event, Put up |
| `motion_4cat` | Nothing, Put down, Put up, Enter or exit |
| `motion_5cat` | Nothing, Put down, Put up, Enter, Exit |

The schemes are nested and mutually consistent; this is asserted by
`validation.check_label_consistency`.

### Structural properties verified on load

- 1,340 labelled epochs, 10 September – 4 October 2021
- 664 contiguous motion events (runs of consecutive 1-second epochs)
- 25 recording days
- No missing values, no duplicate feature vectors, all timestamps unique
- Every derived event is label-pure

### Not present in this extract

There is **no participant, device or bed-net identifier**. Grouping is therefore
done at event and recording-day level rather than participant level. This is a
real limitation on what can be claimed about generalisation to new people, and
it is stated as such throughout the analysis rather than glossed over.
