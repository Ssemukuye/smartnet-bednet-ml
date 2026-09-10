# Skills mapping — Palladium AI & Machine Learning Specialist (DART)

Mapping this project and my professional experience against the posted requirements for the AI & Machine Learning Specialist role on the Data-Activated Architecture for Resilient Transition (DART) project.

Gaps are marked honestly. Where I cannot evidence something, it says so, with the smallest concrete step that would close it.

---

## Qualifications

| Requirement | Evidence in this project | Evidence in professional experience | Status |
|---|---|---|---|
| **Master's in data science, CS, AI, statistics, biomedical engineering or health informatics** | — | MSc Data Science and Analytics (Research), Uganda Christian University. CGPA 4.18/5.0. Coursework includes Applied Machine Learning (77), Big Data Analytics (80), Data Engineering and Cloud Computing (81), AI and Machine Learning | **Met** |
| **5+ years in ML, data science, predictive analytics or AI product development** | This project is built on that work | Six years on the bed-net accelerometer study (Sep 2020–present): model development, validation, feature engineering. Eight years in research data management across four disease areas | **Met** |
| **Python, ML frameworks, model evaluation, data engineering, data visualisation** | ~1,900 lines of Python: pandas, NumPy, scikit-learn, matplotlib. Nine modules, 23 tests, staged pipeline, 14 figures | ETL pipelines converting raw 10 Hz binary accelerometer output to analysis-ready feature sets; Random Forest classifiers evaluated on accuracy, sensitivity, specificity and AUC | **Met** |
| **AI or analytics in health, public health or low-resource settings** *(strongly preferred)* | Malaria-prevention behaviour classification on real study data collected in field conditions | Malaria (bed-net, PBO net), tuberculosis (LAM), HIV (viral load), immunisation (GAVI) studies across multi-site Ugandan facilities | **Strong** |
| **Responsible AI, privacy, bias assessment, human-centred design, model monitoring** | Leakage-aware validation; balanced accuracy reported alongside accuracy; per-class limitations published; data governance policy; abstention recommended over forced prediction | Human-subjects data governance, confidentiality controls, access management, breach identification; subgroup performance analysis across adult and paediatric participants | **Partial** — see gaps 3 and 4 |
| **Technical documentation, stakeholder engagement, English communication** | README written for both technical reviewers and programme staff; every limitation documented; notebooks structured as narrative | Analytical reports for investigators; qualitative fieldwork with District Health Officers, Chief Administrative Officers, EPI focal persons and health-centre in-charges | **Met** |

---

## Primary responsibilities

| Responsibility | Evidence in this project | Evidence in professional experience |
|---|---|---|
| **Support an MOH-owned AI implementation framework — use-case intake, risk classification, data-governance requirements, validation criteria, approval pathways, monitoring, escalation** | The repository is a worked example of several framework components: documented validation criteria (grouped-event CV, balanced accuracy, per-class sensitivity thresholds), an explicit data-governance policy (`data/README.md`), and a recommended operating point with stated boundaries of use | Applied data-protection and confidentiality procedures across four concurrent studies; trained study staff on personal data protection practices |
| **Design, develop, validate and document approved AI use cases — DHIS2 data-entry anomaly detection; AI-enabled acoustic screening for respiratory disease** | End-to-end design, development, validation and documentation of a health AI use case from raw extract to demonstration app | **Acoustic screening:** managed AI-application audio recordings, chest radiography images and TB sputum results on the LAM tuberculosis study — the same data modality as the named use case. **Anomaly detection:** reconciled facility source documents against database records across Health Centre IV sites, systematically identifying data-entry errors |
| **Work with data engineering, interoperability, cybersecurity, clinical and programme teams to ensure approved data sources, secure environments, documented pipelines** | Fully documented pipeline: staged, resumable, seeded, tested. Every preprocessing decision recorded with its rationale, including the decisions *not* taken | Administered Windows Server, Active Directory permissions and VPN access for confidential study data; designed backup and disaster-recovery procedures; collaborated with investigators, statisticians and field teams |
| **Develop model-development, testing, evaluation, bias assessment, performance-monitoring, versioning and retraining procedures** | Model progression from baseline to ensembles; three validation designs compared; leakage quantified at 59.6%; per-class evaluation; held-out permutation importance; feature ablation. **Versioning, monitoring and retraining procedures are not implemented** | Trained and validated Random Forest classifiers across 3-, 4- and 5-category schemes; selected production models against documented criteria; compared performance across demographic subgroups |
| **Ensure human oversight, auditability, privacy protection, appropriate use of de-identified or authorised data** | Study data excluded from version control; synthetic generator provided for reproduction; PI permission required before publication; low-confidence predictions routed to human review in the demo app | Human-subjects research data handling under ethical approval across all studies; confidentiality protocols for patient-level HIV, TB and malaria data |
| **Support user testing, workflow integration, training, post-deployment monitoring, decision thresholds** | Decision thresholds addressed directly: the recommended four-category operating point, and the recommendation that the model abstain on direction rather than guess. **User testing and post-deployment monitoring are not evidenced** | Trained study staff on data security, file management and data protection; resolved data queries with investigators on weekly calls |
| **Document technical methods, model limitations, validation evidence, user guidance, governance decisions for MOH review** | A dedicated Limitations section with seven numbered limitations, including one that undercuts the project's own headline result | Documented modelling methods, evaluation criteria and data-management procedures for investigator and ethical review |
| **Build capacity of MOH and Ugandan technical teams in responsible AI and model monitoring** | Repository is written to be handed over: modular library, tests on synthetic fixtures, narrative notebooks, reproducible from a clean clone | Trained study staff on data protection practices; supported users across multiple sites |

---

## What this project demonstrates that a CV bullet cannot

**1. I report results that undercut my own thesis.**
I built this expecting to show that random splitting inflates performance. Measured, the inflation was 0.3–1.0 percentage points — small. The README leads with that negative result rather than burying it. For a role where a Ministry of Health has to trust reported model performance, this is the relevant trait.

**2. I distinguish a headline number from a usable one.**
94.7% accuracy is technically true and practically misleading. The project's central finding is the 16-point gap between accuracy and balanced accuracy, and the 0.34 sensitivity on entering a net that causes it.

**3. I know when a method does not fit.**
Deep learning is discussed and rejected with a stated reason: 664 independent events and 91 minority-class examples. Frequency-domain features are rejected because the raw waveform is not in the extract. Both would have been easy to add for appearance.

**4. I design validation before modelling.**
The grouping structure was established from the data — 664 label-pure contiguous events — before any model was fitted.

**5. I treat data governance as engineering.**
Study data is git-ignored, a synthetic generator enables reproduction, and PI permission is required before publication. This is the same reasoning MOH data would require.

**6. I found a deployment constraint the accuracy score hides.**
Removing forward context collapses balanced accuracy from 0.797 to 0.593, so the classifier needs ~10 seconds of lookahead and cannot run in true real time. That constraint would only surface in production — or in an ablation.

---

## Gaps closed by building, not explaining

A mentor at US Mission Uganda / CDC reviewed this CV and named the likely interview probes. Two were buildable, so they were built.

### Model governance lifecycle — **now implemented**

The review flagged the absence of *"model registry/versioning → deployment → drift/performance monitoring → retraining trigger → controlled promotion → rollback/audit trail."* That exact chain now lives in `src/smartnet/mlops/`, running and tested:

| Requirement | Implementation | Evidence |
|---|---|---|
| Registry & versioning | `ModelRegistry`, auto-incrementing versions, joblib artefacts | `models/registry/index.json` |
| Training-data lineage | SHA-256 content hash of the exact training matrix | `training_data_hash` per version |
| Controlled promotion | `registered → staging → production`; illegal transitions rejected | `ALLOWED_TRANSITIONS` |
| Quality gates | Thresholds on **balanced accuracy and worst-class sensitivity**, not headline accuracy | `PromotionGate` |
| Drift detection | PSI + Kolmogorov–Smirnov per feature against a captured reference profile | `reports/monitoring/` |
| Prediction drift | Label-free early warning via predicted-class PSI | `detect_prediction_drift` |
| Retraining trigger | Explicit multi-signal policy, not a calendar | `RetrainingPolicy` |
| Rollback | One call restores the last approved version | `registry.rollback()` |
| Audit trail | Append-only JSONL: actor, reason, timestamp, stage change | `models/registry/audit_log.jsonl` |

**The demonstration that matters.** In step 2 of the lifecycle run, the gate **blocked** the five-category model from production:

```
BLOCKED — balanced_accuracy 0.774 < 0.90;
          worst_class_sensitivity 0.340 < 0.75;
          accuracy-balanced gap 0.173 > 0.10
          (headline metric is masking minority-class failure)
```

That is this project's central scientific finding encoded as an automated governance control. The four-category model then passed the same gate on its merits; a later bad promotion was rolled back with the reason recorded. Twelve immutable audit records, 26 tests.

**Honest boundary:** an implemented lifecycle is not one operated under production load. I have not run a model serving live traffic. Say so.

### DHIS2 — **partially closed**

Still no live-instance experience, and the DHIS2 Academy course is in progress. But the named use case is built: `src/smartnet/dhis2/` detects data-entry anomalies against the DHIS2 aggregate data model (`dataElement / period / orgUnit / categoryOptionCombo`), so it runs on a `/api/dataValueSets` export unchanged.

Six detector families — impossible values, MAD outliers, cross-element consistency, level shift, digit preference, missing reports — evaluated on 5,700+ synthetic facility-months with 169 injected errors of known type:

| Metric | Value |
|---|---|
| Precision @ top 10 / 25 / 50 / 100 | **1.000** |
| Overall precision / recall / F1 | 0.618 / 0.899 / 0.733 |
| Recall — impossible magnitude | 1.00 |
| Recall — extra digit | 0.98 |
| Recall — negative value | 0.97 |
| Recall — consistency break | 0.89 |
| Recall — **digit transposition** | **0.61** |

Transposition is the honest weak spot: swapping two digits usually leaves the value inside the plausible range, so univariate detectors miss it. Catching it needs cross-element or historical-ratio reasoning — which is exactly where an ML layer earns its place over rules, and a good thing to say unprompted.

**Still to do:** finish the Academy, then run this pipeline against the [DHIS2 demo database](https://play.im.dhis2.org/) so the claim rests on real metadata rather than a faithful simulation.

---

## Remaining honest gaps

### Deep learning and computer vision — no production work
I have managed chest radiography data; I have not built vision models. This project explains why sequence models were not appropriate here, which demonstrates judgement but not capability.

**Step:** a separate small project on a public dataset — 1D CNN on the UCI HAR accelerometer benchmark. Directly adjacent, and honest about being a learning exercise.

### Direct acoustic-model development
I managed the data pipeline for AI-application audio on the LAM tuberculosis study; I did not build the acoustic model. That is the closest anyone is likely to get without having worked on a cough-classification project, but the distinction is real and should be stated plainly.

### Human-centred design — no formal grounding
Named in the posting; no training or project evidence.

**Step:** read one primer and be able to discuss user testing and workflow integration for a clinical tool.

### 5. Fairness analysis — blocked by the data, not by skill
The published study found models significantly less accurate for children than adults (87.8% vs 70.0%). This extract has no participant identifiers or demographics, so that analysis cannot be reproduced. I have done subgroup performance analysis professionally on the same study programme.

**Step:** request the participant-linked extract from the PI. If granted, participant-level grouping and the adult/child comparison become the strongest additions to this repository.

### 6. Cloud and MLOps tooling — limited
No Azure, GCP or AWS ML experience. Everything here runs locally.

**Step:** the free Microsoft Learn path for DP-700 (Fabric Data Engineer Associate), which also carries a certification.

---

## Summary

| Area | Position |
|---|---|
| Health and public-health data | Strong — eight years, four disease areas, low-resource multi-site settings |
| Applied ML on real health data | Solid — end-to-end, validated, honestly reported |
| Data engineering and pipelines | Strong — professional and demonstrated |
| Responsible AI and governance | Strong — privacy, limitation reporting, and a governance lifecycle implemented in code |
| Model monitoring and MLOps | Implemented and tested; not yet operated in production |
| DHIS2 and national HIS platforms | Use case built against the data model; no live-instance experience yet |
| Deep learning and computer vision | Weak, and stated as such |
| Stakeholder engagement and documentation | Strong |

I am not a senior ML engineer and this project does not claim otherwise. What it shows is a health-data professional with postgraduate data-science training who can take a real study dataset end to end, choose a defensible experimental design, and report what the model genuinely can and cannot do — including when that is unflattering.
