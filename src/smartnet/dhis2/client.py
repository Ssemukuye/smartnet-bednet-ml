"""DHIS2 Web API client — pulls metadata and data into the anomaly pipeline.

Verified against the DHIS2 2.43.1 Sierra Leone demo
(``https://play.im.dhis2.org/stable-2-43-1``). The endpoint shapes, expression
grammar and organisation-unit hierarchy encoded here were read from a live
instance, not from documentation.

Two things this module does that matter:

1. **Imports DHIS2's own validation rules** rather than hand-coding consistency
   checks. DHIS2 already ships rules like ``ANC 2 <= ANC 1``; re-typing them
   would guarantee drift between the platform and the detector. Parsing them
   from ``/api/validationRules`` means the detector inherits whatever the MOH
   has configured.
2. **Normalises analytics output** into the flat
   ``dataElement / period / orgUnit / value`` frame the detectors expect.

Credentials are read from the environment, never from code::

    export DHIS2_BASE_URL="https://play.im.dhis2.org/stable-2-43-1"
    export DHIS2_USERNAME="..."
    export DHIS2_PASSWORD="..."

For a real Ministry instance, use a personal access token instead of a
password and scope the account to read-only.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlencode

import pandas as pd

from smartnet.dhis2.anomaly import ConsistencyRule

logger = logging.getLogger(__name__)

#: DHIS2 validation-rule expressions look like ``#{dataElementUid.categoryOptionComboUid}``
#: joined by arithmetic operators, e.g.
#: ``#{cYeuwXTCPkU.pq2XI5kz2BY}+#{cYeuwXTCPkU.PT59n8BQbqM}`` (ANC 2, Fixed + Outreach).
EXPRESSION_TOKEN = re.compile(r"#\{([A-Za-z][A-Za-z0-9]{10})(?:\.([A-Za-z][A-Za-z0-9]{10}))?\}")

#: DHIS2 operator -> the relation our ConsistencyRule understands.
OPERATOR_MAP = {
    "less_than_or_equal_to": "lte",
    "less_than": "lte",
    "greater_than_or_equal_to": "gte",
    "greater_than": "gte",
}

#: Organisation-unit levels in the Sierra Leone demo. A real deployment may
#: differ; read them from /api/organisationUnitLevels rather than assuming.
DEMO_ORG_LEVELS = {1: "National", 2: "District", 3: "Chiefdom", 4: "Facility"}


def extract_data_elements(expression: str) -> list[str]:
    """Return the data-element UIDs referenced by a validation-rule expression."""
    return [m.group(1) for m in EXPRESSION_TOKEN.finditer(expression or "")]


@dataclass
class DHIS2Client:
    """Thin read-only client. Requests is imported lazily so the rest of the
    package, and the test suite, work with no network dependency at all."""

    base_url: str | None = None
    username: str | None = None
    password: str | None = None
    timeout: int = 60

    def __post_init__(self) -> None:
        self.base_url = (self.base_url or os.environ.get("DHIS2_BASE_URL", "")).rstrip("/")
        self.username = self.username or os.environ.get("DHIS2_USERNAME")
        self.password = self.password or os.environ.get("DHIS2_PASSWORD")
        if not self.base_url:
            raise ValueError(
                "No DHIS2 base URL. Set DHIS2_BASE_URL or pass base_url explicitly."
            )

    # ---------------------------------------------------------------- http
    def get(self, path: str, **params: Any) -> dict:
        import requests  # imported here so the package has no hard dependency

        url = f"{self.base_url}/api/{path.lstrip('/')}"
        if params:
            url = f"{url}?{urlencode(params, doseq=True)}"
        auth = (self.username, self.password) if self.username else None
        resp = requests.get(url, auth=auth, headers={"Accept": "application/json"},
                            timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------ metadata
    def system_info(self) -> dict:
        info = self.get("system/info")
        return {k: info.get(k) for k in ("version", "revision", "calendar", "contextPath")}

    def org_unit_levels(self) -> dict[int, str]:
        j = self.get("organisationUnitLevels", fields="level,name", order="level:asc")
        return {l["level"]: l["name"] for l in j.get("organisationUnitLevels", [])}

    def data_elements(self, name_filter: str | None = None, page_size: int = 200) -> pd.DataFrame:
        params = {"fields": "id,name,valueType,domainType", "pageSize": page_size}
        if name_filter:
            params["filter"] = f"name:like:{name_filter}"
        return pd.DataFrame(self.get("dataElements", **params).get("dataElements", []))

    # ----------------------------------------------------- validation rules
    def fetch_validation_rules(self, page_size: int = 200) -> tuple[list[ConsistencyRule], pd.DataFrame]:
        """Import DHIS2 validation rules as ``ConsistencyRule`` objects.

        Returns ``(rules, raw_frame)``. Rules whose expressions combine several
        data elements are kept in the raw frame but skipped as rules, because a
        single-element comparison is the only form the detector evaluates —
        silently mis-translating a compound expression would be worse than
        skipping it.
        """
        j = self.get(
            "validationRules",
            fields="name,importance,operator,periodType,"
                   "leftSide[expression,description],rightSide[expression,description]",
            pageSize=page_size,
        )
        raw = j.get("validationRules", [])
        uid_to_name = self._data_element_names(
            {u for r in raw for side in ("leftSide", "rightSide")
             for u in extract_data_elements((r.get(side) or {}).get("expression", ""))}
        )

        rules: list[ConsistencyRule] = []
        rows: list[dict] = []
        for r in raw:
            left = (r.get("leftSide") or {})
            right = (r.get("rightSide") or {})
            l_uids = set(extract_data_elements(left.get("expression", "")))
            r_uids = set(extract_data_elements(right.get("expression", "")))
            relation = OPERATOR_MAP.get(r.get("operator", ""))

            translatable = len(l_uids) == 1 and len(r_uids) == 1 and relation is not None
            rows.append({
                "name": r.get("name"),
                "importance": r.get("importance"),
                "operator": r.get("operator"),
                "periodType": r.get("periodType"),
                "left": left.get("description"),
                "right": right.get("description"),
                "translatable": translatable,
            })
            if not translatable:
                continue

            l_uid, r_uid = next(iter(l_uids)), next(iter(r_uids))
            num, den = (l_uid, r_uid) if relation == "lte" else (r_uid, l_uid)
            rules.append(ConsistencyRule(
                name=r.get("name", f"{num}_vs_{den}"),
                numerator=uid_to_name.get(num, num),
                denominator=uid_to_name.get(den, den),
                relation="lte",
                explanation=f"DHIS2 validation rule violated: {r.get('name')}",
            ))

        logger.info("imported %d/%d DHIS2 validation rules as consistency rules",
                    len(rules), len(raw))
        return rules, pd.DataFrame(rows)

    def _data_element_names(self, uids: Iterable[str]) -> dict[str, str]:
        uids = [u for u in uids if u]
        if not uids:
            return {}
        out: dict[str, str] = {}
        for i in range(0, len(uids), 50):  # keep the URL a sane length
            chunk = uids[i:i + 50]
            j = self.get("dataElements", fields="id,name",
                         filter=f"id:in:[{','.join(chunk)}]", pageSize=len(chunk))
            out.update({d["id"]: d["name"] for d in j.get("dataElements", [])})
        return out

    # ---------------------------------------------------------------- data
    def fetch_analytics(
        self,
        data_element_uids: list[str],
        org_unit_uids: list[str],
        period: str = "LAST_12_MONTHS",
        org_unit_level: int | None = 4,
    ) -> pd.DataFrame:
        """Aggregate values as a DHIS2-shaped frame the detectors accept.

        ``skipRounding`` is set because rounded analytics output would corrupt
        digit-preference detection.
        """
        ou = ";".join(org_unit_uids) + (f";LEVEL-{org_unit_level}" if org_unit_level else "")
        j = self.get(
            "analytics",
            dimension=[f"dx:{';'.join(data_element_uids)}", f"pe:{period}", f"ou:{ou}"],
            skipRounding="true",
        )
        items = (j.get("metaData") or {}).get("items") or {}
        name = lambda uid: (items.get(uid) or {}).get("name", uid)

        rows = [{
            "dataElement": name(r[0]),
            "period": r[1],
            "orgUnit": name(r[2]),
            "categoryOptionCombo": "default",
            "value": float(r[3]),
        } for r in j.get("rows", [])]

        df = pd.DataFrame(rows)
        logger.info("fetched %d values across %d org units and %d periods",
                    len(df), df["orgUnit"].nunique() if len(df) else 0,
                    df["period"].nunique() if len(df) else 0)
        return df

    def fetch_data_value_sets(
        self, data_set: str, org_unit: str, start_date: str, end_date: str,
        children: bool = True,
    ) -> pd.DataFrame:
        """Raw (unaggregated) data values — preferred when auditing entry errors.

        Analytics aggregates across category option combos; raw values are what
        a data clerk actually typed, which is what a data-entry anomaly
        detector should be looking at.
        """
        j = self.get("dataValueSets", dataSet=data_set, orgUnit=org_unit,
                     startDate=start_date, endDate=end_date,
                     children=str(children).lower())
        vals = j.get("dataValues", [])
        if not vals:
            return pd.DataFrame(columns=["dataElement", "period", "orgUnit",
                                         "categoryOptionCombo", "value"])
        df = pd.DataFrame(vals)
        df = df.rename(columns={"categoryOptionCombo": "categoryOptionCombo"})
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.dropna(subset=["value"])[
            ["dataElement", "period", "orgUnit", "categoryOptionCombo", "value"]
        ]
