"""
Deterministic heuristic baseline agent shared by the server and inference script.
"""

from __future__ import annotations

from typing import Any


class HeuristicBaselineAgent:
    """Rule-based SOC analyst that investigates before classifying."""

    VALID_INDICATOR_TYPES = {"ip", "domain", "file_hash", "email", "url", "user"}

    BTP_TITLE_KEYWORDS = {
        "pentest",
        "red team",
        "red-team",
        "authorized",
        "maintenance",
        "scheduled backup",
        "backup job",
        "nightly backup",
        "vulnerability scan",
        "vuln scan",
        "gpo update",
        "group policy",
        "key rotation",
        "ssh key",
        "patch",
        "planned",
        "approved",
        "simulated",
        "exercise",
    }

    FP_TITLE_KEYWORDS = {
        "geo-block",
        "geoblock",
        "cdn anomaly",
        "false alarm",
        "scanner noise",
        "dns false",
        "av heuristic",
        "service account lockout",
        "login page scan",
        "automated scan",
        "rate limit",
        "honeypot",
    }

    def __init__(self) -> None:
        self._attempted_correlations: set[frozenset[str]] = set()

    def reset(self) -> None:
        self._attempted_correlations.clear()

    def next_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        alerts = obs.get("alert_queue", [])
        investigations = obs.get("investigations", {})
        budget = obs.get("investigation_budget", 0)

        if len(alerts) == 5:
            return self._next_lateral_action(alerts, investigations)
        if len(alerts) >= 20:
            return self._next_queue_action(alerts, investigations)

        unclassified_alerts = [a for a in alerts if investigations.get(a["alert_id"], {}).get("classification") is None]

        needs_followup = []
        for alert in alerts:
            aid = alert["alert_id"]
            inv = investigations.get(aid, {})
            stored_cls = inv.get("classification")
            if stored_cls in ("true_positive", "benign_true_positive"):
                if not inv.get("mapped_techniques"):
                    needs_followup.append(("technique", alert, inv))
                elif not inv.get("recommended_actions"):
                    needs_followup.append(("response", alert, inv))

        if not unclassified_alerts and not needs_followup:
            return {"action_type": "submit_investigation"}

        if needs_followup:
            action_type, followup_alert, followup_inv = needs_followup[0]
            if action_type == "technique":
                return {
                    "action_type": "map_technique",
                    "alert_id": followup_alert["alert_id"],
                    "technique_id": self._infer_technique(followup_alert),
                }
            return {
                "action_type": "recommend_action",
                "alert_id": followup_alert["alert_id"],
                "response_action": self._infer_response_action(followup_alert, followup_inv.get("classification")),
            }

        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        unclassified_alerts.sort(key=lambda alert: severity_order.get(alert.get("severity", "info"), 4))

        target = unclassified_alerts[0]
        alert_id = target["alert_id"]
        inv = investigations.get(alert_id, {})
        indicators = target.get("indicators", {})
        queried = set(inv.get("queried_sources", {}).keys())
        enriched = set(inv.get("enriched_indicators", {}).keys())
        title_lower = target.get("title", "").lower()
        source_lower = target.get("source_system", "").lower()

        max_enrich_per_type = 1 if len(unclassified_alerts) >= 10 else 2
        for indicator_type, values in indicators.items():
            if indicator_type not in self.VALID_INDICATOR_TYPES:
                continue
            if indicator_type == "user" and len(unclassified_alerts) > 1:
                continue
            for value in values[:max_enrich_per_type]:
                if indicator_type == "ip" and self._looks_private_ip(value):
                    continue
                if value not in enriched:
                    return {
                        "action_type": "enrich_indicator",
                        "indicator": value,
                        "indicator_type": indicator_type,
                        "query_alert_id": alert_id,
                    }

        smart_sources = self._smart_sources(title_lower, source_lower)
        steps_per_alert = max(1, budget // max(1, len(unclassified_alerts)))
        if len(unclassified_alerts) >= 20:
            max_log_queries = 1
        elif len(unclassified_alerts) >= 8:
            max_log_queries = min(len(smart_sources), 2)
        else:
            max_log_queries = min(len(smart_sources), 2 if steps_per_alert <= 3 else 4)

        for source in smart_sources[:max_log_queries]:
            if source not in queried:
                return {
                    "action_type": "query_logs",
                    "log_source": source,
                    "query_alert_id": alert_id,
                    "time_window_hours": 24,
                }

        correlation_action = self._next_correlation_action(
            alerts=alerts,
            target=target,
            budget=budget,
            unclassified_count=len(unclassified_alerts),
        )
        if correlation_action is not None:
            return correlation_action

        classification = self._infer_classification(target, inv)
        if not inv.get("classification"):
            confidence = self._classification_confidence(inv)
            return {
                "action_type": "classify_alert",
                "alert_id": alert_id,
                "classification": classification,
                "confidence": confidence,
            }

        if budget <= 2:
            return {"action_type": "submit_investigation"}

        return {"action_type": "noop"}

    def _next_lateral_action(
        self,
        alerts: list[dict[str, Any]],
        investigations: dict[str, Any],
    ) -> dict[str, Any]:
        ordered_alerts = sorted(alerts, key=self._lateral_stage_rank)
        for idx in range(len(ordered_alerts) - 1):
            aid_a = ordered_alerts[idx]["alert_id"]
            aid_b = ordered_alerts[idx + 1]["alert_id"]
            pair = tuple(sorted([aid_a, aid_b]))
            if frozenset(pair) in self._attempted_correlations:
                continue
            found = False
            for inv in investigations.values():
                for corr in inv.get("correlations_found", []):
                    if tuple(sorted(corr.get("alert_ids", []))) == pair:
                        found = True
                        break
                if found:
                    break
            if not found:
                self._attempted_correlations.add(frozenset(pair))
                return {
                    "action_type": "correlate_alerts",
                    "alert_id_a": aid_a,
                    "alert_id_b": aid_b,
                }

        for alert in alerts:
            aid = alert["alert_id"]
            inv = investigations.get(aid, {})
            if inv.get("classification") is None:
                return {
                    "action_type": "classify_alert",
                    "alert_id": aid,
                    "classification": "true_positive",
                    "confidence": 0.9,
                }

        for alert in alerts:
            aid = alert["alert_id"]
            inv = investigations.get(aid, {})
            if not inv.get("mapped_techniques"):
                return {
                    "action_type": "map_technique",
                    "alert_id": aid,
                    "technique_id": self._infer_technique(alert),
                }

        for alert in alerts:
            aid = alert["alert_id"]
            inv = investigations.get(aid, {})
            if not inv.get("recommended_actions"):
                return {
                    "action_type": "recommend_action",
                    "alert_id": aid,
                    "response_action": self._infer_response_action(alert, "true_positive"),
                }

        return {"action_type": "submit_investigation"}

    def _lateral_stage_rank(self, alert: dict[str, Any]) -> int:
        title = alert.get("title", "").lower()
        if "phishing email" in title:
            return 0
        if "lsass" in title or "credential dumping" in title:
            return 1
        if "rdp" in title or "lateral movement" in title:
            return 2
        if "large archive" in title or "file server" in title:
            return 3
        if "outbound https transfer" in title or "exfil" in title:
            return 4
        return 99

    def _next_queue_action(
        self,
        alerts: list[dict[str, Any]],
        investigations: dict[str, Any],
    ) -> dict[str, Any]:
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        ordered_alerts = sorted(
            alerts,
            key=lambda alert: (
                severity_order.get(alert.get("severity", "info"), 4),
                alert.get("timestamp", ""),
            ),
        )

        for alert in ordered_alerts:
            aid = alert["alert_id"]
            inv = investigations.get(aid, {})
            if inv.get("classification") is None:
                classification = self._classify_by_title(alert)
                return {
                    "action_type": "classify_alert",
                    "alert_id": aid,
                    "classification": classification,
                    "confidence": 0.85 if classification != "false_positive" else 0.75,
                }

        for alert in ordered_alerts:
            aid = alert["alert_id"]
            inv = investigations.get(aid, {})
            if not inv.get("recommended_actions"):
                return {
                    "action_type": "recommend_action",
                    "alert_id": aid,
                    "response_action": self._infer_response_action(alert, inv.get("classification")),
                }

        return {"action_type": "submit_investigation"}

    def _smart_sources(self, title_lower: str, source_lower: str) -> list[str]:
        if "email" in title_lower or "phish" in title_lower:
            return ["email_gateway", "endpoint", "dns", "firewall"]
        if "endpoint" in source_lower or "edr" in source_lower or "credential" in title_lower:
            return ["endpoint", "auth", "ids", "firewall"]
        if "firewall" in source_lower or "exfil" in title_lower or "outbound" in title_lower:
            return ["firewall", "proxy", "ids", "endpoint"]
        if "auth" in source_lower or "rdp" in title_lower or "lateral" in title_lower:
            return ["auth", "endpoint", "firewall", "ids"]
        if "dlp" in source_lower or "file" in title_lower or "stag" in title_lower:
            return ["endpoint", "auth", "firewall", "proxy"]
        if "vpn" in title_lower or "insider" in title_lower or "badge" in title_lower:
            return ["auth", "endpoint", "firewall", "proxy"]
        if "cloud" in title_lower or "s3" in title_lower or "upload" in title_lower:
            return ["cloud_trail", "endpoint", "proxy", "firewall"]
        return ["endpoint", "auth", "firewall", "email_gateway"]

    def _looks_private_ip(self, value: str) -> bool:
        if value.startswith("10.") or value.startswith("192.168."):
            return True
        if value.startswith("172."):
            parts = value.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                return 16 <= int(parts[1]) <= 31
        return False

    def _next_correlation_action(
        self,
        alerts: list[dict[str, Any]],
        target: dict[str, Any],
        budget: int,
        unclassified_count: int,
    ) -> dict[str, Any] | None:
        alert_id = target["alert_id"]
        other_alerts = [alert for alert in alerts if alert["alert_id"] != alert_id]

        target_indicator_vals = set()
        for vals in target.get("indicators", {}).values():
            if isinstance(vals, list):
                target_indicator_vals.update(vals)

        for other in other_alerts:
            pair = frozenset([alert_id, other["alert_id"]])
            if pair in self._attempted_correlations:
                continue
            other_indicator_vals = set()
            for vals in other.get("indicators", {}).values():
                if isinstance(vals, list):
                    other_indicator_vals.update(vals)
            if target_indicator_vals & other_indicator_vals:
                self._attempted_correlations.add(pair)
                return {
                    "action_type": "correlate_alerts",
                    "alert_id_a": alert_id,
                    "alert_id_b": other["alert_id"],
                }

        steps_per_alert = max(1, budget // max(1, unclassified_count))
        corr_attempts_for_alert = sum(1 for pair in self._attempted_correlations if alert_id in pair)
        max_corr = min(len(other_alerts), 3 if steps_per_alert <= 3 else len(other_alerts))
        if corr_attempts_for_alert < max_corr:
            for other in other_alerts[:max_corr]:
                pair = frozenset([alert_id, other["alert_id"]])
                if pair not in self._attempted_correlations:
                    self._attempted_correlations.add(pair)
                    return {
                        "action_type": "correlate_alerts",
                        "alert_id_a": alert_id,
                        "alert_id_b": other["alert_id"],
                    }
        return None

    def _infer_classification(self, target: dict[str, Any], inv: dict[str, Any]) -> str:
        title_lower = target.get("title", "").lower()
        indicators = target.get("indicators", {})
        alert_severity = target.get("severity", "").lower()
        enrichment_results = inv.get("enriched_indicators", {})

        malicious_count = sum(
            1 for result in enrichment_results.values() if isinstance(result, dict) and result.get("malicious")
        )
        high_threat_count = sum(
            1
            for result in enrichment_results.values()
            if isinstance(result, dict) and result.get("threat_score", 0) >= 70
        )

        log_evidence_suspicious = False
        log_evidence_benign = False
        for entries in inv.get("queried_sources", {}).values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                details = entry.get("details", {})
                if details.get("macro_detected") or details.get("encoded"):
                    log_evidence_suspicious = True
                if details.get("authorized") or details.get("scheduled"):
                    log_evidence_benign = True
                if entry.get("severity") == "critical":
                    log_evidence_suspicious = True

        is_likely_btp = False
        if any(keyword in title_lower for keyword in self.BTP_TITLE_KEYWORDS):
            is_likely_btp = True
        if log_evidence_benign and malicious_count == 0:
            is_likely_btp = True
        indicators_str = str(indicators).lower()
        if any(pattern in indicators_str for pattern in ["it.admin", "svc.", "service_account", "backup", "scanner"]):
            if malicious_count == 0 and not log_evidence_suspicious:
                is_likely_btp = True

        is_likely_fp = False
        if any(keyword in title_lower for keyword in self.FP_TITLE_KEYWORDS):
            is_likely_fp = True
        if alert_severity in ("low", "info") and malicious_count == 0 and not log_evidence_suspicious:
            is_likely_fp = True

        is_likely_tp = (
            malicious_count > 0
            or high_threat_count > 0
            or log_evidence_suspicious
            or (alert_severity in ("critical", "high") and not is_likely_fp and not is_likely_btp)
        )

        if is_likely_btp:
            return "benign_true_positive"
        if is_likely_fp:
            return "false_positive"
        if is_likely_tp:
            return "true_positive"
        return "false_positive"

    def _classification_confidence(self, inv: dict[str, Any]) -> float:
        enrichment_results = inv.get("enriched_indicators", {})
        malicious_count = sum(
            1 for result in enrichment_results.values() if isinstance(result, dict) and result.get("malicious")
        )
        high_threat_count = sum(
            1
            for result in enrichment_results.values()
            if isinstance(result, dict) and result.get("threat_score", 0) >= 70
        )
        if malicious_count > 0:
            return 0.9
        if high_threat_count > 0:
            return 0.8
        return 0.7

    def _infer_technique(self, alert: dict[str, Any]) -> str:
        title = alert.get("title", "").lower()
        source = alert.get("source_system", "").lower()

        if "macro-enabled" in title or "macro" in title:
            return "T1566.001"
        if "lsass" in title or "credential dump" in title:
            return "T1003.001"
        if "rdp" in title or "lateral movement" in title:
            return "T1021.001"
        if "large archive" in title:
            return "T1074.001"
        if "outbound https transfer" in title:
            return "T1041"
        if "suspicious cloud storage download" in title:
            return "T1567"
        if "scheduled task" in title:
            return "T1053.005"
        if "unauthorized database access" in title:
            return "T1078"
        if "bulk data export" in title:
            return "T1530"
        if "cloud upload" in title:
            return "T1567"
        if "vpn login anomaly" in title:
            return "T1133"
        if "service account privilege escalation" in title:
            return "T1098"
        if "configuration changes" in title or "firewall rules modified" in title:
            return "T1562.004"
        if "after-hours badge access" in title:
            return "T1078"
        if "mass file deletion" in title:
            return "T1485"
        if "usb mass storage" in title:
            return "T1052.001"
        if "phish" in title or "macro" in title:
            return "T1566.001"
        if "lsass" in title or "credential" in title or "mimikatz" in title:
            return "T1003.001"
        if "rdp" in title or "lateral" in title:
            return "T1021.001"
        if "staging" in title or "archive" in title or "large" in title.split("file")[0:1]:
            return "T1074.001"
        if "exfil" in title or "outbound" in title or "transfer" in title:
            return "T1041"
        if "brute" in title or "stuffing" in title or "failed login" in title:
            return "T1110.004"
        if "takeover" in title or "impossible travel" in title:
            return "T1078"
        if "scheduled" in title or "persistence" in title or "cron" in title:
            return "T1053.005"
        if "spearphish" in title or "click" in title:
            return "T1566.002"
        if "delet" in title or "wipe" in title or "destruction" in title:
            return "T1485"
        if "usb" in title or "removable" in title:
            return "T1052.001"
        if "insider" in title or "unauthorized" in title or "privilege" in title:
            return "T1078"
        if "vpn" in title:
            return "T1133"
        if "powershell" in title or "script" in title:
            return "T1059.001"
        if "email" in source:
            return "T1566.001"
        if "endpoint" in source or "edr" in source:
            return "T1059.001"
        if "firewall" in source:
            return "T1071.001"
        return "T1566.001"

    def _infer_response_action(self, alert: dict[str, Any], classification: str | None) -> str:
        if classification in {"false_positive", "benign_true_positive"}:
            return "no_action"

        title = alert.get("title", "").lower()
        source = alert.get("source_system", "").lower()
        indicators = alert.get("indicators", {})

        if "phishing email" in title:
            return "isolate_endpoint"
        if "lsass" in title or "credential dump" in title:
            return "reset_password"
        if "rdp lateral movement" in title or "account takeover" in title:
            return "revoke_sessions"
        if "large archive" in title or "scheduled task created" in title:
            return "quarantine_file"
        if "outbound https transfer" in title or "credential stuffing" in title:
            return "block_ip"
        if "spearphishing link clicked" in title or "cloud upload" in title:
            return "block_domain"
        if "suspicious cloud storage download" in title:
            return "revoke_sessions"
        if "unauthorized database access" in title:
            return "disable_account"
        if "bulk data export" in title:
            return "quarantine_file"
        if "vpn login anomaly" in title:
            return "block_ip"
        if "service account privilege escalation" in title:
            return "disable_account"
        if "configuration changes" in title or "firewall rules modified" in title:
            return "isolate_endpoint"
        if "after-hours badge access" in title:
            return "disable_account"
        if "mass file deletion" in title or "usb mass storage" in title:
            return "isolate_endpoint"

        if "credential" in title or "lsass" in title or "password" in title or "brute" in title:
            return "disable_account"
        if "rdp" in title or "lateral" in title or "takeover" in title:
            return "revoke_sessions"
        if "malware" in title or "macro" in title or "archive" in title or "file" in title:
            return "quarantine_file"
        if "exfil" in title or "c2" in title or "outbound" in title or "transfer" in title:
            return "block_ip"
        if "phish" in title:
            return "isolate_endpoint"
        if indicators.get("domain"):
            return "block_domain"
        if indicators.get("ip"):
            return "block_ip"
        if "endpoint" in source or "edr" in source:
            return "isolate_endpoint"
        return "block_ip"

    def _classify_by_title(self, alert: dict[str, Any]) -> str:
        title = alert.get("title", "").lower()

        tp_keywords = {
            "credential stuffing attack detected",
            "possible account takeover",
            "suspicious cloud storage download",
            "spearphishing link clicked",
            "scheduled task created for persistence",
            "unauthorized database access",
            "bulk data export",
            "suspicious cloud upload",
            "vpn login anomaly",
            "service account privilege escalation",
            "critical configuration changes",
            "after-hours badge access",
            "mass file deletion",
            "usb mass storage device",
        }
        btp_keywords = {
            "internal penetration test",
            "psexec remote execution",
            "bulk password reset",
            "scheduled vulnerability scan",
            "ssh key rotation",
            "group policy update",
            "backup server",
            "scheduled job",
            "admin maintenance activity",
            "it maintenance window",
        }

        if any(keyword in title for keyword in btp_keywords):
            return "benign_true_positive"
        if any(keyword in title for keyword in tp_keywords):
            return "true_positive"
        if any(keyword in title for keyword in self.FP_TITLE_KEYWORDS):
            return "false_positive"
        return "false_positive"
