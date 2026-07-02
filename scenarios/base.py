"""
Base Scenario Generator
========================
Provides the BaseScenario class with a seeded RNG and shared data generators
used by all three scenario implementations.

CRITICAL: All randomness uses self.rng = random.Random(seed) — NEVER global random.seed().
This guarantees same seed → same scenario → same grader results.
"""

import random
import string
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from models import (
    AssetInfo, EnrichmentResult,
    IndicatorType, LogEntry, LogSource, ScenarioConfig, UserInfo
)


class BaseScenario(ABC):
    """Abstract base for all SOC scenario generators."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self._base_time = datetime.now(timezone.utc) - timedelta(hours=2)

    @abstractmethod
    def generate(self) -> ScenarioConfig:
        """Generate a complete, self-contained scenario configuration."""
        ...

    # ------------------------------------------------------------------
    # IP Generators
    # ------------------------------------------------------------------

    def _private_ip(self) -> str:
        """Generate a random RFC-1918 internal IP."""
        return f"10.{self.rng.randint(0, 10)}.{self.rng.randint(1, 50)}.{self.rng.randint(1, 254)}"

    def _public_ip(self) -> str:
        """Generate a random public IP (avoids private ranges)."""
        first = self.rng.choice([45, 62, 77, 91, 103, 138, 176, 185, 194, 198, 203, 212, 217])
        return f"{first}.{self.rng.randint(0, 255)}.{self.rng.randint(0, 255)}.{self.rng.randint(1, 254)}"

    def _tor_exit_ip(self) -> str:
        """Generate a known Tor exit node-style IP."""
        prefixes = [
            ("185", "220", "101"),
            ("176", "10", "99"),
            ("62", "102", "147"),
            ("107", "189", "1"),
        ]
        p = self.rng.choice(prefixes)
        return f"{p[0]}.{p[1]}.{p[2]}.{self.rng.randint(1, 254)}"

    # ------------------------------------------------------------------
    # Domain Generators
    # ------------------------------------------------------------------

    def _malicious_domain(self) -> str:
        """Generate a suspicious newly-registered domain."""
        prefixes = ["secure", "update", "cdn", "api", "login", "verify", "account", "support"]
        mid = "".join(self.rng.choices(string.ascii_lowercase + string.digits, k=8))
        tld = self.rng.choice(["com", "net", "io", "xyz", "top", "club"])
        return f"{self.rng.choice(prefixes)}-{mid}.{tld}"

    def _legit_domain(self) -> str:
        """Generate a recognizable legitimate domain."""
        companies = [
            "microsoft.com", "google.com", "amazonaws.com", "cloudflare.com",
            "office365.com", "dropbox.com", "salesforce.com", "okta.com",
            "zoom.us", "github.com", "slack.com", "atlassian.com",
        ]
        return self.rng.choice(companies)

    def _internal_domain(self) -> str:
        """Generate an internal corporate domain."""
        corps = ["acmecorp.local", "corp.internal", "ad.acmecorp.com", "acme.corp"]
        return self.rng.choice(corps)

    # ------------------------------------------------------------------
    # Hash Generators
    # ------------------------------------------------------------------

    def _sha256(self) -> str:
        """Generate a random 64-char hex string (SHA-256 lookalike)."""
        return "".join(self.rng.choices(string.hexdigits.lower()[:16], k=64))

    def _md5(self) -> str:
        """Generate a random 32-char hex string (MD5 lookalike)."""
        return "".join(self.rng.choices(string.hexdigits.lower()[:16], k=32))

    # ------------------------------------------------------------------
    # Name / Account Generators
    # ------------------------------------------------------------------

    def _username(self) -> str:
        first_names = ["john", "jane", "mike", "sarah", "david", "lisa", "tom", "emily",
                       "robert", "anna", "james", "mary", "william", "patricia", "richard"]
        last_names = ["smith", "jones", "williams", "brown", "davis", "miller", "wilson",
                      "moore", "taylor", "anderson", "thomas", "jackson", "white", "harris"]
        return f"{self.rng.choice(first_names)}.{self.rng.choice(last_names)}"

    def _hostname(self, prefix: str = "WORKSTATION") -> str:
        return f"{prefix}-{self.rng.randint(10, 99)}"

    def _department(self) -> str:
        depts = ["Finance", "Engineering", "HR", "Sales", "Marketing", "IT", "Legal", "Operations"]
        return self.rng.choice(depts)

    # ------------------------------------------------------------------
    # Timestamp Generators
    # ------------------------------------------------------------------

    def _timestamp(self, hours_ago: float = 0.0, jitter_minutes: int = 0) -> str:
        """Generate an ISO8601 timestamp relative to the episode base time."""
        t = self._base_time - timedelta(hours=hours_ago)
        if jitter_minutes:
            t += timedelta(minutes=self.rng.randint(-jitter_minutes, jitter_minutes))
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")

    # ------------------------------------------------------------------
    # Alert ID Generators
    # ------------------------------------------------------------------

    def _alert_id(self, prefix: str) -> str:
        suffix = "".join(self.rng.choices(string.ascii_uppercase + string.digits, k=6))
        return f"{prefix}-{suffix}"

    # ------------------------------------------------------------------
    # Database Builders
    # ------------------------------------------------------------------

    def _make_enrichment_result(
        self,
        indicator: str,
        indicator_type: IndicatorType,
        malicious: bool,
        confidence: float,
        threat_score: int,
        threat_type: Optional[str] = None,
        geo: Optional[str] = None,
        tags: Optional[List[str]] = None,
        malware: Optional[List[str]] = None,
        whois: Optional[str] = None,
    ) -> EnrichmentResult:
        return EnrichmentResult(
            indicator=indicator,
            indicator_type=indicator_type,
            malicious=malicious,
            confidence=confidence,
            threat_score=threat_score,
            threat_type=threat_type,
            geo_location=geo,
            whois_info=whois,
            associated_malware=malware or [],
            tags=tags or [],
            source="threat_intel",
        )

    def _make_log_entry(
        self,
        source: LogSource,
        event_type: str,
        hours_ago: float = 0.5,
        src_ip: Optional[str] = None,
        dst_ip: Optional[str] = None,
        user: Optional[str] = None,
        hostname: Optional[str] = None,
        action: Optional[str] = None,
        severity: Optional[str] = None,
        details: Optional[Dict] = None,
    ) -> LogEntry:
        return LogEntry(
            timestamp=self._timestamp(hours_ago=hours_ago),
            source=source,
            event_type=event_type,
            src_ip=src_ip,
            dst_ip=dst_ip,
            user=user,
            hostname=hostname,
            action=action,
            severity=severity,
            details=details or {},
        )

    def _make_asset(
        self,
        hostname: str,
        asset_type: str,
        owner: str,
        department: str,
        ip: str,
        criticality: str = "medium",
        os: str = "Windows 10",
    ) -> AssetInfo:
        return AssetInfo(
            asset_id=f"AST-{hostname}",
            hostname=hostname,
            asset_type=asset_type,
            criticality=criticality,
            owner=owner,
            department=department,
            ip_address=ip,
            os=os,
            patch_status="current" if self.rng.random() > 0.3 else "behind",
            last_scan=self._timestamp(hours_ago=self.rng.randint(24, 168)),
            open_vulnerabilities=self.rng.randint(0, 5),
            recent_activity_summary=f"Normal activity for {department} workstation",
        )

    def _make_user(
        self,
        username: str,
        role: str,
        department: str,
        risk_score: float = 0.1,
        is_privileged: bool = False,
        access_level: str = "standard",
    ) -> UserInfo:
        parts = username.split(".")
        display = f"{parts[0].capitalize()} {parts[1].capitalize()}" if len(parts) >= 2 else username.capitalize()
        return UserInfo(
            user_id=f"USR-{username}",
            username=username,
            display_name=display,
            email=f"{username}@acmecorp.com",
            role=role,
            department=department,
            access_level=access_level,
            is_privileged=is_privileged,
            manager=self._username() if not is_privileged else None,
            last_login=self._timestamp(hours_ago=self.rng.uniform(0.1, 24)),
            login_anomaly_score=risk_score * 0.5,
            risk_score=risk_score,
            recent_actions=[],
        )

    def _empty_log_db(self, alert_ids: List[str]) -> Dict[str, Dict[str, List[LogEntry]]]:
        """Create an empty log DB structure for all sources and alert IDs."""
        db: Dict[str, Dict[str, List[LogEntry]]] = {}
        for source in LogSource:
            db[source.value] = {aid: [] for aid in alert_ids}
        return db
