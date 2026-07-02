"""Red-Team scenario generator for SOC-Triage-Gym v2 self-improvement curriculum."""
import hashlib
import random
import string
import uuid
from datetime import UTC, datetime, timedelta

from models import (
    AlertClassification,
    AlertMeta,
    AlertSeverity,
    AssetInfo,
    EnrichmentResult,
    GroundTruth,
    IndicatorType,
    LogEntry,
    LogSource,
    RedTeamConfig,
    ResponseActionType,
    ScenarioConfig,
    UserInfo,
)


class RedTeamGenerator:
    """
    Synthesizes adversarial SOC scenarios for self-play curriculum training.

    Fully deterministic: same RedTeamConfig + seed → same ScenarioConfig.
    All randomness flows through self._rng = random.Random(seed).
    """

    MAX_STEPS = 30

    def __init__(self, config: RedTeamConfig = None, seed: int = 42) -> None:
        self.config = config if config is not None else RedTeamConfig()
        self.seed = seed
        self._rng = random.Random(seed)
        self._base_time = datetime.now(UTC) - timedelta(hours=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> ScenarioConfig:
        """Generate a complete adversarial ScenarioConfig."""
        diff = self.config.difficulty_floor

        # Select attack pattern weighted toward patterns that match difficulty
        patterns = self.config.attack_patterns or ["phishing", "lateral_movement", "insider_threat"]

        if diff < 0.4:
            # Low difficulty: prefer phishing-style (1-2 TP alerts)
            preferred = [p for p in patterns if p == "phishing"] or patterns
            pattern = self._rng.choice(preferred)
            num_tp = self._rng.randint(1, 2)
        elif diff <= 0.7:
            # Medium difficulty: prefer lateral_movement (3-5 TP alerts)
            preferred = [p for p in patterns if p == "lateral_movement"] or patterns
            pattern = self._rng.choice(preferred)
            num_tp = self._rng.randint(3, 5)
        else:
            # High difficulty: prefer insider_threat (6-10 TP alerts)
            preferred = [p for p in patterns if p == "insider_threat"] or patterns
            pattern = self._rng.choice(preferred)
            num_tp = self._rng.randint(6, 10)

        # Calculate FP count from noise_density
        # noise_density = fp_count / (fp_count + tp_count)  =>  fp = tp * density / (1 - density)
        density = max(0.0, min(self.config.noise_density, 0.95))
        num_fp = max(0, round(num_tp * density / max(1 - density, 0.05)))

        tp_alerts = self._generate_attack_alerts(pattern, num_tp)
        fp_alerts = self._generate_noise_alerts(num_fp, shared_ips=[
            ip
            for a in tp_alerts
            for ip in a.indicators.get("ip", [])
        ])

        all_alerts = tp_alerts + fp_alerts
        self._rng.shuffle(all_alerts)

        enrichment_db = self._build_enrichment_db(all_alerts, self.config.ioc_freshness)
        log_db = self._build_log_db(all_alerts)
        asset_db, user_db = self._build_asset_and_user_db(all_alerts)
        ground_truth = self._build_ground_truth(tp_alerts, fp_alerts)

        scenario_id = f"red_team_{pattern}_{self.seed}"
        description = (
            f"Red-Team generated scenario: {pattern.replace('_', ' ')} attack pattern. "
            f"{num_tp} TP alert(s), {num_fp} FP/noise alert(s). "
            f"difficulty={diff:.2f}, noise={self.config.noise_density:.2f}, "
            f"ioc_freshness={self.config.ioc_freshness:.2f}, "
            f"obfuscation={self.config.correlation_obfuscation:.2f}."
        )

        return ScenarioConfig(
            scenario_id=scenario_id,
            task_id="red_team_generated",
            seed=self.seed,
            description=description,
            max_steps=self.MAX_STEPS,
            alerts=all_alerts,
            enrichment_db=enrichment_db,
            log_db=log_db,
            asset_db=asset_db,
            user_db=user_db,
            ground_truth=ground_truth,
            difficulty_floor=self.config.difficulty_floor,
            noise_density=self.config.noise_density,
            ioc_freshness=self.config.ioc_freshness,
            correlation_obfuscation=self.config.correlation_obfuscation,
        )

    def generate_fingerprint(self) -> str:
        """Return a 16-char SHA-256 fingerprint of this config for novelty scoring."""
        return hashlib.sha256(self.config.model_dump_json().encode()).hexdigest()[:16]

    def adapt_difficulty(self, blue_win_rate: float) -> "RedTeamGenerator":
        """
        Return a new generator with curriculum-adjusted difficulty.

        - blue_win_rate >= 0.75: increase difficulty_floor by 0.1 (cap 0.95)
        - blue_win_rate <= 0.45: decrease difficulty_floor by 0.1 (floor 0.05)
        - otherwise: unchanged
        """
        new_floor = self.config.difficulty_floor
        if blue_win_rate >= 0.75:
            new_floor = min(0.95, new_floor + 0.1)
        elif blue_win_rate <= 0.45:
            new_floor = max(0.05, new_floor - 0.1)

        new_config = RedTeamConfig(
            difficulty_floor=new_floor,
            attack_patterns=list(self.config.attack_patterns),
            noise_density=self.config.noise_density,
            ioc_freshness=self.config.ioc_freshness,
            correlation_obfuscation=self.config.correlation_obfuscation,
            blue_team_win_rate=blue_win_rate,
            episode_count=self.config.episode_count,
        )
        return RedTeamGenerator(new_config, seed=self.seed + 1)

    # ------------------------------------------------------------------
    # Internal: Alert Generators
    # ------------------------------------------------------------------

    def _generate_attack_alerts(self, pattern: str, num_tp: int) -> list[AlertMeta]:
        """Create num_tp true-positive alerts for the given attack pattern."""
        alerts: list[AlertMeta] = []

        if pattern == "phishing":
            alerts = self._phishing_alerts(num_tp)
        elif pattern == "lateral_movement":
            alerts = self._lateral_movement_alerts(num_tp)
        else:  # insider_threat
            alerts = self._insider_threat_alerts(num_tp)

        return alerts

    def _phishing_alerts(self, count: int) -> list[AlertMeta]:
        alerts = []
        for i in range(count):
            alert_id = f"ALT-RT-{str(uuid.UUID(int=self._rng.getrandbits(128)))[:6].upper()}"
            attacker_ip = self._public_ip()
            malicious_domain = self._malicious_domain()
            file_hash = self._sha256()
            victim_user = self._username()
            sender = f"billing@{self._malicious_domain()}"

            hours_ago = 2.0 - i * 0.3
            alerts.append(AlertMeta(
                alert_id=alert_id,
                title="Suspicious Email with Malicious Link",
                description=(
                    f"Email security gateway flagged an inbound email from {sender} "
                    f"containing a malicious link pointing to {malicious_domain}. "
                    f"SPF/DKIM failed. Recipient: {victim_user}@acmecorp.com."
                ),
                severity=AlertSeverity.HIGH,
                source_system="Email Security Gateway",
                timestamp=self._timestamp(hours_ago=hours_ago),
                rule_triggered=f"PHISH_LINK_{i + 1:03d}",
                indicators={
                    "ip": [attacker_ip],
                    "domain": [malicious_domain],
                    "file_hash": [file_hash],
                    "email": [sender],
                },
                raw_log_snippet=(
                    f"FROM={sender} TO={victim_user}@acmecorp.com "
                    f"URL=http://{malicious_domain}/payload "
                    f"SPF=fail DKIM=fail DMARC=fail"
                ),
            ))
        return alerts

    def _lateral_movement_alerts(self, count: int) -> list[AlertMeta]:
        """Chain of escalating SMB/RDP lateral movement alerts."""
        alerts = []
        base_src_ip = self._private_ip()
        severities = [
            AlertSeverity.LOW,
            AlertSeverity.MEDIUM,
            AlertSeverity.HIGH,
            AlertSeverity.HIGH,
            AlertSeverity.CRITICAL,
        ]
        protocols = ["SMB", "RDP", "SMB", "RDP", "WMI"]
        for i in range(count):
            alert_id = f"ALT-RT-{str(uuid.UUID(int=self._rng.getrandbits(128)))[:6].upper()}"
            dst_ip = self._private_ip()
            src_user = self._username()
            protocol = protocols[i % len(protocols)]
            severity = severities[min(i, len(severities) - 1)]

            hours_ago = 3.0 - i * 0.4
            alerts.append(AlertMeta(
                alert_id=alert_id,
                title=f"Lateral Movement Detected via {protocol}",
                description=(
                    f"Anomalous {protocol} connection from {base_src_ip} to {dst_ip}. "
                    f"User {src_user} accessed multiple hosts in a short time window. "
                    f"Step {i + 1} in potential lateral movement chain."
                ),
                severity=severity,
                source_system="EDR / Network IDS",
                timestamp=self._timestamp(hours_ago=hours_ago),
                rule_triggered=f"LATERAL_{protocol}_{i + 1:03d}",
                indicators={
                    "ip": [base_src_ip, dst_ip],
                    "domain": [self._internal_domain()],
                    "file_hash": [self._sha256()] if protocol == "SMB" else [],
                },
                raw_log_snippet=(
                    f"src={base_src_ip} dst={dst_ip} proto={protocol} "
                    f"user={src_user} action=connect status=success"
                ),
            ))
        return alerts

    def _insider_threat_alerts(self, count: int) -> list[AlertMeta]:
        """Data access anomaly alerts consistent with insider threat."""
        alerts = []
        insider_user = self._username()
        insider_ip = self._private_ip()
        cloud_domain = self._malicious_domain()
        for i in range(count):
            alert_id = f"ALT-RT-{str(uuid.UUID(int=self._rng.getrandbits(128)))[:6].upper()}"
            hours_ago = 4.0 - i * 0.35
            resource = self._rng.choice(["S3_bucket", "SharePoint_Library", "OneDrive_root", "HR_database"])
            mb_transferred = self._rng.randint(50, 2000)

            alerts.append(AlertMeta(
                alert_id=alert_id,
                title=f"Anomalous Data Access — {resource.replace('_', ' ')}",
                description=(
                    f"User {insider_user} accessed {resource} at an unusual hour "
                    f"and transferred {mb_transferred} MB. "
                    f"Destination includes external cloud domain {cloud_domain}. "
                    f"Access pattern deviates significantly from baseline."
                ),
                severity=AlertSeverity.MEDIUM,
                source_system="UEBA / DLP",
                timestamp=self._timestamp(hours_ago=hours_ago),
                rule_triggered=f"INSIDER_DATA_ACCESS_{i + 1:03d}",
                indicators={
                    "ip": [insider_ip],
                    "domain": [cloud_domain],
                    "file_hash": [],
                },
                raw_log_snippet=(
                    f"user={insider_user} src_ip={insider_ip} "
                    f"resource={resource} bytes_out={mb_transferred * 1024 * 1024} "
                    f"dst_domain={cloud_domain} time=03:{self._rng.randint(0, 59):02d}:00"
                ),
            ))
        return alerts

    def _generate_noise_alerts(
        self,
        num_fp: int,
        shared_ips: list[str] | None = None,
    ) -> list[AlertMeta]:
        """
        Create benign FP noise alerts.
        When correlation_obfuscation is high, some FPs share IPs with real alerts.
        """
        shared_ips = shared_ips or []
        noise_templates = [
            ("DNS Query Volume Spike", "DNS resolver flagged elevated query count from workstation.", "DNS_VOLUME_001", AlertSeverity.INFO),
            ("Routine Admin Login Outside Hours", "Admin account authenticated outside business hours.", "AUTH_AFTER_HOURS_002", AlertSeverity.LOW),
            ("Port Scan Detected — Internal", "Internal host performed sequential port scan (routine vulnerability assessment).", "PORTSCAN_INTERNAL_003", AlertSeverity.LOW),
            ("File Compression Utility Execution", "7zip executed on workstation during backup window.", "COMPRESS_EXEC_004", AlertSeverity.INFO),
            ("Large Email Attachment Sent", "Outbound email with attachment exceeding 10 MB size limit.", "EMAIL_SIZE_005", AlertSeverity.LOW),
            ("Cloud Storage Access — Authorized", "SharePoint access from known corporate IP during work hours.", "CLOUD_ACCESS_006", AlertSeverity.INFO),
        ]

        alerts = []
        for i in range(num_fp):
            template = noise_templates[i % len(noise_templates)]
            title, description, rule, severity = template
            alert_id = f"ALT-RT-{str(uuid.UUID(int=self._rng.getrandbits(128)))[:6].upper()}"
            hours_ago = self._rng.uniform(0.5, 5.0)
            fp_ip = self._private_ip()

            # Correlation obfuscation: occasionally inject a shared IP from real alerts
            inject_shared = (
                shared_ips
                and self.config.correlation_obfuscation > 0.5
                and self._rng.random() < self.config.correlation_obfuscation - 0.5
            )
            ip_list = ([self._rng.choice(shared_ips)] if inject_shared else [fp_ip])

            alerts.append(AlertMeta(
                alert_id=alert_id,
                title=title,
                description=description,
                severity=severity,
                source_system="SIEM",
                timestamp=self._timestamp(hours_ago=hours_ago),
                rule_triggered=rule,
                indicators={
                    "ip": ip_list,
                    "domain": [],
                    "file_hash": [],
                },
                raw_log_snippet=f"src={ip_list[0]} event={rule} status=observed",
            ))
        return alerts

    # ------------------------------------------------------------------
    # Internal: Enrichment DB
    # ------------------------------------------------------------------

    def _build_enrichment_db(
        self,
        alerts: list[AlertMeta],
        ioc_freshness: float,
    ) -> dict[str, EnrichmentResult]:
        """
        Build a threat-intel database for all indicators in all alerts.

        TP alert indicators are marked malicious=True.
        FP alert indicators are marked malicious=False.
        ioc_freshness controls confidence for malicious indicators:
          - freshness >= 0.5: confidence = freshness * 0.9  (fresh, easy to detect)
          - freshness <  0.5: confidence = 0.3-0.5           (stale, harder to detect)
        """
        db: dict[str, EnrichmentResult] = {}

        # Classify each alert's maliciousness by checking severity
        # TP alerts have severity >= MEDIUM and are not labelled as routine rules
        # We rely on title patterns to distinguish TP vs FP indicators
        fp_titles = {
            "DNS Query Volume Spike",
            "Routine Admin Login Outside Hours",
            "Port Scan Detected — Internal",
            "File Compression Utility Execution",
            "Large Email Attachment Sent",
            "Cloud Storage Access — Authorized",
        }

        for alert in alerts:
            is_fp = alert.title in fp_titles

            for ip in alert.indicators.get("ip", []):
                if ip in db:
                    continue
                if is_fp:
                    db[ip] = EnrichmentResult(
                        indicator=ip,
                        indicator_type=IndicatorType.IP,
                        malicious=False,
                        confidence=0.9,
                        threat_score=0,
                        tags=["internal", "clean"],
                        source="threat_intel",
                    )
                else:
                    confidence = self._fresh_confidence(ioc_freshness)
                    db[ip] = EnrichmentResult(
                        indicator=ip,
                        indicator_type=IndicatorType.IP,
                        malicious=True,
                        confidence=confidence,
                        threat_score=int(confidence * 95),
                        threat_type="command-and-control",
                        tags=["malicious", "c2"],
                        source="threat_intel",
                    )

            for domain in alert.indicators.get("domain", []):
                if domain in db:
                    continue
                if is_fp:
                    db[domain] = EnrichmentResult(
                        indicator=domain,
                        indicator_type=IndicatorType.DOMAIN,
                        malicious=False,
                        confidence=0.95,
                        threat_score=0,
                        tags=["legitimate"],
                        source="threat_intel",
                    )
                else:
                    confidence = self._fresh_confidence(ioc_freshness)
                    db[domain] = EnrichmentResult(
                        indicator=domain,
                        indicator_type=IndicatorType.DOMAIN,
                        malicious=True,
                        confidence=confidence,
                        threat_score=int(confidence * 90),
                        threat_type="phishing",
                        tags=["malicious", "newly-registered"],
                        source="threat_intel",
                    )

            for file_hash in alert.indicators.get("file_hash", []):
                if not file_hash or file_hash in db:
                    continue
                if is_fp:
                    db[file_hash] = EnrichmentResult(
                        indicator=file_hash,
                        indicator_type=IndicatorType.FILE_HASH,
                        malicious=False,
                        confidence=0.9,
                        threat_score=0,
                        tags=["clean"],
                        source="threat_intel",
                    )
                else:
                    confidence = self._fresh_confidence(ioc_freshness)
                    db[file_hash] = EnrichmentResult(
                        indicator=file_hash,
                        indicator_type=IndicatorType.FILE_HASH,
                        malicious=True,
                        confidence=confidence,
                        threat_score=int(confidence * 98),
                        threat_type="malware",
                        tags=["trojan", "dropper"],
                        source="threat_intel",
                    )

            for email in alert.indicators.get("email", []):
                if email in db:
                    continue
                if is_fp:
                    db[email] = EnrichmentResult(
                        indicator=email,
                        indicator_type=IndicatorType.EMAIL,
                        malicious=False,
                        confidence=0.92,
                        threat_score=0,
                        tags=["legitimate-sender"],
                        source="threat_intel",
                    )
                else:
                    confidence = self._fresh_confidence(ioc_freshness)
                    db[email] = EnrichmentResult(
                        indicator=email,
                        indicator_type=IndicatorType.EMAIL,
                        malicious=True,
                        confidence=confidence,
                        threat_score=int(confidence * 85),
                        threat_type="phishing",
                        tags=["phishing-sender"],
                        source="threat_intel",
                    )

        return db

    def _fresh_confidence(self, ioc_freshness: float) -> float:
        """Convert ioc_freshness to a per-indicator confidence value."""
        if ioc_freshness >= 0.5:
            return round(ioc_freshness * 0.9, 3)
        else:
            # Stale data: low, variable confidence — harder to detect
            return round(self._rng.uniform(0.3, 0.5), 3)

    # ------------------------------------------------------------------
    # Internal: Log DB
    # ------------------------------------------------------------------

    def _build_log_db(
        self, alerts: list[AlertMeta]
    ) -> dict[str, dict[str, list[LogEntry]]]:
        """
        Generate minimal log entries for every alert.
        TP alerts get suspicious log entries; FP alerts get benign entries.
        """
        fp_titles = {
            "DNS Query Volume Spike",
            "Routine Admin Login Outside Hours",
            "Port Scan Detected — Internal",
            "File Compression Utility Execution",
            "Large Email Attachment Sent",
            "Cloud Storage Access — Authorized",
        }

        all_ids = [a.alert_id for a in alerts]
        log_db: dict[str, dict[str, list[LogEntry]]] = {}
        for source in LogSource:
            log_db[source.value] = {aid: [] for aid in all_ids}

        for alert in alerts:
            aid = alert.alert_id
            is_fp = alert.title in fp_titles
            src_ip = (alert.indicators.get("ip") or [None])[0]
            hours_ago = self._rng.uniform(0.2, 1.5)

            if is_fp:
                # Benign firewall entry
                log_db[LogSource.FIREWALL.value][aid].append(
                    self._make_log_entry(
                        LogSource.FIREWALL, "outbound_connection",
                        hours_ago=hours_ago, src_ip=src_ip,
                        action="allowed", severity="info",
                        details={"dst_port": 443, "protocol": "TCP", "bytes_sent": 512, "direction": "outbound"},
                    )
                )
                # Benign endpoint entry
                log_db[LogSource.ENDPOINT.value][aid].append(
                    self._make_log_entry(
                        LogSource.ENDPOINT, "process_created",
                        hours_ago=hours_ago, src_ip=src_ip,
                        action="execute", severity="info",
                        details={"process_name": "explorer.exe", "parent_process": "winlogon.exe"},
                    )
                )
            else:
                # Suspicious firewall entry
                dst_ip = self._public_ip()
                log_db[LogSource.FIREWALL.value][aid].append(
                    self._make_log_entry(
                        LogSource.FIREWALL, "outbound_connection",
                        hours_ago=hours_ago, src_ip=src_ip, dst_ip=dst_ip,
                        action="allowed", severity="high",
                        details={"dst_port": 443, "protocol": "TCP", "bytes_sent": 4096, "bytes_recv": 28672, "direction": "outbound"},
                    )
                )
                # Suspicious endpoint entry
                log_db[LogSource.ENDPOINT.value][aid].append(
                    self._make_log_entry(
                        LogSource.ENDPOINT, "process_created",
                        hours_ago=hours_ago, src_ip=src_ip,
                        action="execute", severity="high",
                        details={
                            "process_name": "cmd.exe",
                            "parent_process": "suspicious_payload.exe",
                            "command_line": "cmd.exe /c whoami && net user",
                        },
                    )
                )
                # IDS signature match for TP
                log_db[LogSource.IDS.value][aid].append(
                    self._make_log_entry(
                        LogSource.IDS, "signature_match",
                        hours_ago=hours_ago, src_ip=src_ip, dst_ip=dst_ip,
                        severity="critical",
                        details={
                            "signature": "MALWARE-CNC suspicious beacon",
                            "category": "Malware-CnC",
                            "priority": 1,
                        },
                    )
                )

        return log_db

    # ------------------------------------------------------------------
    # Internal: Asset & User DB
    # ------------------------------------------------------------------

    def _build_asset_and_user_db(
        self, alerts: list[AlertMeta]
    ) -> tuple:
        """Build one workstation asset + one user per alert."""
        asset_db: dict[str, AssetInfo] = {}
        user_db: dict[str, UserInfo] = {}

        for alert in alerts:
            hostname = self._hostname("WORKSTATION")
            username = self._username()
            ip = (alert.indicators.get("ip") or [self._private_ip()])[0]
            dept = self._department()

            if hostname not in asset_db:
                asset_db[hostname] = AssetInfo(
                    asset_id=f"AST-{hostname}",
                    hostname=hostname,
                    asset_type="workstation",
                    criticality="medium",
                    owner=username,
                    department=dept,
                    ip_address=ip,
                    os="Windows 10",
                    patch_status="current" if self._rng.random() > 0.3 else "behind",
                    last_scan=self._timestamp(hours_ago=self._rng.randint(24, 168)),
                    open_vulnerabilities=self._rng.randint(0, 5),
                    recent_activity_summary=f"Activity on {hostname} in {dept}",
                )

            if username not in user_db:
                parts = username.split(".")
                display = (
                    f"{parts[0].capitalize()} {parts[1].capitalize()}"
                    if len(parts) >= 2
                    else username.capitalize()
                )
                user_db[username] = UserInfo(
                    user_id=f"USR-{username}",
                    username=username,
                    display_name=display,
                    email=f"{username}@acmecorp.com",
                    role=self._rng.choice(["Analyst", "Engineer", "HR Coordinator", "Sales Rep", "Finance Manager"]),
                    department=dept,
                    access_level="standard",
                    is_privileged=False,
                    manager=self._username(),
                    last_login=self._timestamp(hours_ago=self._rng.uniform(0.1, 24)),
                    login_anomaly_score=self._rng.uniform(0.0, 0.3),
                    risk_score=self._rng.uniform(0.05, 0.4),
                    recent_actions=[],
                )

        return asset_db, user_db

    # ------------------------------------------------------------------
    # Internal: Ground Truth
    # ------------------------------------------------------------------

    def _build_ground_truth(
        self,
        tp_alerts: list[AlertMeta],
        fp_alerts: list[AlertMeta],
    ) -> GroundTruth:
        """
        Build the answer key.

        Techniques are inferred from alert titles.
        required_escalations: all TP IDs with severity >= HIGH.
        required_containments: {tp_id: ["isolate_host"]} for HIGH/CRITICAL TPs.
        expected_manager_flags: TP IDs where correlation_obfuscation > 0.5
            (these are alerts Tier-1 is likely to miss due to shared noise IPs).
        """
        alert_classifications: dict[str, AlertClassification] = {}
        expected_techniques: dict[str, list[str]] = {}
        expected_response_actions: dict[str, list[ResponseActionType]] = {}
        relevant_log_sources: dict[str, list[LogSource]] = {}
        relevant_indicators: dict[str, list[str]] = {}
        required_escalations: list[str] = []
        required_containments: dict[str, list[str]] = {}
        expected_manager_flags: list[str] = []

        high_severities = {AlertSeverity.HIGH, AlertSeverity.CRITICAL}

        for alert in tp_alerts:
            aid = alert.alert_id
            alert_classifications[aid] = AlertClassification.TRUE_POSITIVE

            # Derive MITRE techniques from title
            if "phishing" in alert.title.lower() or "email" in alert.title.lower():
                expected_techniques[aid] = ["T1566.001"]
            elif "lateral" in alert.title.lower() or "smb" in alert.title.lower() or "rdp" in alert.title.lower():
                expected_techniques[aid] = ["T1021.002", "T1078"]
            else:
                expected_techniques[aid] = ["T1078", "T1530"]

            expected_response_actions[aid] = [
                ResponseActionType.ISOLATE_ENDPOINT,
                ResponseActionType.BLOCK_IP,
            ]
            relevant_log_sources[aid] = [LogSource.FIREWALL, LogSource.ENDPOINT, LogSource.IDS]
            all_iocs = (
                alert.indicators.get("ip", [])
                + alert.indicators.get("domain", [])
                + alert.indicators.get("file_hash", [])
                + alert.indicators.get("email", [])
            )
            relevant_indicators[aid] = [ioc for ioc in all_iocs if ioc]

            if alert.severity in high_severities:
                required_escalations.append(aid)
                required_containments[aid] = ["isolate_host"]

            if self.config.correlation_obfuscation > 0.5:
                expected_manager_flags.append(aid)

        for alert in fp_alerts:
            aid = alert.alert_id
            alert_classifications[aid] = AlertClassification.FALSE_POSITIVE
            expected_response_actions[aid] = [ResponseActionType.NO_ACTION]
            relevant_log_sources[aid] = [LogSource.FIREWALL]
            relevant_indicators[aid] = alert.indicators.get("ip", [])

        return GroundTruth(
            alert_classifications=alert_classifications,
            true_positive_ids=[a.alert_id for a in tp_alerts],
            false_positive_ids=[a.alert_id for a in fp_alerts],
            benign_tp_ids=[],
            expected_techniques=expected_techniques,
            expected_response_actions=expected_response_actions,
            relevant_log_sources=relevant_log_sources,
            relevant_indicators=relevant_indicators,
            required_escalations=required_escalations,
            required_containments=required_containments,
            expected_manager_flags=expected_manager_flags,
        )

    # ------------------------------------------------------------------
    # Internal: Primitive Generators  (mirror BaseScenario helpers)
    # ------------------------------------------------------------------

    def _private_ip(self) -> str:
        return f"192.168.{self._rng.randint(0, 254)}.{self._rng.randint(1, 254)}"

    def _public_ip(self) -> str:
        first = self._rng.choice([45, 62, 77, 91, 103, 138, 176, 185, 194, 198, 203, 212, 217])
        return f"{first}.{self._rng.randint(0, 255)}.{self._rng.randint(0, 255)}.{self._rng.randint(1, 254)}"

    def _malicious_domain(self) -> str:
        num = self._rng.randint(1000, 9999)
        return f"malicious-{num}.evil.com"

    def _internal_domain(self) -> str:
        corps = ["acmecorp.local", "corp.internal", "ad.acmecorp.com"]
        return self._rng.choice(corps)

    def _sha256(self) -> str:
        return "".join(self._rng.choices(string.hexdigits.lower()[:16], k=64))

    def _username(self) -> str:
        first_names = ["john", "jane", "mike", "sarah", "david", "lisa", "tom", "emily",
                       "robert", "anna", "james", "mary", "william", "patricia", "richard"]
        last_names = ["smith", "jones", "williams", "brown", "davis", "miller", "wilson",
                      "moore", "taylor", "anderson", "thomas", "jackson", "white", "harris"]
        return f"{self._rng.choice(first_names)}.{self._rng.choice(last_names)}"

    def _hostname(self, prefix: str = "WORKSTATION") -> str:
        return f"{prefix}-{self._rng.randint(10, 99)}"

    def _department(self) -> str:
        depts = ["Finance", "Engineering", "HR", "Sales", "Marketing", "IT", "Legal", "Operations"]
        return self._rng.choice(depts)

    def _timestamp(self, hours_ago: float = 0.0) -> str:
        t = self._base_time - timedelta(hours=hours_ago)
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _make_log_entry(
        self,
        source: LogSource,
        event_type: str,
        hours_ago: float = 0.5,
        src_ip: str | None = None,
        dst_ip: str | None = None,
        user: str | None = None,
        hostname: str | None = None,
        action: str | None = None,
        severity: str | None = None,
        details: dict | None = None,
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
