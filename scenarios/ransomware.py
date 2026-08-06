"""
Ransomware Scenario Generator — single-alert triage
===================================================
Generates one endpoint ransomware alert with full supporting evidence.

Two variants depending on seed parity:
  - TRUE_POSITIVE: unsigned binary, mass file encryption, shadow-copy deletion
                   (vssadmin), ransom note dropped, C2 beacon to a malicious host.
  - FALSE_POSITIVE: signed backup/encryption agent doing scheduled high-volume
                    reads/writes; no shadow-copy deletion, no ransom note, clean IOCs.

Mirrors PhishingScenario's TP/FP shape so it plugs into the same graders. All
randomness uses self.rng (from BaseScenario) — same seed → same scenario.
"""

from models import (
    AlertClassification,
    AlertMeta,
    AlertSeverity,
    GroundTruth,
    IndicatorType,
    LogSource,
    ResponseActionType,
    ScenarioConfig,
)
from scenarios.base import BaseScenario


class RansomwareScenario(BaseScenario):
    """Easy/medium task: single endpoint ransomware alert triage."""

    MAX_STEPS = 15

    def generate(self) -> ScenarioConfig:
        is_tp = (self.seed % 2 == 0) or (self.seed % 7 < 4)
        return self._generate_true_positive() if is_tp else self._generate_false_positive()

    # ------------------------------------------------------------------
    # True Positive Variant — real ransomware detonation
    # ------------------------------------------------------------------

    def _generate_true_positive(self) -> ScenarioConfig:
        alert_id = self._alert_id("RANSOM")

        payload_hash = self._sha256()
        c2_domain = self._malicious_domain()
        c2_ip = self._public_ip()
        victim_username = self._username()
        victim_host = self._hostname("WORKSTATION")
        victim_ip = self._private_ip()
        family = self.rng.choice(["LockBit", "BlackCat", "Akira", "Royal"])
        note_name = self.rng.choice(["HOW_TO_RESTORE_FILES.txt", "readme_decrypt.txt", "RESTORE-MY-FILES.hta"])

        alert = AlertMeta(
            alert_id=alert_id,
            title="Ransomware Behavior — Mass File Encryption and Shadow Copy Deletion",
            description=(
                f"EDR detected a high-rate file-modification burst on {victim_host} "
                f"(user {victim_username}) alongside deletion of Volume Shadow Copies "
                f"and creation of a ransom note '{note_name}'. Suspected {family}."
            ),
            severity=AlertSeverity.CRITICAL,
            source_system="EDR",
            timestamp=self._timestamp(hours_ago=0.6),
            rule_triggered="RANSOM_MASS_ENCRYPT_VSS_DELETE_001",
            indicators={
                "ip": [c2_ip],
                "domain": [c2_domain],
                "file_hash": [payload_hash],
            },
            raw_log_snippet=(
                f"HOST={victim_host} USER={victim_username} PROC=svchost32.exe "
                f"HASH={payload_hash[:16]}... FILES_MODIFIED=4127 EXT=.{family.lower()} "
                f"VSSADMIN=delete_shadows NOTE={note_name}"
            ),
        )

        enrichment_db = {
            payload_hash: self._make_enrichment_result(
                payload_hash, IndicatorType.FILE_HASH, malicious=True,
                confidence=0.98, threat_score=98,
                threat_type="ransomware",
                tags=["ransomware", family.lower(), "unsigned", "dropper"],
                malware=[family],
            ),
            c2_domain: self._make_enrichment_result(
                c2_domain, IndicatorType.DOMAIN, malicious=True,
                confidence=0.94, threat_score=91,
                threat_type="command-and-control",
                geo="Russia",
                tags=["c2", "ransomware-c2", family.lower()],
                malware=[family],
                whois="Registered 6 days ago via anonymous registrar.",
            ),
            c2_ip: self._make_enrichment_result(
                c2_ip, IndicatorType.IP, malicious=True,
                confidence=0.93, threat_score=90,
                threat_type="command-and-control",
                geo="Russia",
                tags=["c2", "bulletproof-hosting", family.lower()],
                malware=[family],
            ),
        }

        log_db = self._empty_log_db([alert_id])

        # Endpoint — unsigned payload, shadow-copy deletion, mass encryption, ransom note.
        log_db[LogSource.ENDPOINT.value][alert_id] = [
            self._make_log_entry(
                LogSource.ENDPOINT, "process_created", hours_ago=0.8,
                hostname=victim_host, src_ip=victim_ip, user=victim_username,
                action="execute", severity="critical",
                details={
                    "process_name": "svchost32.exe",
                    "process_path": f"C:\\Users\\{victim_username}\\AppData\\Local\\Temp\\svchost32.exe",
                    "parent_process": "explorer.exe",
                    "hash_sha256": payload_hash,
                    "signed": False,
                    "hostname": victim_host,
                },
            ),
            self._make_log_entry(
                LogSource.ENDPOINT, "process_created", hours_ago=0.75,
                hostname=victim_host, user=victim_username, action="execute",
                severity="critical",
                details={
                    "process_name": "vssadmin.exe",
                    "parent_process": "svchost32.exe",
                    "command_line": "vssadmin.exe delete shadows /all /quiet",
                    "hostname": victim_host,
                },
            ),
            self._make_log_entry(
                LogSource.ENDPOINT, "file_modification_burst", hours_ago=0.7,
                hostname=victim_host, user=victim_username, action="encrypt",
                severity="critical",
                details={
                    "files_modified": 4127,
                    "new_extension": f".{family.lower()}",
                    "ransom_note": note_name,
                    "rate_files_per_min": 1830,
                    "hostname": victim_host,
                },
            ),
        ]

        # DNS + Firewall — C2 beacon.
        log_db[LogSource.DNS.value][alert_id] = [
            self._make_log_entry(
                LogSource.DNS, "dns_query", hours_ago=0.78,
                hostname=victim_host, src_ip=victim_ip,
                details={"query": c2_domain, "response": c2_ip, "record_type": "A"},
            ),
        ]
        log_db[LogSource.FIREWALL.value][alert_id] = [
            self._make_log_entry(
                LogSource.FIREWALL, "outbound_connection", hours_ago=0.76,
                src_ip=victim_ip, dst_ip=c2_ip, action="allowed", severity="high",
                details={"dst_port": 443, "protocol": "TCP", "bytes_sent": 12800,
                         "direction": "outbound"},
            ),
        ]
        log_db[LogSource.IDS.value][alert_id] = [
            self._make_log_entry(
                LogSource.IDS, "signature_match", hours_ago=0.77,
                src_ip=victim_ip, dst_ip=c2_ip, severity="critical",
                details={"signature": f"MALWARE-CNC {family} ransomware checkin",
                         "category": "Malware-CnC", "priority": 1},
            ),
        ]

        asset_db = {victim_host: self._make_asset(
            victim_host, "workstation", victim_username, self._department(),
            victim_ip, criticality="high",
        )}
        user_db = {victim_username: self._make_user(
            victim_username, "Finance Analyst", "Finance", risk_score=0.2,
        )}

        ground_truth = GroundTruth(
            alert_classifications={alert_id: AlertClassification.TRUE_POSITIVE},
            true_positive_ids=[alert_id],
            false_positive_ids=[],
            benign_tp_ids=[],
            expected_techniques={
                alert_id: ["T1486", "T1490", "T1489", "T1071.001", "T1083"]
            },
            expected_response_actions={
                alert_id: [
                    ResponseActionType.ISOLATE_ENDPOINT,
                    ResponseActionType.QUARANTINE_FILE,
                    ResponseActionType.BLOCK_IP,
                    ResponseActionType.BLOCK_DOMAIN,
                ]
            },
            relevant_log_sources={
                alert_id: [LogSource.ENDPOINT, LogSource.DNS, LogSource.FIREWALL, LogSource.IDS]
            },
            relevant_indicators={alert_id: [payload_hash, c2_domain, c2_ip]},
        )

        return ScenarioConfig(
            scenario_id=f"ransomware-tp-{self.seed}",
            task_id="ransomware",
            seed=self.seed,
            description=f"Single ransomware alert — TRUE POSITIVE. {family} encryption with shadow-copy deletion.",
            max_steps=self.MAX_STEPS,
            alerts=[alert],
            enrichment_db=enrichment_db,
            log_db=log_db,
            asset_db=asset_db,
            user_db=user_db,
            ground_truth=ground_truth,
        )

    # ------------------------------------------------------------------
    # False Positive Variant — legitimate backup/encryption agent
    # ------------------------------------------------------------------

    def _generate_false_positive(self) -> ScenarioConfig:
        alert_id = self._alert_id("RANSOM")

        agent_hash = self._sha256()
        agent = self.rng.choice(["Veeam Agent", "Acronis TrueImage", "CrashPlan", "Duplicati"])
        vendor_ip = self._public_ip()
        vendor_domain = self._legit_domain()
        username = self._username()
        host = self._hostname("WORKSTATION")
        host_ip = self._private_ip()

        alert = AlertMeta(
            alert_id=alert_id,
            title="Possible Ransomware — High-Volume File Access by Backup Process",
            description=(
                f"EDR flagged a high-rate file-access burst on {host} by '{agent}'. "
                f"Behavioural rule fired on volume; process is code-signed and no shadow "
                f"copies were deleted, no ransom note observed."
            ),
            severity=AlertSeverity.MEDIUM,
            source_system="EDR",
            timestamp=self._timestamp(hours_ago=0.4),
            rule_triggered="RANSOM_HIGH_VOLUME_FILE_ACCESS_002",
            indicators={
                "ip": [vendor_ip],
                "domain": [vendor_domain],
                "file_hash": [agent_hash],
            },
            raw_log_snippet=(
                f"HOST={host} USER={username} PROC={agent.replace(' ', '')}.exe "
                f"HASH={agent_hash[:16]}... FILES_READ=51230 SIGNED=true VSSADMIN=none NOTE=none"
            ),
        )

        enrichment_db = {
            agent_hash: self._make_enrichment_result(
                agent_hash, IndicatorType.FILE_HASH, malicious=False,
                confidence=0.97, threat_score=0,
                tags=["backup-software", "code-signed", "clean"],
            ),
            vendor_domain: self._make_enrichment_result(
                vendor_domain, IndicatorType.DOMAIN, malicious=False,
                confidence=0.99, threat_score=0,
                geo="United States",
                tags=["legitimate", "trusted", "software-vendor"],
                whois="Registered 10+ years ago. WHOIS matches vendor registration.",
            ),
            vendor_ip: self._make_enrichment_result(
                vendor_ip, IndicatorType.IP, malicious=False,
                confidence=0.95, threat_score=1,
                geo="United States",
                tags=["legitimate", "cloud-backup-endpoint"],
            ),
        }

        log_db = self._empty_log_db([alert_id])
        log_db[LogSource.ENDPOINT.value][alert_id] = [
            self._make_log_entry(
                LogSource.ENDPOINT, "process_created", hours_ago=0.5,
                hostname=host, src_ip=host_ip, user=username, action="execute",
                details={
                    "process_name": f"{agent.replace(' ', '')}.exe",
                    "parent_process": "services.exe",
                    "hash_sha256": agent_hash,
                    "signed": True,
                    "signer": f"{agent.split()[0]} Software Inc.",
                    "hostname": host,
                },
            ),
            self._make_log_entry(
                LogSource.ENDPOINT, "file_access_burst", hours_ago=0.45,
                hostname=host, user=username, action="read",
                details={
                    "files_read": 51230,
                    "files_modified": 0,
                    "new_extension": None,
                    "ransom_note": None,
                    "scheduled_task": "Nightly Backup",
                    "hostname": host,
                },
            ),
        ]

        asset_db = {host: self._make_asset(
            host, "workstation", username, self._department(), host_ip,
        )}
        user_db = {username: self._make_user(
            username, "Operations Specialist", "Operations", risk_score=0.05,
        )}

        ground_truth = GroundTruth(
            alert_classifications={alert_id: AlertClassification.FALSE_POSITIVE},
            true_positive_ids=[],
            false_positive_ids=[alert_id],
            benign_tp_ids=[],
            expected_techniques={},
            expected_response_actions={alert_id: [ResponseActionType.NO_ACTION]},
            relevant_log_sources={alert_id: [LogSource.ENDPOINT]},
            relevant_indicators={alert_id: [vendor_ip, vendor_domain, agent_hash]},
        )

        return ScenarioConfig(
            scenario_id=f"ransomware-fp-{self.seed}",
            task_id="ransomware",
            seed=self.seed,
            description=f"Single ransomware alert — FALSE POSITIVE. Legitimate {agent} backup burst.",
            max_steps=self.MAX_STEPS,
            alerts=[alert],
            enrichment_db=enrichment_db,
            log_db=log_db,
            asset_db=asset_db,
            user_db=user_db,
            ground_truth=ground_truth,
        )
