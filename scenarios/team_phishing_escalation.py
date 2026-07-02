"""
Team Phishing Escalation Scenario Generator
=============================================
Team-mode phishing scenario designed for Tier-1 + Tier-2 + Manager coordination.

- 1 alert (phishing TP with malicious file hash + malicious sender IP)
- Tier-1 must classify and escalate ALT-TEAM-001
- Tier-2 must quarantine file, block IP, isolate host, block IOC, sandbox detonate
- Manager reviews; no inconsistencies expected in this easy scenario
"""


from models import (
    AlertMeta, AlertSeverity, AlertClassification, ResponseActionType,
    LogSource, IndicatorType,
    AssetInfo, GroundTruth,
    ScenarioConfig,
)
from scenarios.base import BaseScenario


class TeamPhishingEscalationScenario(BaseScenario):
    """
    Team-mode phishing scenario: Tier-1 triages and escalates, Tier-2 contains,
    Manager reviews. Single TRUE_POSITIVE alert with clear malicious indicators.
    """

    ALERT_ID = "ALT-TEAM-001"
    MAX_STEPS = 68

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)

    def generate(self) -> ScenarioConfig:
        rng = self.rng  # seeded random.Random(seed) from BaseScenario

        alert_id = self.ALERT_ID

        # --- IOCs ---
        sender_ip = self._public_ip()           # malicious sender IP (Eastern Europe)
        malicious_domain = self._malicious_domain()
        file_hash = self._sha256()              # malicious file hash
        sender_email = f"invoice@{malicious_domain}"
        recipient_user = self._username()
        victim_host = "WORKSTATION-ALPHA"
        victim_ip = self._private_ip()
        c2_domain = self._malicious_domain()
        c2_ip = self._public_ip()

        # --- Alert ---
        alert = AlertMeta(
            alert_id=alert_id,
            title="Phishing Email with Malicious Executable — Team Escalation Required",
            description=(
                f"Email security gateway flagged an inbound message from {sender_email} "
                f"containing a malicious executable attachment 'InvoiceQ4.exe'. "
                f"Sender IP is known-malicious (Eastern Europe). SPF, DKIM, and DMARC all failed. "
                f"File hash matches known malware in threat intel database."
            ),
            severity=AlertSeverity.HIGH,
            source_system="Email Security Gateway",
            timestamp=self._timestamp(hours_ago=1.5),
            rule_triggered="TEAM_PHISH_EXEC_ESCALATION_001",
            indicators={
                "ip": [sender_ip],
                "domain": [malicious_domain],
                "file_hash": [file_hash],
                "email": [sender_email],
            },
            raw_log_snippet=(
                f"FROM={sender_email} TO={recipient_user}@acmecorp.com "
                f"SUBJECT='Invoice Q4 - Urgent Action Required' ATTACHMENT=InvoiceQ4.exe "
                f"HASH={file_hash[:16]}... SPF=fail DKIM=fail DMARC=fail SRC_IP={sender_ip}"
            ),
        )

        # --- Enrichment DB ---
        enrichment_db = {
            sender_ip: self._make_enrichment_result(
                sender_ip, IndicatorType.IP, malicious=True,
                confidence=0.95, threat_score=88,
                threat_type="phishing",
                geo="Eastern Europe",
                tags=["phishing-actor", "bulletproof-hosting", "eastern-europe"],
                malware=["AgentTesla"],
                whois="Registered 7 days ago. AS: BULLETPROOF-EE-002. Geolocation: Romania.",
            ),
            malicious_domain: self._make_enrichment_result(
                malicious_domain, IndicatorType.DOMAIN, malicious=True,
                confidence=0.90, threat_score=82,
                threat_type="phishing",
                geo="Eastern Europe",
                tags=["newly-registered", "phishing-kit", "lookalike"],
                whois="Registered 4 days ago via anonymous registrar. No abuse contact.",
            ),
            file_hash: self._make_enrichment_result(
                file_hash, IndicatorType.FILE_HASH, malicious=True,
                confidence=0.99, threat_score=85,
                threat_type="malware",
                tags=["trojan", "dropper", "agent-tesla"],
                malware=["AgentTesla"],
            ),
            sender_email: self._make_enrichment_result(
                sender_email, IndicatorType.EMAIL, malicious=True,
                confidence=0.87, threat_score=78,
                threat_type="phishing",
                tags=["phishing-sender", "spoofed-invoice"],
            ),
            c2_domain: self._make_enrichment_result(
                c2_domain, IndicatorType.DOMAIN, malicious=True,
                confidence=0.93, threat_score=90,
                threat_type="command-and-control",
                geo="Eastern Europe",
                tags=["c2", "agent-tesla-c2"],
                malware=["AgentTesla"],
            ),
            c2_ip: self._make_enrichment_result(
                c2_ip, IndicatorType.IP, malicious=True,
                confidence=0.92, threat_score=88,
                threat_type="command-and-control",
                geo="Eastern Europe",
                tags=["c2", "agent-tesla-c2"],
                malware=["AgentTesla"],
            ),
        }

        # --- Log DB ---
        log_db = self._empty_log_db([alert_id])

        # Email Gateway — phishing email delivered
        log_db[LogSource.EMAIL_GATEWAY.value][alert_id] = [
            self._make_log_entry(
                LogSource.EMAIL_GATEWAY, "email_received",
                hours_ago=1.5, src_ip=sender_ip, user=recipient_user,
                action="delivered",
                details={
                    "from": sender_email,
                    "to": f"{recipient_user}@acmecorp.com",
                    "subject": "Invoice Q4 - Urgent Action Required",
                    "attachment": "InvoiceQ4.exe",
                    "attachment_hash": file_hash,
                    "size_bytes": 763904,
                    "spf": "fail",
                    "dkim": "fail",
                    "dmarc": "fail",
                    "spam_score": 9.1,
                    "sender_ip": sender_ip,
                    "sender_domain": malicious_domain,
                }
            ),
        ]

        # Endpoint — suspicious file execution
        log_db[LogSource.ENDPOINT.value][alert_id] = [
            self._make_log_entry(
                LogSource.ENDPOINT, "process_created",
                hours_ago=1.2, src_ip=victim_ip, hostname=victim_host,
                user=recipient_user, action="execute",
                severity="high",
                details={
                    "process_name": "InvoiceQ4.exe",
                    "process_path": f"C:\\Users\\{recipient_user}\\Downloads\\InvoiceQ4.exe",
                    "parent_process": "OUTLOOK.EXE",
                    "hash_sha256": file_hash,
                    "command_line": "InvoiceQ4.exe /silent /install",
                    "hostname": victim_host,
                }
            ),
            self._make_log_entry(
                LogSource.ENDPOINT, "process_created",
                hours_ago=1.1, hostname=victim_host, user=recipient_user,
                action="execute", severity="critical",
                details={
                    "process_name": "powershell.exe",
                    "parent_process": "InvoiceQ4.exe",
                    "command_line": "powershell -enc SGVsbG8gV29ybGQ=",
                    "encoded_payload": True,
                    "hostname": victim_host,
                }
            ),
            self._make_log_entry(
                LogSource.ENDPOINT, "file_created",
                hours_ago=1.05, hostname=victim_host, user=recipient_user,
                action="write", severity="medium",
                details={
                    "file_path": "C:\\ProgramData\\svchost32.exe",
                    "hash_sha256": file_hash,
                    "hostname": victim_host,
                    "hidden": True,
                    "persistence": "startup_folder",
                }
            ),
        ]

        # DNS — C2 beacon
        log_db[LogSource.DNS.value][alert_id] = [
            self._make_log_entry(
                LogSource.DNS, "dns_query",
                hours_ago=1.0, hostname=victim_host, src_ip=victim_ip,
                details={
                    "query": c2_domain,
                    "response": c2_ip,
                    "record_type": "A",
                    "hostname": victim_host,
                }
            ),
        ]

        # Firewall — outbound C2 connection
        log_db[LogSource.FIREWALL.value][alert_id] = [
            self._make_log_entry(
                LogSource.FIREWALL, "outbound_connection",
                hours_ago=0.9, src_ip=victim_ip, dst_ip=c2_ip,
                action="allowed", severity="high",
                details={
                    "dst_port": 587,
                    "protocol": "TCP",
                    "bytes_sent": 8192,
                    "bytes_recv": 16384,
                    "duration_seconds": 1800,
                    "direction": "outbound",
                }
            ),
        ]

        # IDS — malware signature match
        log_db[LogSource.IDS.value][alert_id] = [
            self._make_log_entry(
                LogSource.IDS, "signature_match",
                hours_ago=1.1, src_ip=victim_ip, dst_ip=c2_ip,
                severity="critical",
                details={
                    "signature": "MALWARE-CNC AgentTesla checkin",
                    "category": "Malware-CnC",
                    "priority": 1,
                    "hostname": victim_host,
                }
            ),
        ]

        # --- Asset DB ---
        asset_db = {
            victim_host: AssetInfo(
                asset_id=f"AST-{victim_host}",
                hostname=victim_host,
                asset_type="workstation",
                criticality="medium",
                owner=recipient_user,
                department="Finance",
                ip_address=victim_ip,
                os="Windows 10",
                patch_status="current",
                last_scan=self._timestamp(hours_ago=48),
                open_vulnerabilities=2,
                recent_activity_summary="Standard Finance workstation. Normal activity prior to incident.",
            )
        }

        # --- User DB ---
        user_db = {
            recipient_user: self._make_user(
                recipient_user, "Financial Analyst", "Finance",
                risk_score=0.2,
            )
        }

        # --- Ground Truth ---
        ground_truth = GroundTruth(
            alert_classifications={alert_id: AlertClassification.TRUE_POSITIVE},
            true_positive_ids=[alert_id],
            false_positive_ids=[],
            benign_tp_ids=[],
            expected_techniques={
                alert_id: ["T1566.001", "T1204.002"]
            },
            expected_response_actions={
                alert_id: [
                    ResponseActionType.QUARANTINE_FILE,
                    ResponseActionType.BLOCK_IP,
                ]
            },
            relevant_log_sources={
                alert_id: [
                    LogSource.EMAIL_GATEWAY,
                    LogSource.ENDPOINT,
                    LogSource.DNS,
                    LogSource.FIREWALL,
                    LogSource.IDS,
                ]
            },
            relevant_indicators={
                alert_id: [sender_ip, malicious_domain, file_hash, sender_email]
            },
            # Team-mode escalation ground truth
            required_escalations=[alert_id],
            required_containments={
                alert_id: ["isolate_host", "block_ioc", "sandbox_detonate"]
            },
            expected_manager_flags=[],  # Easy scenario — no inconsistencies to flag
        )

        return ScenarioConfig(
            scenario_id=f"team-phishing-escalation-{self.seed}",
            task_id="team_phishing_escalation",
            seed=self.seed,
            description="Team-mode phishing: Tier-1 triages and escalates, Tier-2 contains, Manager reviews.",
            max_steps=self.MAX_STEPS,
            alerts=[alert],
            enrichment_db=enrichment_db,
            log_db=log_db,
            asset_db=asset_db,
            user_db=user_db,
            ground_truth=ground_truth,
        )
