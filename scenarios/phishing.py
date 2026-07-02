"""
Phishing Scenario Generator — Task 1 (Easy)
============================================
Generates a single phishing alert with full supporting evidence.

Two variants depending on seed parity:
  - TRUE_POSITIVE: malicious sender, newly-registered domain, SPF/DKIM fail,
                   user executed attachment, DNS callback, outbound C2 connection
  - FALSE_POSITIVE: legitimate newsletter, clean domain, SPF/DKIM/DMARC pass,
                    no endpoint execution, no suspicious indicators
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


class PhishingScenario(BaseScenario):
    """Easy task: single phishing alert triage."""

    MAX_STEPS = 15

    def generate(self) -> ScenarioConfig:
        # Determine variant based on seed
        is_tp = (self.seed % 2 == 0) or (self.seed % 7 < 4)

        if is_tp:
            return self._generate_true_positive()
        else:
            return self._generate_false_positive()

    # ------------------------------------------------------------------
    # True Positive Variant
    # ------------------------------------------------------------------

    def _generate_true_positive(self) -> ScenarioConfig:
        alert_id = self._alert_id("PHI")

        # Generate IOCs
        attacker_ip = self._public_ip()
        malicious_domain = self._malicious_domain()
        attachment_hash = self._sha256()
        sender_email = f"billing@{self._malicious_domain()}"
        victim_username = self._username()
        victim_host = self._hostname("WORKSTATION")
        victim_ip = self._private_ip()
        c2_domain = self._malicious_domain()
        c2_ip = self._public_ip()

        # Alert
        alert = AlertMeta(
            alert_id=alert_id,
            title="Suspicious Phishing Email with Executable Attachment",
            description=(
                f"Email security gateway flagged an inbound email from {sender_email} "
                f"containing an executable attachment 'Invoice_Q4.exe'. "
                f"Sender domain has low reputation. SPF and DKIM checks failed."
            ),
            severity=AlertSeverity.HIGH,
            source_system="Email Security Gateway",
            timestamp=self._timestamp(hours_ago=1.5),
            rule_triggered="PHISH_EXEC_ATTACHMENT_001",
            indicators={
                "ip": [attacker_ip, c2_ip],
                "domain": [malicious_domain, c2_domain],
                "file_hash": [attachment_hash],
                "email": [sender_email],
            },
            raw_log_snippet=(
                f"FROM={sender_email} TO={victim_username}@acmecorp.com "
                f"SUBJECT='Invoice Q4 - Action Required' ATTACHMENT=Invoice_Q4.exe "
                f"HASH={attachment_hash[:16]}... SPF=fail DKIM=fail DMARC=fail"
            ),
        )

        # Enrichment DB
        enrichment_db = {
            attacker_ip: self._make_enrichment_result(
                attacker_ip, IndicatorType.IP, malicious=True,
                confidence=0.93, threat_score=88,
                threat_type="phishing",
                geo="Russia",
                tags=["phishing-actor", "bulletproof-hosting"],
                malware=["Emotet"],
                whois="Registered 3 days ago. AS: BULLETPROOF-HOSTING-001",
            ),
            malicious_domain: self._make_enrichment_result(
                malicious_domain, IndicatorType.DOMAIN, malicious=True,
                confidence=0.91, threat_score=85,
                threat_type="phishing",
                geo="Russia",
                tags=["newly-registered", "phishing-kit", "lookalike"],
                whois="Registered 5 days ago via anonymous registrar.",
            ),
            attachment_hash: self._make_enrichment_result(
                attachment_hash, IndicatorType.FILE_HASH, malicious=True,
                confidence=0.98, threat_score=97,
                threat_type="malware",
                tags=["trojan", "dropper", "emotet"],
                malware=["Emotet", "TrickBot"],
            ),
            sender_email: self._make_enrichment_result(
                sender_email, IndicatorType.EMAIL, malicious=True,
                confidence=0.85, threat_score=80,
                threat_type="phishing",
                tags=["phishing-sender", "spoofed-billing"],
            ),
            c2_domain: self._make_enrichment_result(
                c2_domain, IndicatorType.DOMAIN, malicious=True,
                confidence=0.95, threat_score=92,
                threat_type="command-and-control",
                geo="Ukraine",
                tags=["c2", "malware-c2", "emotet-c2"],
                malware=["Emotet"],
            ),
            c2_ip: self._make_enrichment_result(
                c2_ip, IndicatorType.IP, malicious=True,
                confidence=0.94, threat_score=90,
                threat_type="command-and-control",
                geo="Ukraine",
                tags=["c2", "emotet-c2"],
                malware=["Emotet"],
            ),
        }

        # Log DB
        log_db = self._empty_log_db([alert_id])

        # Email Gateway — email received
        log_db[LogSource.EMAIL_GATEWAY.value][alert_id] = [
            self._make_log_entry(
                LogSource.EMAIL_GATEWAY, "email_received",
                hours_ago=1.5, src_ip=attacker_ip, user=victim_username,
                action="delivered",
                details={
                    "from": sender_email,
                    "to": f"{victim_username}@acmecorp.com",
                    "subject": "Invoice Q4 - Action Required",
                    "attachment": "Invoice_Q4.exe",
                    "attachment_hash": attachment_hash,
                    "size_bytes": 847392,
                    "spf": "fail",
                    "dkim": "fail",
                    "dmarc": "fail",
                    "spam_score": 8.7,
                }
            ),
        ]

        # Endpoint — user executed attachment
        log_db[LogSource.ENDPOINT.value][alert_id] = [
            self._make_log_entry(
                LogSource.ENDPOINT, "process_created",
                hours_ago=1.2, src_ip=victim_ip, hostname=victim_host,
                user=victim_username, action="execute",
                severity="high",
                details={
                    "process_name": "Invoice_Q4.exe",
                    "process_path": "C:\\Users\\{victim_username}\\Downloads\\Invoice_Q4.exe",
                    "parent_process": "OUTLOOK.EXE",
                    "hash_sha256": attachment_hash,
                    "command_line": "Invoice_Q4.exe /silent",
                    "hostname": victim_host,
                }
            ),
            self._make_log_entry(
                LogSource.ENDPOINT, "process_created",
                hours_ago=1.1, hostname=victim_host, user=victim_username,
                action="execute", severity="critical",
                details={
                    "process_name": "powershell.exe",
                    "parent_process": "Invoice_Q4.exe",
                    "command_line": "powershell -enc JABzAD0ATgBlAHcALQBPAGIAagBlAGMAdA==",
                    "encoded_payload": True,
                    "hostname": victim_host,
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
                    "dst_port": 443,
                    "protocol": "TCP",
                    "bytes_sent": 4096,
                    "bytes_recv": 28672,
                    "duration_seconds": 3601,
                    "direction": "outbound",
                }
            ),
        ]

        # IDS — exploit detection
        log_db[LogSource.IDS.value][alert_id] = [
            self._make_log_entry(
                LogSource.IDS, "signature_match",
                hours_ago=1.1, src_ip=victim_ip, dst_ip=c2_ip,
                severity="critical",
                details={
                    "signature": "MALWARE-CNC Emotet checkin POST",
                    "category": "Malware-CnC",
                    "priority": 1,
                }
            ),
        ]

        # Asset + User DB
        asset_db = {victim_host: self._make_asset(
            victim_host, "workstation", victim_username,
            self._department(), victim_ip, criticality="medium"
        )}
        user_db = {victim_username: self._make_user(
            victim_username, "Financial Analyst", "Finance",
            risk_score=0.25
        )}

        # Ground Truth
        ground_truth = GroundTruth(
            alert_classifications={alert_id: AlertClassification.TRUE_POSITIVE},
            true_positive_ids=[alert_id],
            false_positive_ids=[],
            benign_tp_ids=[],
            expected_techniques={
                alert_id: ["T1566.001", "T1204.002", "T1059.001", "T1071.001", "T1041"]
            },
            expected_response_actions={
                alert_id: [
                    ResponseActionType.ISOLATE_ENDPOINT,
                    ResponseActionType.BLOCK_IP,
                    ResponseActionType.BLOCK_DOMAIN,
                    ResponseActionType.QUARANTINE_FILE,
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
                alert_id: [attacker_ip, malicious_domain, attachment_hash, sender_email, c2_domain, c2_ip]
            },
        )

        return ScenarioConfig(
            scenario_id=f"phishing-tp-{self.seed}",
            task_id="phishing",
            seed=self.seed,
            description="Single phishing alert — TRUE POSITIVE. Emotet dropper via malicious invoice attachment.",
            max_steps=self.MAX_STEPS,
            alerts=[alert],
            enrichment_db=enrichment_db,
            log_db=log_db,
            asset_db=asset_db,
            user_db=user_db,
            ground_truth=ground_truth,
        )

    # ------------------------------------------------------------------
    # False Positive Variant
    # ------------------------------------------------------------------

    def _generate_false_positive(self) -> ScenarioConfig:
        alert_id = self._alert_id("PHI")

        # Generate benign IOCs
        sender_domain = self._legit_domain()
        sender_ip = self._public_ip()
        pdf_hash = self._sha256()
        sender_email = f"newsletter@{sender_domain}"
        victim_username = self._username()

        alert = AlertMeta(
            alert_id=alert_id,
            title="Potential Phishing Email — Newsletter with Attachment",
            description=(
                f"Email security gateway flagged a newsletter from {sender_email} "
                f"containing a PDF attachment. Rule triggered due to attachment size. "
                f"SPF and DKIM passed."
            ),
            severity=AlertSeverity.MEDIUM,
            source_system="Email Security Gateway",
            timestamp=self._timestamp(hours_ago=0.5),
            rule_triggered="PHISH_ATTACHMENT_SIZE_002",
            indicators={
                "ip": [sender_ip],
                "domain": [sender_domain],
                "file_hash": [pdf_hash],
                "email": [sender_email],
            },
            raw_log_snippet=(
                f"FROM={sender_email} TO={victim_username}@acmecorp.com "
                f"SUBJECT='Monthly Security Digest' ATTACHMENT=SecDigest_Nov.pdf "
                f"HASH={pdf_hash[:16]}... SPF=pass DKIM=pass DMARC=pass"
            ),
        )

        # Enrichment DB — all clean
        enrichment_db = {
            sender_ip: self._make_enrichment_result(
                sender_ip, IndicatorType.IP, malicious=False,
                confidence=0.95, threat_score=2,
                geo="United States",
                tags=["legitimate-mailer", "mailchimp-sendgrid"],
                whois="Registered 5+ years ago. Large legitimate email provider.",
            ),
            sender_domain: self._make_enrichment_result(
                sender_domain, IndicatorType.DOMAIN, malicious=False,
                confidence=0.99, threat_score=0,
                geo="United States",
                tags=["legitimate", "trusted", "high-volume-sender"],
                whois="Registered 8+ years ago. WHOIS matches company registration.",
            ),
            pdf_hash: self._make_enrichment_result(
                pdf_hash, IndicatorType.FILE_HASH, malicious=False,
                confidence=0.90, threat_score=0,
                tags=["pdf", "document", "clean"],
            ),
            sender_email: self._make_enrichment_result(
                sender_email, IndicatorType.EMAIL, malicious=False,
                confidence=0.92, threat_score=1,
                tags=["newsletter", "opt-in", "legitimate-sender"],
            ),
        }

        # Log DB — no endpoint execution, no DNS anomaly
        log_db = self._empty_log_db([alert_id])

        log_db[LogSource.EMAIL_GATEWAY.value][alert_id] = [
            self._make_log_entry(
                LogSource.EMAIL_GATEWAY, "email_received",
                hours_ago=0.5, src_ip=sender_ip, user=victim_username,
                action="delivered",
                details={
                    "from": sender_email,
                    "to": f"{victim_username}@acmecorp.com",
                    "subject": "Monthly Security Digest",
                    "attachment": "SecDigest_Nov.pdf",
                    "attachment_hash": pdf_hash,
                    "size_bytes": 1523000,
                    "spf": "pass",
                    "dkim": "pass",
                    "dmarc": "pass",
                    "spam_score": 0.3,
                }
            ),
        ]

        # Asset + User DB
        victim_host = self._hostname("WORKSTATION")
        victim_ip = self._private_ip()
        asset_db = {victim_host: self._make_asset(
            victim_host, "workstation", victim_username,
            self._department(), victim_ip
        )}
        user_db = {victim_username: self._make_user(
            victim_username, "IT Security Analyst", "IT",
            risk_score=0.05
        )}

        # Ground Truth
        ground_truth = GroundTruth(
            alert_classifications={alert_id: AlertClassification.FALSE_POSITIVE},
            true_positive_ids=[],
            false_positive_ids=[alert_id],
            benign_tp_ids=[],
            expected_techniques={},
            expected_response_actions={alert_id: [ResponseActionType.NO_ACTION]},
            relevant_log_sources={
                alert_id: [LogSource.EMAIL_GATEWAY, LogSource.ENDPOINT]
            },
            relevant_indicators={
                alert_id: [sender_ip, sender_domain, pdf_hash]
            },
        )

        return ScenarioConfig(
            scenario_id=f"phishing-fp-{self.seed}",
            task_id="phishing",
            seed=self.seed,
            description="Single phishing alert — FALSE POSITIVE. Legitimate newsletter triggering attachment size rule.",
            max_steps=self.MAX_STEPS,
            alerts=[alert],
            enrichment_db=enrichment_db,
            log_db=log_db,
            asset_db=asset_db,
            user_db=user_db,
            ground_truth=ground_truth,
        )
