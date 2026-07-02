"""
Team Lateral Movement Scenario Generator (Team Mode)
======================================================
Team-mode lateral movement scenario with 5 kill-chain TPs + 3 FP noise alerts = 8 alerts total.

Kill chain (all TRUE_POSITIVE):
  ALT-TLT-001: Phishing email — initial access
  ALT-TLT-002: Credential dump on workstation — escalate required
  ALT-TLT-003: Lateral movement via SMB — escalate required
  ALT-TLT-004: Data staging to temp folder — Manager should flag if missed
  ALT-TLT-005: Outbound data exfiltration — escalate required

Noise (all FALSE_POSITIVE):
  ALT-TLT-006: DNS query noise
  ALT-TLT-007: Login from new location (benign employee travel)
  ALT-TLT-008: Port scan from internal monitoring tool

Tier-1 triages all 8, escalates 3 (002, 003, 005).
Tier-2 contains: disable_user (002), isolate_host (003), block_ioc (005).
Manager should flag ALT-TLT-004 (data staging often missed).
"""


from models import (
    AlertClassification,
    AlertMeta,
    AlertSeverity,
    AssetInfo,
    GroundTruth,
    IndicatorType,
    LogSource,
    ResponseActionType,
    ScenarioConfig,
)
from scenarios.base import BaseScenario


class TeamLateralTeamScenario(BaseScenario):
    """
    Team-mode lateral movement: Tier-1 triages 8 alerts, escalates 3,
    Tier-2 contains, Manager flags missed staging.
    """

    MAX_STEPS = 68

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)

    def generate(self) -> ScenarioConfig:
        rng = self.rng  # seeded random.Random(seed) from BaseScenario

        # --- Shared IOCs tying the kill chain together ---
        compromised_user = self._username()
        attacker_ip = self._public_ip()          # external phishing source IP
        workstation_ip = self._private_ip()       # compromised workstation IP
        workstation_host = self._hostname("WORKSTATION")
        dc_ip = self._private_ip()               # domain controller IP
        dc_host = "DC-01"
        staging_server_ip = self._private_ip()   # internal staging server
        staging_server_host = self._hostname("FILESERVER")
        exfil_ip = self._public_ip()             # external exfil destination IP
        exfil_domain = self._malicious_domain()

        phishing_domain = self._malicious_domain()
        phishing_hash = self._sha256()           # phishing dropper hash
        cred_dump_hash = self._sha256()          # credential dump tool hash (known bad)
        malicious_domain = self._malicious_domain()

        # Noise alert IOCs (benign / FP)
        noise_ip_1 = self._private_ip()          # internal DNS noise
        noise_ip_2 = self._public_ip()           # login from new location
        monitoring_ip = self._private_ip()       # internal monitoring tool

        # --- ALERT 001: Phishing email (initial access) ---
        alert_001 = AlertMeta(
            alert_id="ALT-TLT-001",
            title="Phishing Email with Malicious Attachment Delivered",
            description=(
                f"Email security gateway flagged a phishing email from billing@{phishing_domain} "
                f"delivered to {compromised_user}. Attachment 'Q3Invoice.exe' matched known malware hash. "
                f"SPF and DKIM failed."
            ),
            severity=AlertSeverity.HIGH,
            source_system="Email Security Gateway",
            timestamp=self._timestamp(hours_ago=6.0),
            rule_triggered="PHISH_EXEC_ATTACHMENT_001",
            indicators={
                "ip": [attacker_ip],
                "domain": [phishing_domain],
                "file_hash": [phishing_hash],
                "email": [f"billing@{phishing_domain}"],
            },
            raw_log_snippet=(
                f"FROM=billing@{phishing_domain} TO={compromised_user}@acmecorp.com "
                f"SUBJECT='Q3 Invoice - Urgent' ATTACHMENT=Q3Invoice.exe "
                f"HASH={phishing_hash[:16]}... SPF=fail DKIM=fail DMARC=fail"
            ),
        )

        # --- ALERT 002: Credential dump (post-compromise) ---
        alert_002 = AlertMeta(
            alert_id="ALT-TLT-002",
            title="Credential Dumping Tool Executed on Workstation",
            description=(
                f"EDR detected execution of a known credential dumping tool on {workstation_host}. "
                f"Process mimikatz-variant executed under user context '{compromised_user}'. "
                f"File hash matches known credential harvesting malware."
            ),
            severity=AlertSeverity.CRITICAL,
            source_system="EDR",
            timestamp=self._timestamp(hours_ago=5.5),
            rule_triggered="CRED_DUMP_TOOL_EXEC_001",
            indicators={
                "ip": [workstation_ip],
                "file_hash": [cred_dump_hash],
            },
            raw_log_snippet=(
                f"HOST={workstation_host} USER={compromised_user} "
                f"PROCESS=svchost_x64.exe HASH={cred_dump_hash[:16]}... "
                f"PARENT=powershell.exe LSASS_ACCESS=True"
            ),
            related_alert_ids=["ALT-TLT-001"],
        )

        # --- ALERT 003: Lateral movement via SMB ---
        alert_003 = AlertMeta(
            alert_id="ALT-TLT-003",
            title="Lateral Movement Detected — SMB Spread to Domain Controller",
            description=(
                f"IDS and firewall detected SMB lateral movement from {workstation_host} ({workstation_ip}) "
                f"to domain controller {dc_host} ({dc_ip}). "
                f"Suspicious admin share access using harvested credentials."
            ),
            severity=AlertSeverity.CRITICAL,
            source_system="IDS / Firewall",
            timestamp=self._timestamp(hours_ago=5.0),
            rule_triggered="LATERAL_MOVE_SMB_ADMIN_SHARE_001",
            indicators={
                "ip": [workstation_ip, dc_ip],
            },
            raw_log_snippet=(
                f"SRC={workstation_ip} DST={dc_ip} PORT=445 PROTOCOL=SMB "
                f"SHARE=ADMIN$ USER={compromised_user} ACTION=connect EVENT=admin_share_access"
            ),
            related_alert_ids=["ALT-TLT-001", "ALT-TLT-002"],
        )

        # --- ALERT 004: Data staging (often missed) ---
        alert_004 = AlertMeta(
            alert_id="ALT-TLT-004",
            title="Suspicious Data Staging to Temp Folder on File Server",
            description=(
                f"Endpoint telemetry shows bulk file copy from network shares to a temp folder "
                f"on {staging_server_host} ({staging_server_ip}). "
                f"Activity initiated under compromised user '{compromised_user}' context from {dc_host}. "
                f"Over 2 GB of documents copied in under 10 minutes."
            ),
            severity=AlertSeverity.MEDIUM,
            source_system="DLP / Endpoint",
            timestamp=self._timestamp(hours_ago=4.0),
            rule_triggered="DATA_STAGING_BULK_COPY_001",
            indicators={
                "ip": [staging_server_ip, dc_ip],
            },
            raw_log_snippet=(
                f"HOST={staging_server_host} USER={compromised_user} "
                f"ACTION=bulk_file_copy SRC_SHARE=\\\\{dc_host}\\finance "
                f"DST=C:\\Windows\\Temp\\arch\\ FILES_COPIED=3847 BYTES=2147483648"
            ),
            related_alert_ids=["ALT-TLT-002", "ALT-TLT-003"],
        )

        # --- ALERT 005: Data exfiltration ---
        alert_005 = AlertMeta(
            alert_id="ALT-TLT-005",
            title="Outbound Data Exfiltration to External IP",
            description=(
                f"Firewall and proxy detected large outbound data transfer from {staging_server_host} "
                f"to external IP {exfil_ip} ({exfil_domain}). "
                f"2.1 GB transferred over encrypted channel. Destination matches known exfil infrastructure."
            ),
            severity=AlertSeverity.CRITICAL,
            source_system="Firewall / Proxy",
            timestamp=self._timestamp(hours_ago=3.0),
            rule_triggered="EXFIL_LARGE_OUTBOUND_001",
            indicators={
                "ip": [staging_server_ip, exfil_ip],
                "domain": [exfil_domain],
            },
            raw_log_snippet=(
                f"SRC={staging_server_ip} DST={exfil_ip} DST_DOMAIN={exfil_domain} "
                f"PROTOCOL=HTTPS PORT=443 BYTES_OUT=2254857830 DURATION_S=3600 "
                f"ACTION=allowed"
            ),
            related_alert_ids=["ALT-TLT-003", "ALT-TLT-004"],
        )

        # --- ALERT 006: DNS query noise (FP) ---
        alert_006 = AlertMeta(
            alert_id="ALT-TLT-006",
            title="High-Volume DNS Queries — Possible DNS Tunneling",
            description=(
                "DNS monitoring flagged an internal host generating high-volume DNS queries. "
                "Investigation shows this is a known software update check behavior. "
                "No malicious domain resolution observed."
            ),
            severity=AlertSeverity.LOW,
            source_system="DNS Monitor",
            timestamp=self._timestamp(hours_ago=2.5),
            rule_triggered="DNS_HIGH_VOLUME_QUERIES_001",
            indicators={
                "ip": [noise_ip_1],
            },
            raw_log_snippet=(
                f"SRC={noise_ip_1} DNS_QUERIES_PER_MIN=180 "
                f"DOMAINS=windowsupdate.com,microsoft.com PATTERN=update_check"
            ),
        )

        # --- ALERT 007: Login from new location (benign employee travel, FP) ---
        benign_user = self._username()
        alert_007 = AlertMeta(
            alert_id="ALT-TLT-007",
            title="Login from Unusual Geographic Location",
            description=(
                f"Identity platform flagged {benign_user} logging in from an unusual location. "
                f"HR records confirm employee is on approved business travel. MFA passed."
            ),
            severity=AlertSeverity.MEDIUM,
            source_system="Identity Platform",
            timestamp=self._timestamp(hours_ago=2.0),
            rule_triggered="IDENTITY_UNUSUAL_LOCATION_001",
            indicators={
                "ip": [noise_ip_2],
            },
            raw_log_snippet=(
                f"USER={benign_user} SRC_IP={noise_ip_2} GEO=United Kingdom "
                f"MFA_STATUS=passed HR_TRAVEL=confirmed RISK_SCORE=0.15"
            ),
        )

        # --- ALERT 008: Port scan from monitoring tool (FP) ---
        alert_008 = AlertMeta(
            alert_id="ALT-TLT-008",
            title="Internal Port Scan Detected",
            description=(
                "IDS detected TCP port scanning activity from an internal IP. "
                "Asset inventory confirms this IP belongs to the Nessus vulnerability scanner. "
                "Scan matches scheduled weekly vulnerability assessment window."
            ),
            severity=AlertSeverity.LOW,
            source_system="IDS",
            timestamp=self._timestamp(hours_ago=1.0),
            rule_triggered="IDS_PORT_SCAN_INTERNAL_001",
            indicators={
                "ip": [monitoring_ip],
            },
            raw_log_snippet=(
                f"SRC={monitoring_ip} SCAN_TYPE=tcp_syn PORTS_SCANNED=65535 "
                f"SCHEDULE=weekly_vuln_scan ASSET=nessus-scanner-01"
            ),
        )

        # --- Enrichment DB ---
        enrichment_db = {
            # Kill chain malicious indicators
            attacker_ip: self._make_enrichment_result(
                attacker_ip, IndicatorType.IP, malicious=True,
                confidence=0.94, threat_score=87,
                threat_type="phishing",
                geo="Eastern Europe",
                tags=["phishing-actor", "bulletproof-hosting"],
                malware=["AgentTesla"],
                whois="Registered 10 days ago. Bulletproof hosting.",
            ),
            phishing_domain: self._make_enrichment_result(
                phishing_domain, IndicatorType.DOMAIN, malicious=True,
                confidence=0.91, threat_score=83,
                threat_type="phishing",
                geo="Eastern Europe",
                tags=["newly-registered", "phishing-kit"],
                whois="Registered 6 days ago. Anonymous registrar.",
            ),
            phishing_hash: self._make_enrichment_result(
                phishing_hash, IndicatorType.FILE_HASH, malicious=True,
                confidence=0.97, threat_score=90,
                threat_type="malware",
                tags=["trojan", "dropper"],
                malware=["AgentTesla"],
            ),
            cred_dump_hash: self._make_enrichment_result(
                cred_dump_hash, IndicatorType.FILE_HASH, malicious=True,
                confidence=0.99, threat_score=96,
                threat_type="credential-dumping",
                tags=["mimikatz", "credential-dumper", "lsass"],
                malware=["Mimikatz"],
            ),
            exfil_ip: self._make_enrichment_result(
                exfil_ip, IndicatorType.IP, malicious=True,
                confidence=0.96, threat_score=93,
                threat_type="exfiltration",
                geo="Russia",
                tags=["exfil-infra", "known-bad"],
                malware=[],
            ),
            exfil_domain: self._make_enrichment_result(
                exfil_domain, IndicatorType.DOMAIN, malicious=True,
                confidence=0.94, threat_score=91,
                threat_type="exfiltration",
                geo="Russia",
                tags=["exfil-domain", "known-bad"],
            ),
            # Noise / benign indicators
            noise_ip_1: self._make_enrichment_result(
                noise_ip_1, IndicatorType.IP, malicious=False,
                confidence=0.97, threat_score=0,
                tags=["internal", "software-update"],
            ),
            noise_ip_2: self._make_enrichment_result(
                noise_ip_2, IndicatorType.IP, malicious=False,
                confidence=0.92, threat_score=5,
                geo="United Kingdom",
                tags=["employee-travel", "legitimate"],
            ),
            monitoring_ip: self._make_enrichment_result(
                monitoring_ip, IndicatorType.IP, malicious=False,
                confidence=0.99, threat_score=0,
                tags=["internal", "vulnerability-scanner", "nessus"],
            ),
        }

        # --- Log DB (all 8 alerts) ---
        all_alert_ids = [
            "ALT-TLT-001", "ALT-TLT-002", "ALT-TLT-003",
            "ALT-TLT-004", "ALT-TLT-005", "ALT-TLT-006",
            "ALT-TLT-007", "ALT-TLT-008",
        ]
        log_db = self._empty_log_db(all_alert_ids)

        # ALT-TLT-001: Email gateway + endpoint (dropper execution)
        log_db[LogSource.EMAIL_GATEWAY.value]["ALT-TLT-001"] = [
            self._make_log_entry(
                LogSource.EMAIL_GATEWAY, "email_received",
                hours_ago=6.0, src_ip=attacker_ip, user=compromised_user,
                action="delivered",
                details={
                    "from": f"billing@{phishing_domain}",
                    "to": f"{compromised_user}@acmecorp.com",
                    "subject": "Q3 Invoice - Urgent",
                    "attachment": "Q3Invoice.exe",
                    "attachment_hash": phishing_hash,
                    "spf": "fail",
                    "dkim": "fail",
                    "dmarc": "fail",
                    "spam_score": 8.9,
                }
            ),
        ]
        log_db[LogSource.ENDPOINT.value]["ALT-TLT-001"] = [
            self._make_log_entry(
                LogSource.ENDPOINT, "process_created",
                hours_ago=5.8, src_ip=workstation_ip, hostname=workstation_host,
                user=compromised_user, action="execute", severity="high",
                details={
                    "process_name": "Q3Invoice.exe",
                    "parent_process": "OUTLOOK.EXE",
                    "hash_sha256": phishing_hash,
                    "hostname": workstation_host,
                }
            ),
        ]

        # ALT-TLT-002: Endpoint (cred dump tool) + Auth (lsass access)
        log_db[LogSource.ENDPOINT.value]["ALT-TLT-002"] = [
            self._make_log_entry(
                LogSource.ENDPOINT, "process_created",
                hours_ago=5.5, src_ip=workstation_ip, hostname=workstation_host,
                user=compromised_user, action="execute", severity="critical",
                details={
                    "process_name": "svchost_x64.exe",
                    "parent_process": "powershell.exe",
                    "hash_sha256": cred_dump_hash,
                    "lsass_access": True,
                    "hostname": workstation_host,
                }
            ),
        ]
        log_db[LogSource.AUTH.value]["ALT-TLT-002"] = [
            self._make_log_entry(
                LogSource.AUTH, "credential_access",
                hours_ago=5.4, src_ip=workstation_ip, hostname=workstation_host,
                user=compromised_user, action="lsass_dump", severity="critical",
                details={
                    "target_process": "lsass.exe",
                    "technique": "T1003.001",
                    "hostname": workstation_host,
                }
            ),
        ]

        # ALT-TLT-003: Firewall + Auth (SMB lateral movement)
        log_db[LogSource.FIREWALL.value]["ALT-TLT-003"] = [
            self._make_log_entry(
                LogSource.FIREWALL, "lateral_movement",
                hours_ago=5.0, src_ip=workstation_ip, dst_ip=dc_ip,
                action="allowed", severity="critical",
                details={
                    "dst_port": 445,
                    "protocol": "SMB",
                    "share": "ADMIN$",
                    "user": compromised_user,
                    "src_host": workstation_host,
                    "dst_host": dc_host,
                }
            ),
        ]
        log_db[LogSource.AUTH.value]["ALT-TLT-003"] = [
            self._make_log_entry(
                LogSource.AUTH, "admin_share_access",
                hours_ago=4.9, src_ip=workstation_ip, dst_ip=dc_ip,
                user=compromised_user, action="connect", severity="high",
                details={
                    "share": "ADMIN$",
                    "src_host": workstation_host,
                    "dst_host": dc_host,
                    "auth_type": "NTLM",
                    "pass_the_hash": True,
                }
            ),
        ]

        # ALT-TLT-004: Endpoint (data staging)
        log_db[LogSource.ENDPOINT.value]["ALT-TLT-004"] = [
            self._make_log_entry(
                LogSource.ENDPOINT, "bulk_file_copy",
                hours_ago=4.0, src_ip=dc_ip, hostname=dc_host,
                user=compromised_user, action="write", severity="medium",
                details={
                    "src_share": f"\\\\{dc_host}\\finance",
                    "dst_path": "C:\\Windows\\Temp\\arch\\",
                    "files_copied": 3847,
                    "bytes_total": 2147483648,
                    "duration_seconds": 587,
                    "hostname": staging_server_host,
                }
            ),
        ]

        # ALT-TLT-005: Firewall + Proxy (data exfiltration)
        log_db[LogSource.FIREWALL.value]["ALT-TLT-005"] = [
            self._make_log_entry(
                LogSource.FIREWALL, "outbound_large_transfer",
                hours_ago=3.0, src_ip=staging_server_ip, dst_ip=exfil_ip,
                action="allowed", severity="critical",
                details={
                    "dst_port": 443,
                    "protocol": "HTTPS",
                    "bytes_sent": 2254857830,
                    "duration_seconds": 3600,
                    "dst_domain": exfil_domain,
                }
            ),
        ]
        log_db[LogSource.PROXY.value]["ALT-TLT-005"] = [
            self._make_log_entry(
                LogSource.PROXY, "https_connect",
                hours_ago=2.9, src_ip=staging_server_ip, dst_ip=exfil_ip,
                action="allowed", severity="critical",
                details={
                    "url": f"https://{exfil_domain}/upload",
                    "method": "POST",
                    "bytes_sent": 2254857830,
                    "response_code": 200,
                }
            ),
        ]

        # ALT-TLT-006: DNS noise (FP — software update)
        log_db[LogSource.DNS.value]["ALT-TLT-006"] = [
            self._make_log_entry(
                LogSource.DNS, "high_volume_dns_queries",
                hours_ago=2.5, src_ip=noise_ip_1,
                action="allowed", severity="low",
                details={
                    "queries_per_minute": 180,
                    "top_domains": ["windowsupdate.com", "microsoft.com", "download.microsoft.com"],
                    "pattern": "software_update_check",
                    "anomaly": False,
                }
            ),
        ]

        # ALT-TLT-007: Auth noise (FP — employee travel)
        log_db[LogSource.AUTH.value]["ALT-TLT-007"] = [
            self._make_log_entry(
                LogSource.AUTH, "login_unusual_location",
                hours_ago=2.0, src_ip=noise_ip_2,
                user=benign_user, action="login_success", severity="medium",
                details={
                    "geo": "United Kingdom",
                    "mfa_passed": True,
                    "hr_travel_approved": True,
                    "risk_score": 0.15,
                    "user": benign_user,
                }
            ),
        ]

        # ALT-TLT-008: IDS noise (FP — monitoring tool scan)
        log_db[LogSource.IDS.value]["ALT-TLT-008"] = [
            self._make_log_entry(
                LogSource.IDS, "port_scan_detected",
                hours_ago=1.0, src_ip=monitoring_ip,
                action="allowed", severity="low",
                details={
                    "scan_type": "tcp_syn",
                    "ports_scanned": 65535,
                    "schedule": "weekly_vuln_scan",
                    "asset": "nessus-scanner-01",
                    "authorized": True,
                }
            ),
        ]

        # --- Asset DB ---
        asset_db = {
            workstation_host: AssetInfo(
                asset_id=f"AST-{workstation_host}",
                hostname=workstation_host,
                asset_type="workstation",
                criticality="medium",
                owner=compromised_user,
                department="Finance",
                ip_address=workstation_ip,
                os="Windows 10",
                patch_status="current",
                last_scan=self._timestamp(hours_ago=72),
                open_vulnerabilities=3,
                recent_activity_summary=(
                    "Standard Finance workstation. Anomalous process execution detected in last 6 hours."
                ),
            ),
            dc_host: AssetInfo(
                asset_id=f"AST-{dc_host}",
                hostname=dc_host,
                asset_type="domain_controller",
                criticality="critical",
                owner="svc.domain",
                department="IT",
                ip_address=dc_ip,
                os="Windows Server 2019",
                patch_status="current",
                last_scan=self._timestamp(hours_ago=24),
                open_vulnerabilities=1,
                recent_activity_summary=(
                    "Domain controller. Lateral movement from Finance workstation detected."
                ),
            ),
            staging_server_host: AssetInfo(
                asset_id=f"AST-{staging_server_host}",
                hostname=staging_server_host,
                asset_type="file_server",
                criticality="high",
                owner="svc.fileserver",
                department="IT",
                ip_address=staging_server_ip,
                os="Windows Server 2016",
                patch_status="behind",
                last_scan=self._timestamp(hours_ago=120),
                open_vulnerabilities=5,
                recent_activity_summary=(
                    "File server. Bulk data staging activity detected from temp folder."
                ),
            ),
        }

        # --- User DB ---
        user_db = {
            compromised_user: self._make_user(
                compromised_user, "Financial Analyst", "Finance",
                risk_score=0.3,
            ),
            benign_user: self._make_user(
                benign_user, "Sales Manager", "Sales",
                risk_score=0.1,
            ),
        }

        # --- Ground Truth ---
        ground_truth = GroundTruth(
            alert_classifications={
                "ALT-TLT-001": AlertClassification.TRUE_POSITIVE,
                "ALT-TLT-002": AlertClassification.TRUE_POSITIVE,
                "ALT-TLT-003": AlertClassification.TRUE_POSITIVE,
                "ALT-TLT-004": AlertClassification.TRUE_POSITIVE,
                "ALT-TLT-005": AlertClassification.TRUE_POSITIVE,
                "ALT-TLT-006": AlertClassification.FALSE_POSITIVE,
                "ALT-TLT-007": AlertClassification.FALSE_POSITIVE,
                "ALT-TLT-008": AlertClassification.FALSE_POSITIVE,
            },
            true_positive_ids=[
                "ALT-TLT-001", "ALT-TLT-002", "ALT-TLT-003",
                "ALT-TLT-004", "ALT-TLT-005",
            ],
            false_positive_ids=[
                "ALT-TLT-006", "ALT-TLT-007", "ALT-TLT-008",
            ],
            benign_tp_ids=[],
            kill_chain_order=[
                "ALT-TLT-001", "ALT-TLT-002", "ALT-TLT-003",
                "ALT-TLT-004", "ALT-TLT-005",
            ],
            expected_techniques={
                "ALT-TLT-001": ["T1566.001"],
                "ALT-TLT-002": ["T1003.001"],
                "ALT-TLT-003": ["T1021.002"],
                "ALT-TLT-004": ["T1074.001"],
                "ALT-TLT-005": ["T1041"],
            },
            expected_response_actions={
                "ALT-TLT-001": [ResponseActionType.QUARANTINE_FILE, ResponseActionType.BLOCK_IP],
                "ALT-TLT-002": [ResponseActionType.DISABLE_ACCOUNT, ResponseActionType.ISOLATE_ENDPOINT],
                "ALT-TLT-003": [ResponseActionType.ISOLATE_ENDPOINT, ResponseActionType.BLOCK_IP],
                "ALT-TLT-004": [ResponseActionType.ISOLATE_ENDPOINT],
                "ALT-TLT-005": [ResponseActionType.BLOCK_IP, ResponseActionType.BLOCK_DOMAIN],
                "ALT-TLT-006": [ResponseActionType.NO_ACTION],
                "ALT-TLT-007": [ResponseActionType.NO_ACTION],
                "ALT-TLT-008": [ResponseActionType.NO_ACTION],
            },
            relevant_log_sources={
                "ALT-TLT-001": [LogSource.EMAIL_GATEWAY, LogSource.ENDPOINT],
                "ALT-TLT-002": [LogSource.ENDPOINT, LogSource.AUTH],
                "ALT-TLT-003": [LogSource.FIREWALL, LogSource.AUTH],
                "ALT-TLT-004": [LogSource.ENDPOINT],
                "ALT-TLT-005": [LogSource.FIREWALL, LogSource.PROXY],
                "ALT-TLT-006": [LogSource.DNS],
                "ALT-TLT-007": [LogSource.AUTH],
                "ALT-TLT-008": [LogSource.IDS],
            },
            relevant_indicators={
                "ALT-TLT-001": [attacker_ip, phishing_domain, phishing_hash],
                "ALT-TLT-002": [cred_dump_hash, workstation_ip],
                "ALT-TLT-003": [workstation_ip, dc_ip],
                "ALT-TLT-004": [staging_server_ip, dc_ip],
                "ALT-TLT-005": [exfil_ip, exfil_domain],
                "ALT-TLT-006": [noise_ip_1],
                "ALT-TLT-007": [noise_ip_2],
                "ALT-TLT-008": [monitoring_ip],
            },
            attack_chain_ids=[
                [
                    "ALT-TLT-001", "ALT-TLT-002", "ALT-TLT-003",
                    "ALT-TLT-004", "ALT-TLT-005",
                ]
            ],
            # Team-mode escalation ground truth
            required_escalations=["ALT-TLT-002", "ALT-TLT-003", "ALT-TLT-005"],
            required_containments={
                "ALT-TLT-002": ["disable_user"],
                "ALT-TLT-003": ["isolate_host"],
                "ALT-TLT-005": ["block_ioc"],
            },
            expected_manager_flags=["ALT-TLT-004"],  # Data staging often missed; Manager should catch it
        )

        return ScenarioConfig(
            scenario_id=f"team-lateral-team-{self.seed}",
            task_id="team_lateral_team",
            seed=self.seed,
            description=(
                "Team-mode lateral movement: Tier-1 triages 8 alerts, escalates 3, "
                "Tier-2 contains, Manager flags missed staging."
            ),
            max_steps=self.MAX_STEPS,
            alerts=[
                alert_001, alert_002, alert_003, alert_004,
                alert_005, alert_006, alert_007, alert_008,
            ],
            enrichment_db=enrichment_db,
            log_db=log_db,
            asset_db=asset_db,
            user_db=user_db,
            ground_truth=ground_truth,
        )
