"""
Lateral Movement Scenario Generator — Task 2 (Medium)
=======================================================
Generates 5 correlated alerts representing a complete attacker kill chain:

  Alert 1: Phishing email delivery (T1566.001)
  Alert 2: LSASS credential dump on compromised host (T1003.001, T1059.001)
  Alert 3: RDP lateral movement using stolen credentials (T1021.001, T1078)
  Alert 4: Data staging — large archive on file server (T1074.001, T1560.001)
  Alert 5: Exfiltration over C2 channel (T1041, T1071.001)

All 5 alerts are TRUE POSITIVEs. Adjacent alerts share at least one indicator
(IP, user, or hostname) to form a connected kill chain that the agent must discover.
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


class LateralMovementScenario(BaseScenario):
    """Medium task: 5-alert lateral movement kill chain."""

    MAX_STEPS = 30

    def generate(self) -> ScenarioConfig:
        # ---- Shared IOCs across the kill chain ----
        ext_c2_ip = self._public_ip()           # attacker's C2 server (appears in Alert 1 + Alert 5)
        ext_c2_domain = self._malicious_domain()
        phishing_domain = self._malicious_domain()
        phishing_ip = self._public_ip()

        dropper_hash = self._sha256()
        mimikatz_hash = self._sha256()
        archive_hash = self._sha256()

        victim_host1 = self._hostname("WORKSTATION")   # initial compromise
        victim_ip1 = self._private_ip()
        victim_username = self._username()              # compromised user (appears in Alerts 2, 3, 4)

        lateral_host = self._hostname("SERVER")         # target of lateral movement
        lateral_ip = self._private_ip()

        file_server = "FS-01"                           # file server for staging
        file_server_ip = self._private_ip()

        attacker_email = f"invoice@{phishing_domain}"

        # ---- Alert IDs ----
        aid1 = self._alert_id("LAT")
        aid2 = self._alert_id("LAT")
        aid3 = self._alert_id("LAT")
        aid4 = self._alert_id("LAT")
        aid5 = self._alert_id("LAT")
        all_ids = [aid1, aid2, aid3, aid4, aid5]

        # ---- Alert 1: Phishing Email ----
        alert1 = AlertMeta(
            alert_id=aid1,
            title="Phishing Email with Macro-Enabled Attachment",
            description=(
                f"Email gateway detected phishing email from {attacker_email} "
                f"containing a macro-enabled Word document. SPF and DKIM failed."
            ),
            severity=AlertSeverity.HIGH,
            source_system="Email Security Gateway",
            timestamp=self._timestamp(hours_ago=4.0),
            rule_triggered="PHISH_MACRO_ATTACHMENT_003",
            indicators={
                "ip": [phishing_ip, ext_c2_ip],
                "domain": [phishing_domain],
                "file_hash": [dropper_hash],
                "email": [attacker_email],
            },
            raw_log_snippet=f"FROM={attacker_email} ATTACH=Q4_Invoice.docm HASH={dropper_hash[:12]}...",
        )

        # ---- Alert 2: LSASS Credential Dump ----
        alert2 = AlertMeta(
            alert_id=aid2,
            title="Suspicious Process Accessing LSASS Memory (Credential Dumping)",
            description=(
                f"EDR detected a process matching Mimikatz signatures accessing "
                f"lsass.exe memory on {victim_host1}. Credential theft in progress."
            ),
            severity=AlertSeverity.CRITICAL,
            source_system="Endpoint Detection & Response",
            timestamp=self._timestamp(hours_ago=3.5),
            rule_triggered="CRED_DUMP_LSASS_001",
            indicators={
                "ip": [victim_ip1],
                "file_hash": [mimikatz_hash],
                "user": [victim_username],
            },
            raw_log_snippet=f"PROCESS=chrome_helper.exe HASH={mimikatz_hash[:12]}... ACCESS=lsass.exe FLAGS=0x1fffff",
        )

        # ---- Alert 3: RDP Lateral Movement ----
        alert3 = AlertMeta(
            alert_id=aid3,
            title="Anomalous RDP Lateral Movement — Unusual User/Host Pair",
            description=(
                f"Auth logs show {victim_username} authenticating via RDP to {lateral_host} "
                f"— a server this user has never accessed. Login from {victim_ip1}."
            ),
            severity=AlertSeverity.HIGH,
            source_system="SIEM Correlation",
            timestamp=self._timestamp(hours_ago=3.0),
            rule_triggered="LATERAL_RDP_NEW_HOST_007",
            indicators={
                "ip": [victim_ip1, lateral_ip],
                "user": [victim_username],
            },
            raw_log_snippet=f"USER={victim_username} SRC={victim_ip1} DST={lateral_ip} PORT=3389 AUTH=NTLM SUCCESS",
        )

        # ---- Alert 4: Data Staging ----
        alert4 = AlertMeta(
            alert_id=aid4,
            title="Large Archive File Created on File Server",
            description=(
                f"DLP detected creation of a large compressed archive (2.8GB) on {file_server} "
                f"by {victim_username}. File contains documents from multiple departments."
            ),
            severity=AlertSeverity.MEDIUM,
            source_system="DLP / File Activity Monitor",
            timestamp=self._timestamp(hours_ago=2.0),
            rule_triggered="DLP_LARGE_ARCHIVE_003",
            indicators={
                "ip": [lateral_ip],
                "file_hash": [archive_hash],
                "user": [victim_username],
                "hostname": [file_server],
            },
            raw_log_snippet=f"USER={victim_username} HOST={file_server} FILE=backup_2024.zip SIZE=2.8GB",
        )

        # ---- Alert 5: Data Exfiltration ----
        alert5 = AlertMeta(
            alert_id=aid5,
            title="Large Outbound HTTPS Transfer to Known Malicious IP",
            description=(
                f"Firewall detected sustained large outbound HTTPS transfer from "
                f"{lateral_host} to {ext_c2_ip} (known C2 server). "
                f"3.1GB transferred over 45 minutes."
            ),
            severity=AlertSeverity.CRITICAL,
            source_system="Next-Gen Firewall",
            timestamp=self._timestamp(hours_ago=1.5),
            rule_triggered="EXFIL_LARGE_TRANSFER_C2_002",
            indicators={
                "ip": [lateral_ip, ext_c2_ip],
                "domain": [ext_c2_domain],
                "hostname": [file_server],
            },
            raw_log_snippet=f"SRC={lateral_ip} DST={ext_c2_ip} PORT=443 BYTES_OUT=3338035200 DURATION=2700s",
        )

        # ---- Enrichment DB ----
        enrichment_db = {
            phishing_ip: self._make_enrichment_result(
                phishing_ip, IndicatorType.IP, malicious=True,
                confidence=0.88, threat_score=82,
                threat_type="phishing", geo="Nigeria",
                tags=["phishing-actor", "smtp-relay"],
            ),
            phishing_domain: self._make_enrichment_result(
                phishing_domain, IndicatorType.DOMAIN, malicious=True,
                confidence=0.90, threat_score=85,
                threat_type="phishing",
                tags=["newly-registered", "lookalike-domain"],
                whois="Registered 2 days ago.",
            ),
            dropper_hash: self._make_enrichment_result(
                dropper_hash, IndicatorType.FILE_HASH, malicious=True,
                confidence=0.97, threat_score=95,
                threat_type="malware",
                tags=["macro-dropper", "office-macro"],
                malware=["Trickbot-dropper"],
            ),
            attacker_email: self._make_enrichment_result(
                attacker_email, IndicatorType.EMAIL, malicious=True,
                confidence=0.82, threat_score=75,
                threat_type="phishing",
                tags=["phishing-sender"],
            ),
            ext_c2_ip: self._make_enrichment_result(
                ext_c2_ip, IndicatorType.IP, malicious=True,
                confidence=0.96, threat_score=94,
                threat_type="command-and-control",
                geo="Netherlands",
                tags=["c2-server", "cobalt-strike-beacon"],
                malware=["CobaltStrike"],
            ),
            ext_c2_domain: self._make_enrichment_result(
                ext_c2_domain, IndicatorType.DOMAIN, malicious=True,
                confidence=0.95, threat_score=91,
                threat_type="command-and-control",
                tags=["c2", "cobalt-strike"],
                malware=["CobaltStrike"],
            ),
            mimikatz_hash: self._make_enrichment_result(
                mimikatz_hash, IndicatorType.FILE_HASH, malicious=True,
                confidence=0.99, threat_score=100,
                threat_type="credential-dumping",
                tags=["mimikatz", "credential-theft", "lsass-dump"],
                malware=["Mimikatz"],
            ),
            archive_hash: self._make_enrichment_result(
                archive_hash, IndicatorType.FILE_HASH, malicious=False,
                confidence=0.3, threat_score=15,
                tags=["archive", "zip", "potential-staging"],
            ),
        }

        # ---- Log DB ----
        log_db = self._empty_log_db(all_ids)

        # Alert 1 logs: Email gateway + endpoint execution
        log_db[LogSource.EMAIL_GATEWAY.value][aid1] = [
            self._make_log_entry(
                LogSource.EMAIL_GATEWAY, "email_received",
                hours_ago=4.0, src_ip=phishing_ip, user=victim_username,
                action="delivered",
                details={
                    "from": attacker_email, "attachment": "Q4_Invoice.docm",
                    "attachment_hash": dropper_hash,
                    "spf": "fail", "dkim": "fail", "dmarc": "fail",
                    "macro_detected": True,
                }
            ),
        ]
        log_db[LogSource.ENDPOINT.value][aid1] = [
            self._make_log_entry(
                LogSource.ENDPOINT, "macro_execution",
                hours_ago=3.8, hostname=victim_host1, user=victim_username,
                action="execute",
                details={
                    "process": "WINWORD.EXE", "document": "Q4_Invoice.docm",
                    "hash": dropper_hash, "macro_behavior": "downloads_payload",
                    "hostname": victim_host1,
                }
            ),
        ]

        # Alert 2 logs: EDR + IDS
        log_db[LogSource.ENDPOINT.value][aid2] = [
            self._make_log_entry(
                LogSource.ENDPOINT, "lsass_memory_access",
                hours_ago=3.5, hostname=victim_host1, user=victim_username,
                action="read_memory", severity="critical",
                details={
                    "process": "chrome_helper.exe",
                    "hash": mimikatz_hash,
                    "target_process": "lsass.exe",
                    "access_flags": "0x1fffff",
                    "hostname": victim_host1,
                }
            ),
            self._make_log_entry(
                LogSource.ENDPOINT, "powershell_execution",
                hours_ago=3.4, hostname=victim_host1, user=victim_username,
                action="execute",
                details={
                    "cmdline": "powershell -enc JABzAGUAYwByAGUAdAAgAD0AIAB...",
                    "encoded": True, "hostname": victim_host1,
                }
            ),
        ]
        log_db[LogSource.IDS.value][aid2] = [
            self._make_log_entry(
                LogSource.IDS, "signature_match",
                hours_ago=3.5, src_ip=victim_ip1, severity="critical",
                details={
                    "signature": "INDICATOR-COMPROMISE Mimikatz credential dumper",
                    "category": "Trojan-Activity",
                    "priority": 1,
                }
            ),
        ]

        # Alert 3 logs: Auth + Firewall
        log_db[LogSource.AUTH.value][aid3] = [
            self._make_log_entry(
                LogSource.AUTH, "rdp_login",
                hours_ago=3.0, src_ip=victim_ip1, dst_ip=lateral_ip,
                user=victim_username, hostname=lateral_host,
                action="login_success",
                details={
                    "auth_type": "NTLM",
                    "src_host": victim_host1,
                    "dst_host": lateral_host,
                    "first_time_user_host": True,
                    "port": 3389,
                }
            ),
        ]
        log_db[LogSource.FIREWALL.value][aid3] = [
            self._make_log_entry(
                LogSource.FIREWALL, "connection_allowed",
                hours_ago=3.0, src_ip=victim_ip1, dst_ip=lateral_ip,
                action="allow",
                details={"dst_port": 3389, "protocol": "TCP"}
            ),
        ]

        # Alert 4 logs: Endpoint (file creation) + Firewall
        log_db[LogSource.ENDPOINT.value][aid4] = [
            self._make_log_entry(
                LogSource.ENDPOINT, "file_created",
                hours_ago=2.0, hostname=file_server, user=victim_username,
                src_ip=lateral_ip,
                details={
                    "file_path": "C:\\Temp\\backup_2024.zip",
                    "file_size_bytes": 2986981376,
                    "tool_used": "7za.exe",
                    "hash": archive_hash,
                    "hostname": file_server,
                }
            ),
        ]
        log_db[LogSource.AUTH.value][aid4] = [
            self._make_log_entry(
                LogSource.AUTH, "smb_access",
                hours_ago=2.1, src_ip=lateral_ip, dst_ip=file_server_ip,
                user=victim_username, hostname=file_server,
                action="share_accessed",
                details={
                    "share": "\\\\FS-01\\Finance", "access": "read+write",
                }
            ),
        ]

        # Alert 5 logs: Firewall + Proxy + IDS
        log_db[LogSource.FIREWALL.value][aid5] = [
            self._make_log_entry(
                LogSource.FIREWALL, "large_outbound_transfer",
                hours_ago=1.5, src_ip=lateral_ip, dst_ip=ext_c2_ip,
                action="allowed", severity="critical",
                details={
                    "dst_port": 443, "protocol": "TCP",
                    "bytes_out": 3338035200, "duration_seconds": 2700,
                    "avg_throughput_mbps": 9.9,
                }
            ),
        ]
        log_db[LogSource.PROXY.value][aid5] = [
            self._make_log_entry(
                LogSource.PROXY, "https_request",
                hours_ago=1.5, src_ip=lateral_ip, hostname=file_server,
                dst_ip=ext_c2_ip,
                details={
                    "url": f"https://{ext_c2_domain}/upload",
                    "method": "POST",
                    "bytes_out": 3338035200,
                    "user_agent": "WinHTTP/1.0",
                }
            ),
        ]
        log_db[LogSource.IDS.value][aid5] = [
            self._make_log_entry(
                LogSource.IDS, "signature_match",
                hours_ago=1.5, src_ip=lateral_ip, dst_ip=ext_c2_ip,
                severity="critical",
                details={
                    "signature": "MALWARE-CNC CobaltStrike beacon POST",
                    "category": "Malware-CnC",
                    "priority": 1,
                }
            ),
        ]

        # ---- Asset DB ----
        asset_db = {
            victim_host1: self._make_asset(
                victim_host1, "workstation", victim_username,
                "Finance", victim_ip1, criticality="medium"
            ),
            lateral_host: self._make_asset(
                lateral_host, "application_server", "svc.app",
                "IT", lateral_ip, criticality="high", os="Windows Server 2019"
            ),
            file_server: self._make_asset(
                file_server, "file_server", "svc.storage",
                "IT", file_server_ip, criticality="critical", os="Windows Server 2019"
            ),
        }

        # ---- User DB ----
        user_db = {
            victim_username: self._make_user(
                victim_username, "Finance Manager", "Finance",
                risk_score=0.45,
            ),
        }

        # ---- Ground Truth ----
        ground_truth = GroundTruth(
            alert_classifications={
                aid1: AlertClassification.TRUE_POSITIVE,
                aid2: AlertClassification.TRUE_POSITIVE,
                aid3: AlertClassification.TRUE_POSITIVE,
                aid4: AlertClassification.TRUE_POSITIVE,
                aid5: AlertClassification.TRUE_POSITIVE,
            },
            true_positive_ids=all_ids,
            false_positive_ids=[],
            benign_tp_ids=[],
            expected_techniques={
                aid1: ["T1566.001", "T1204.002"],
                aid2: ["T1003.001", "T1059.001"],
                aid3: ["T1021.001", "T1078"],
                aid4: ["T1074.001", "T1560.001"],
                aid5: ["T1041", "T1071.001"],
            },
            expected_response_actions={
                aid1: [ResponseActionType.ISOLATE_ENDPOINT, ResponseActionType.QUARANTINE_FILE],
                aid2: [ResponseActionType.ISOLATE_ENDPOINT, ResponseActionType.RESET_PASSWORD],
                aid3: [ResponseActionType.DISABLE_ACCOUNT, ResponseActionType.REVOKE_SESSIONS],
                aid4: [ResponseActionType.ISOLATE_ENDPOINT, ResponseActionType.QUARANTINE_FILE],
                aid5: [ResponseActionType.BLOCK_IP, ResponseActionType.BLOCK_DOMAIN],
            },
            kill_chain_order=[aid1, aid2, aid3, aid4, aid5],
            relevant_log_sources={
                aid1: [LogSource.EMAIL_GATEWAY, LogSource.ENDPOINT],
                aid2: [LogSource.ENDPOINT, LogSource.IDS],
                aid3: [LogSource.AUTH, LogSource.FIREWALL],
                aid4: [LogSource.ENDPOINT, LogSource.AUTH],
                aid5: [LogSource.FIREWALL, LogSource.PROXY, LogSource.IDS],
            },
            relevant_indicators={
                aid1: [phishing_ip, phishing_domain, dropper_hash, attacker_email],
                aid2: [mimikatz_hash],
                aid3: [],  # correlation via shared user/host
                aid4: [archive_hash],
                aid5: [ext_c2_ip, ext_c2_domain],
            },
            attack_chain_ids=[[aid1, aid2, aid3, aid4, aid5]],
        )

        return ScenarioConfig(
            scenario_id=f"lateral-movement-{self.seed}",
            task_id="lateral_movement",
            seed=self.seed,
            description=(
                "Multi-alert lateral movement kill chain. Phishing → LSASS dump → "
                "RDP lateral move → data staging → exfiltration."
            ),
            max_steps=self.MAX_STEPS,
            alerts=[alert1, alert2, alert3, alert4, alert5],
            enrichment_db=enrichment_db,
            log_db=log_db,
            asset_db=asset_db,
            user_db=user_db,
            ground_truth=ground_truth,
        )
