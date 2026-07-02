"""
Insider Threat Scenario Generator -- Task 4 (Expert)
======================================================
Generates 30 alerts in shuffled order:
  - Chain A (insider): Unauthorized DB access -> Data export -> Cloud upload (3 alerts)
  - Chain B (compromised vendor): VPN anomaly -> Service account abuse -> Config changes (3 alerts)
  - Chain C (disgruntled employee): After-hours access -> Mass file deletion -> USB exfil (3 alerts)
  - 5 benign true positives (IT maintenance, pentest, backup jobs, etc.)
  - 16 false positives (DNS noise, scanner alerts, CDN anomalies, geoblocking, etc.)

The agent must triage all 30, dismiss FPs efficiently, uncover 3 hidden attack chains,
and correctly identify insider threat patterns.
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


class InsiderThreatScenario(BaseScenario):
    """Expert task: 30-alert insider threat investigation."""

    MAX_STEPS = 80

    def generate(self) -> ScenarioConfig:
        all_alerts = []
        all_ids = []
        enrichment_db = {}
        log_db = {}
        asset_db = {}
        user_db = {}

        # Build each alert group
        chain_a_alerts, chain_a_data = self._build_chain_a()
        chain_b_alerts, chain_b_data = self._build_chain_b()
        chain_c_alerts, chain_c_data = self._build_chain_c()
        btp_alerts, btp_data = self._build_benign_tps()
        fp_alerts, fp_data = self._build_false_positives()

        all_alert_groups = (
            chain_a_alerts + chain_b_alerts + chain_c_alerts
            + btp_alerts + fp_alerts
        )

        # Shuffle so alerts appear in random order
        self.rng.shuffle(all_alert_groups)

        for alert in all_alert_groups:
            all_alerts.append(alert)
            all_ids.append(alert.alert_id)

        # Initialize full log DB
        for source in LogSource:
            log_db[source.value] = {aid: [] for aid in all_ids}

        # Merge data from each group
        for data in [chain_a_data, chain_b_data, chain_c_data, btp_data, fp_data]:
            enrichment_db.update(data.get("enrichment_db", {}))
            for source, alert_map in data.get("log_db", {}).items():
                for aid, entries in alert_map.items():
                    log_db[source][aid] = entries
            asset_db.update(data.get("asset_db", {}))
            user_db.update(data.get("user_db", {}))

        # Build ground truth
        chain_a_ids = [a.alert_id for a in chain_a_alerts]
        chain_b_ids = [a.alert_id for a in chain_b_alerts]
        chain_c_ids = [a.alert_id for a in chain_c_alerts]
        btp_ids = [a.alert_id for a in btp_alerts]
        fp_ids = [a.alert_id for a in fp_alerts]
        true_positive_ids = chain_a_ids + chain_b_ids + chain_c_ids

        alert_classifications = {}
        for aid in true_positive_ids:
            alert_classifications[aid] = AlertClassification.TRUE_POSITIVE
        for aid in btp_ids:
            alert_classifications[aid] = AlertClassification.BENIGN_TRUE_POSITIVE
        for aid in fp_ids:
            alert_classifications[aid] = AlertClassification.FALSE_POSITIVE

        expected_techniques = {}
        expected_techniques.update(chain_a_data.get("expected_techniques", {}))
        expected_techniques.update(chain_b_data.get("expected_techniques", {}))
        expected_techniques.update(chain_c_data.get("expected_techniques", {}))
        expected_techniques.update(btp_data.get("expected_techniques", {}))

        expected_response_actions = {}
        expected_response_actions.update(chain_a_data.get("expected_response_actions", {}))
        expected_response_actions.update(chain_b_data.get("expected_response_actions", {}))
        expected_response_actions.update(chain_c_data.get("expected_response_actions", {}))
        for aid in btp_ids + fp_ids:
            expected_response_actions[aid] = [ResponseActionType.NO_ACTION]

        relevant_log_sources = {}
        relevant_log_sources.update(chain_a_data.get("relevant_log_sources", {}))
        relevant_log_sources.update(chain_b_data.get("relevant_log_sources", {}))
        relevant_log_sources.update(chain_c_data.get("relevant_log_sources", {}))
        relevant_log_sources.update(btp_data.get("relevant_log_sources", {}))
        for aid in fp_ids:
            relevant_log_sources[aid] = [LogSource.FIREWALL, LogSource.DNS]

        relevant_indicators = {}
        relevant_indicators.update(chain_a_data.get("relevant_indicators", {}))
        relevant_indicators.update(chain_b_data.get("relevant_indicators", {}))
        relevant_indicators.update(chain_c_data.get("relevant_indicators", {}))
        for aid in btp_ids + fp_ids:
            relevant_indicators[aid] = []

        ground_truth = GroundTruth(
            alert_classifications=alert_classifications,
            true_positive_ids=true_positive_ids,
            false_positive_ids=fp_ids,
            benign_tp_ids=btp_ids,
            expected_techniques=expected_techniques,
            expected_response_actions=expected_response_actions,
            kill_chain_order=chain_a_ids,  # Chain A is the primary ordered kill chain
            relevant_log_sources=relevant_log_sources,
            relevant_indicators=relevant_indicators,
            attack_chain_ids=[chain_a_ids, chain_b_ids, chain_c_ids],
        )

        return ScenarioConfig(
            scenario_id=f"insider-threat-{self.seed}",
            task_id="insider_threat",
            seed=self.seed,
            description=(
                "30-alert queue: 9 TPs (3 attack chains — insider data theft, "
                "compromised vendor, disgruntled employee), 5 benign TPs, 16 FPs. "
                "Uncover the insider threats hidden in heavy SOC noise."
            ),
            max_steps=self.MAX_STEPS,
            alerts=all_alerts,
            enrichment_db=enrichment_db,
            log_db=log_db,
            asset_db=asset_db,
            user_db=user_db,
            ground_truth=ground_truth,
        )

    # ------------------------------------------------------------------
    # Chain A: Insider Data Theft
    #   Unauthorized DB access -> Data export -> Cloud upload
    # ------------------------------------------------------------------

    def _build_chain_a(self):
        insider_user = self._username()
        insider_host = self._hostname("WORKSTATION")
        insider_ip = self._private_ip()
        db_server = self._hostname("DBSRV")
        db_ip = self._private_ip()
        cloud_domain = self._malicious_domain()
        cloud_ip = self._public_ip()
        export_hash = self._sha256()

        aid1 = self._alert_id("INS")
        aid2 = self._alert_id("INS")
        aid3 = self._alert_id("INS")

        alert1 = AlertMeta(
            alert_id=aid1,
            title="Unauthorized Database Access -- Sensitive Tables Queried",
            description=(
                f"User {insider_user} executed 47 SELECT queries against restricted "
                f"HR and Finance tables on {db_server} from {insider_host}. "
                f"User's role does not include database access privileges."
            ),
            severity=AlertSeverity.HIGH,
            source_system="Database Activity Monitor",
            timestamp=self._timestamp(hours_ago=6.0),
            rule_triggered="DAM_UNAUTHORIZED_QUERY_001",
            indicators={"ip": [insider_ip, db_ip], "user": [insider_user], "hostname": [db_server]},
            raw_log_snippet=(
                f"USER={insider_user} SRC={insider_ip} DST={db_ip} DB=HR_PAYROLL "
                f"QUERIES=47 TABLES=employees,salaries,ssn DURATION=1200s"
            ),
        )

        alert2 = AlertMeta(
            alert_id=aid2,
            title="Bulk Data Export to Local CSV Files",
            description=(
                f"DLP detected {insider_user} exporting 12,000 records from the "
                f"HR database to local CSV files on {insider_host}. Files total 340MB."
            ),
            severity=AlertSeverity.HIGH,
            source_system="Data Loss Prevention",
            timestamp=self._timestamp(hours_ago=5.5),
            rule_triggered="DLP_BULK_EXPORT_002",
            indicators={
                "ip": [insider_ip],
                "file_hash": [export_hash],
                "user": [insider_user],
                "hostname": [insider_host],
            },
            raw_log_snippet=(
                f"USER={insider_user} HOST={insider_host} FILES=hr_export_*.csv "
                f"RECORDS=12000 SIZE=340MB HASH={export_hash[:12]}..."
            ),
        )

        alert3 = AlertMeta(
            alert_id=aid3,
            title="Suspicious Cloud Upload -- Personal Storage Service",
            description=(
                f"CASB detected {insider_user} uploading 340MB to personal cloud storage "
                f"at {cloud_domain}. Corporate policy prohibits personal cloud uploads."
            ),
            severity=AlertSeverity.CRITICAL,
            source_system="Cloud Access Security Broker",
            timestamp=self._timestamp(hours_ago=5.0),
            rule_triggered="CASB_PERSONAL_CLOUD_UPLOAD_001",
            indicators={
                "ip": [insider_ip, cloud_ip],
                "domain": [cloud_domain],
                "user": [insider_user],
                "file_hash": [export_hash],
            },
            raw_log_snippet=(
                f"USER={insider_user} DST={cloud_domain} UPLOAD_SIZE=340MB "
                f"SRC_IP={insider_ip} DST_IP={cloud_ip} POLICY_VIOLATION=personal_cloud"
            ),
        )

        enrichment_db = {
            cloud_domain: self._make_enrichment_result(
                cloud_domain, IndicatorType.DOMAIN, malicious=True, confidence=0.72,
                threat_score=65, threat_type="data-exfiltration",
                tags=["personal-cloud", "file-sharing", "policy-violation"],
                whois="Registered 6 months ago. Known personal storage service.",
            ),
            cloud_ip: self._make_enrichment_result(
                cloud_ip, IndicatorType.IP, malicious=False, confidence=0.40,
                threat_score=30, threat_type="cloud-hosting",
                geo="United States", tags=["cloud-provider", "file-sharing"],
            ),
            export_hash: self._make_enrichment_result(
                export_hash, IndicatorType.FILE_HASH, malicious=False, confidence=0.20,
                threat_score=10, tags=["csv-data", "bulk-export"],
            ),
        }

        log_db = {}
        for source in LogSource:
            log_db[source.value] = {aid1: [], aid2: [], aid3: []}

        log_db[LogSource.AUTH.value][aid1] = [
            self._make_log_entry(LogSource.AUTH, "database_login", hours_ago=6.0,
                src_ip=insider_ip, dst_ip=db_ip, user=insider_user,
                hostname=db_server,
                details={
                    "db_name": "HR_PAYROLL", "auth_type": "windows_integrated",
                    "role": "read_only", "unauthorized": True,
                }),
        ]
        log_db[LogSource.CLOUD_TRAIL.value][aid1] = [
            self._make_log_entry(LogSource.CLOUD_TRAIL, "sql_query_burst", hours_ago=6.0,
                user=insider_user, src_ip=insider_ip,
                details={
                    "query_count": 47, "tables": ["employees", "salaries", "ssn"],
                    "duration_seconds": 1200,
                }),
        ]
        log_db[LogSource.ENDPOINT.value][aid2] = [
            self._make_log_entry(LogSource.ENDPOINT, "file_created", hours_ago=5.5,
                hostname=insider_host, user=insider_user,
                details={
                    "file_path": f"C:\\Users\\{insider_user}\\Documents\\hr_export_full.csv",
                    "file_size_bytes": 356515840, "records": 12000,
                    "hash": export_hash,
                }),
        ]
        log_db[LogSource.PROXY.value][aid3] = [
            self._make_log_entry(LogSource.PROXY, "https_upload", hours_ago=5.0,
                src_ip=insider_ip, dst_ip=cloud_ip, user=insider_user,
                details={
                    "url": f"https://{cloud_domain}/upload",
                    "method": "POST", "bytes_out": 356515840,
                    "user_agent": "Chrome/120.0",
                    "category": "personal-cloud-storage",
                }),
        ]

        user_db = {
            insider_user: self._make_user(
                insider_user, "Financial Analyst", "Finance",
                risk_score=0.65, access_level="standard",
            ),
        }
        asset_db = {
            insider_host: self._make_asset(
                insider_host, "workstation", insider_user, "Finance", insider_ip,
            ),
            db_server: self._make_asset(
                db_server, "database_server", "svc.database", "IT", db_ip,
                criticality="critical", os="Windows Server 2019",
            ),
        }

        return [alert1, alert2, alert3], {
            "enrichment_db": enrichment_db,
            "log_db": log_db,
            "asset_db": asset_db,
            "user_db": user_db,
            "expected_techniques": {
                aid1: ["T1078"],
                aid2: ["T1530"],
                aid3: ["T1567"],
            },
            "expected_response_actions": {
                aid1: [ResponseActionType.DISABLE_ACCOUNT, ResponseActionType.REVOKE_SESSIONS],
                aid2: [ResponseActionType.ISOLATE_ENDPOINT, ResponseActionType.QUARANTINE_FILE],
                aid3: [ResponseActionType.BLOCK_DOMAIN, ResponseActionType.DISABLE_ACCOUNT],
            },
            "relevant_log_sources": {
                aid1: [LogSource.AUTH, LogSource.CLOUD_TRAIL],
                aid2: [LogSource.ENDPOINT],
                aid3: [LogSource.PROXY],
            },
            "relevant_indicators": {
                aid1: [insider_ip, db_ip],
                aid2: [export_hash],
                aid3: [cloud_domain, cloud_ip, export_hash],
            },
        }

    # ------------------------------------------------------------------
    # Chain B: Compromised Vendor
    #   VPN anomaly -> Service account abuse -> Config changes
    # ------------------------------------------------------------------

    def _build_chain_b(self):
        vendor_vpn_ip = self._public_ip()
        vendor_user = "vendor.msp01"
        svc_account = "svc.monitoring"
        target_server = self._hostname("SERVER")
        target_ip = self._private_ip()
        config_server = self._hostname("CFGSRV")
        config_ip = self._private_ip()
        tool_hash = self._sha256()

        aid1 = self._alert_id("INS")
        aid2 = self._alert_id("INS")
        aid3 = self._alert_id("INS")

        alert1 = AlertMeta(
            alert_id=aid1,
            title="VPN Login Anomaly -- Vendor Account from Unusual Location",
            description=(
                f"Vendor account {vendor_user} connected via VPN from {vendor_vpn_ip} "
                f"(Eastern Europe). Previous logins were exclusively from US-based IPs. "
                f"Session established at 02:30 AM local time."
            ),
            severity=AlertSeverity.HIGH,
            source_system="VPN Gateway",
            timestamp=self._timestamp(hours_ago=8.0),
            rule_triggered="VPN_GEO_ANOMALY_001",
            indicators={"ip": [vendor_vpn_ip], "user": [vendor_user]},
            raw_log_snippet=(
                f"USER={vendor_user} SRC={vendor_vpn_ip} VPN=connected "
                f"GEO=Romania USUAL_GEO=US TIME=02:30"
            ),
        )

        alert2 = AlertMeta(
            alert_id=aid2,
            title="Service Account Privilege Escalation",
            description=(
                f"Service account {svc_account} granted Domain Admin privileges "
                f"by {vendor_user} on {target_server}. This service account "
                f"previously had only monitoring read-only access."
            ),
            severity=AlertSeverity.CRITICAL,
            source_system="Active Directory Monitor",
            timestamp=self._timestamp(hours_ago=7.5),
            rule_triggered="AD_PRIV_ESCALATION_003",
            indicators={
                "ip": [target_ip],
                "user": [vendor_user, svc_account],
                "hostname": [target_server],
            },
            raw_log_snippet=(
                f"ACTOR={vendor_user} TARGET={svc_account} ACTION=add_to_group "
                f"GROUP=Domain Admins HOST={target_server}"
            ),
        )

        alert3 = AlertMeta(
            alert_id=aid3,
            title="Critical Configuration Changes -- Firewall Rules Modified",
            description=(
                f"Service account {svc_account} modified 8 firewall rules on "
                f"{config_server}, opening inbound ports 4444, 5555, 8443. "
                f"No change ticket associated. Changes made outside maintenance window."
            ),
            severity=AlertSeverity.CRITICAL,
            source_system="Configuration Management",
            timestamp=self._timestamp(hours_ago=7.0),
            rule_triggered="CONFIG_UNAUTH_CHANGE_002",
            indicators={
                "ip": [config_ip],
                "user": [svc_account],
                "hostname": [config_server],
                "file_hash": [tool_hash],
            },
            raw_log_snippet=(
                f"USER={svc_account} HOST={config_server} ACTION=modify_firewall "
                f"RULES_CHANGED=8 PORTS_OPENED=4444,5555,8443 TICKET=NONE"
            ),
        )

        enrichment_db = {
            vendor_vpn_ip: self._make_enrichment_result(
                vendor_vpn_ip, IndicatorType.IP, malicious=True, confidence=0.85,
                threat_score=80, threat_type="suspicious-vpn",
                geo="Romania", tags=["geo-anomaly", "vendor-compromise", "eastern-europe"],
            ),
            tool_hash: self._make_enrichment_result(
                tool_hash, IndicatorType.FILE_HASH, malicious=True, confidence=0.88,
                threat_score=85, threat_type="attack-tool",
                tags=["config-manipulation", "firewall-bypass"],
                malware=["Custom-ConfigTool"],
            ),
        }

        log_db = {}
        for source in LogSource:
            log_db[source.value] = {aid1: [], aid2: [], aid3: []}

        log_db[LogSource.AUTH.value][aid1] = [
            self._make_log_entry(LogSource.AUTH, "vpn_login", hours_ago=8.0,
                src_ip=vendor_vpn_ip, user=vendor_user,
                details={
                    "vpn_type": "SSL", "geo": "Romania",
                    "usual_geo": "United States", "time_anomaly": True,
                    "first_seen_country": True,
                }),
        ]
        log_db[LogSource.AUTH.value][aid2] = [
            self._make_log_entry(LogSource.AUTH, "privilege_escalation", hours_ago=7.5,
                user=vendor_user, hostname=target_server, src_ip=target_ip,
                details={
                    "target_account": svc_account,
                    "group_added": "Domain Admins",
                    "previous_groups": ["Monitoring-ReadOnly"],
                }),
        ]
        log_db[LogSource.FIREWALL.value][aid3] = [
            self._make_log_entry(LogSource.FIREWALL, "rule_modification", hours_ago=7.0,
                user=svc_account, hostname=config_server,
                details={
                    "rules_modified": 8,
                    "ports_opened": [4444, 5555, 8443],
                    "change_ticket": None,
                    "maintenance_window": False,
                }),
        ]
        log_db[LogSource.ENDPOINT.value][aid3] = [
            self._make_log_entry(LogSource.ENDPOINT, "tool_execution", hours_ago=7.0,
                hostname=config_server, user=svc_account,
                details={
                    "process": "fw_config_tool.exe",
                    "hash": tool_hash,
                    "unsigned": True,
                }),
        ]

        user_db = {
            vendor_user: self._make_user(
                vendor_user, "MSP Technician", "External Vendor",
                risk_score=0.50, access_level="limited",
            ),
            svc_account: self._make_user(
                svc_account, "Service Account", "IT",
                risk_score=0.15, is_privileged=False, access_level="monitoring",
            ),
        }
        asset_db = {
            target_server: self._make_asset(
                target_server, "domain_controller", "svc.ad", "IT", target_ip,
                criticality="critical", os="Windows Server 2022",
            ),
            config_server: self._make_asset(
                config_server, "firewall_management", "svc.network", "IT", config_ip,
                criticality="critical", os="Linux",
            ),
        }

        return [alert1, alert2, alert3], {
            "enrichment_db": enrichment_db,
            "log_db": log_db,
            "asset_db": asset_db,
            "user_db": user_db,
            "expected_techniques": {
                aid1: ["T1133"],
                aid2: ["T1098"],
                aid3: ["T1562.004"],
            },
            "expected_response_actions": {
                aid1: [ResponseActionType.BLOCK_IP, ResponseActionType.DISABLE_ACCOUNT],
                aid2: [ResponseActionType.DISABLE_ACCOUNT, ResponseActionType.REVOKE_SESSIONS],
                aid3: [ResponseActionType.ISOLATE_ENDPOINT, ResponseActionType.DISABLE_ACCOUNT],
            },
            "relevant_log_sources": {
                aid1: [LogSource.AUTH],
                aid2: [LogSource.AUTH],
                aid3: [LogSource.FIREWALL, LogSource.ENDPOINT],
            },
            "relevant_indicators": {
                aid1: [vendor_vpn_ip],
                aid2: [],
                aid3: [tool_hash],
            },
        }

    # ------------------------------------------------------------------
    # Chain C: Disgruntled Employee
    #   After-hours access -> Mass file deletion -> USB exfiltration
    # ------------------------------------------------------------------

    def _build_chain_c(self):
        disgruntled_user = self._username()
        disgruntled_host = self._hostname("WORKSTATION")
        disgruntled_ip = self._private_ip()
        file_server = self._hostname("FILESRV")
        file_server_ip = self._private_ip()
        usb_hash = self._sha256()

        aid1 = self._alert_id("INS")
        aid2 = self._alert_id("INS")
        aid3 = self._alert_id("INS")

        alert1 = AlertMeta(
            alert_id=aid1,
            title="After-Hours Badge Access -- Terminated Employee on PIP",
            description=(
                f"User {disgruntled_user} badged into the building at 11:45 PM on a Saturday. "
                f"HR records show this employee is on a Performance Improvement Plan (PIP) "
                f"with termination scheduled for next week."
            ),
            severity=AlertSeverity.MEDIUM,
            source_system="Physical Access Control",
            timestamp=self._timestamp(hours_ago=10.0),
            rule_triggered="PHYS_AFTERHOURS_PIP_001",
            indicators={"user": [disgruntled_user], "ip": [disgruntled_ip]},
            raw_log_snippet=(
                f"USER={disgruntled_user} BADGE=GRANTED DOOR=MAIN_ENTRANCE "
                f"TIME=23:45 DAY=Saturday HR_FLAG=PIP"
            ),
        )

        alert2 = AlertMeta(
            alert_id=aid2,
            title="Mass File Deletion -- Engineering Share",
            description=(
                f"User {disgruntled_user} deleted 2,847 files from the Engineering "
                f"shared drive on {file_server} in 15 minutes. Deletion pattern "
                f"suggests automated scripting (rm -rf style)."
            ),
            severity=AlertSeverity.CRITICAL,
            source_system="File Integrity Monitor",
            timestamp=self._timestamp(hours_ago=9.5),
            rule_triggered="FIM_MASS_DELETE_001",
            indicators={
                "ip": [disgruntled_ip, file_server_ip],
                "user": [disgruntled_user],
                "hostname": [file_server],
            },
            raw_log_snippet=(
                f"USER={disgruntled_user} HOST={file_server} ACTION=delete "
                f"FILES=2847 SHARE=\\\\{file_server}\\Engineering DURATION=900s"
            ),
        )

        alert3 = AlertMeta(
            alert_id=aid3,
            title="USB Mass Storage Device -- Large Data Copy Detected",
            description=(
                f"EDR detected {disgruntled_user} copying 18GB of data to a USB "
                f"mass storage device on {disgruntled_host}. USB device not in "
                f"approved device list. Data includes source code repositories."
            ),
            severity=AlertSeverity.CRITICAL,
            source_system="Endpoint Detection & Response",
            timestamp=self._timestamp(hours_ago=9.0),
            rule_triggered="EDR_USB_EXFIL_002",
            indicators={
                "ip": [disgruntled_ip],
                "user": [disgruntled_user],
                "hostname": [disgruntled_host],
                "file_hash": [usb_hash],
            },
            raw_log_snippet=(
                f"USER={disgruntled_user} HOST={disgruntled_host} "
                f"DEVICE=USB_MASS_STORAGE VENDOR=SanDisk SIZE=18GB "
                f"FILES_COPIED=source_repos HASH={usb_hash[:12]}..."
            ),
        )

        enrichment_db = {
            usb_hash: self._make_enrichment_result(
                usb_hash, IndicatorType.FILE_HASH, malicious=False, confidence=0.30,
                threat_score=20, tags=["archive", "source-code", "usb-copy"],
            ),
        }

        log_db = {}
        for source in LogSource:
            log_db[source.value] = {aid1: [], aid2: [], aid3: []}

        log_db[LogSource.AUTH.value][aid1] = [
            self._make_log_entry(LogSource.AUTH, "badge_access", hours_ago=10.0,
                user=disgruntled_user,
                details={
                    "door": "MAIN_ENTRANCE", "time": "23:45",
                    "day_of_week": "Saturday", "hr_flag": "PIP",
                    "termination_scheduled": True,
                }),
        ]
        log_db[LogSource.AUTH.value][aid2] = [
            self._make_log_entry(LogSource.AUTH, "smb_access", hours_ago=9.5,
                src_ip=disgruntled_ip, dst_ip=file_server_ip,
                user=disgruntled_user, hostname=file_server,
                action="delete",
                details={
                    "share": f"\\\\{file_server}\\Engineering",
                    "files_deleted": 2847, "duration_seconds": 900,
                    "pattern": "recursive",
                }),
        ]
        log_db[LogSource.ENDPOINT.value][aid2] = [
            self._make_log_entry(LogSource.ENDPOINT, "script_execution", hours_ago=9.5,
                hostname=disgruntled_host, user=disgruntled_user,
                details={
                    "script": "cleanup.ps1",
                    "cmdline": "powershell -ExecutionPolicy Bypass -File cleanup.ps1",
                    "action": "mass_delete",
                }),
        ]
        log_db[LogSource.ENDPOINT.value][aid3] = [
            self._make_log_entry(LogSource.ENDPOINT, "usb_device_connected", hours_ago=9.0,
                hostname=disgruntled_host, user=disgruntled_user,
                details={
                    "device_type": "USB Mass Storage",
                    "vendor": "SanDisk", "serial": "4C530001240811",
                    "approved_device": False,
                    "data_copied_gb": 18,
                    "file_types": [".py", ".java", ".go", ".sql", ".env"],
                }),
        ]

        user_db = {
            disgruntled_user: self._make_user(
                disgruntled_user, "Senior Software Engineer", "Engineering",
                risk_score=0.80, access_level="standard",
            ),
        }
        asset_db = {
            disgruntled_host: self._make_asset(
                disgruntled_host, "workstation", disgruntled_user,
                "Engineering", disgruntled_ip,
            ),
            file_server: self._make_asset(
                file_server, "file_server", "svc.storage", "IT", file_server_ip,
                criticality="high", os="Windows Server 2019",
            ),
        }

        return [alert1, alert2, alert3], {
            "enrichment_db": enrichment_db,
            "log_db": log_db,
            "asset_db": asset_db,
            "user_db": user_db,
            "expected_techniques": {
                aid1: ["T1078"],
                aid2: ["T1485"],
                aid3: ["T1052.001"],
            },
            "expected_response_actions": {
                aid1: [ResponseActionType.DISABLE_ACCOUNT],
                aid2: [ResponseActionType.DISABLE_ACCOUNT, ResponseActionType.ISOLATE_ENDPOINT],
                aid3: [ResponseActionType.ISOLATE_ENDPOINT, ResponseActionType.DISABLE_ACCOUNT],
            },
            "relevant_log_sources": {
                aid1: [LogSource.AUTH],
                aid2: [LogSource.AUTH, LogSource.ENDPOINT],
                aid3: [LogSource.ENDPOINT],
            },
            "relevant_indicators": {
                aid1: [],
                aid2: [disgruntled_ip, file_server_ip],
                aid3: [usb_hash],
            },
        }

    # ------------------------------------------------------------------
    # Benign True Positives (5 alerts)
    # ------------------------------------------------------------------

    def _build_benign_tps(self):
        aids = [self._alert_id("BTP") for _ in range(5)]

        pentest_ip = self._private_ip()
        pentest_user = "pentest.red"
        admin_user = self._username()
        admin_host = self._hostname("SERVER")
        admin_ip = self._private_ip()
        backup_user = "svc.backup"
        backup_host = self._hostname("BACKUP")
        backup_ip = self._private_ip()
        it_user = "it.ops"
        devops_user = self._username()

        alert_pentest = AlertMeta(
            alert_id=aids[0],
            title="Internal Penetration Test -- Credential Spraying Detected",
            description=(
                f"Internal host {pentest_ip} performing credential spray against "
                f"Active Directory. Authorized red team exercise (PENTEST-2024-Q4)."
            ),
            severity=AlertSeverity.HIGH,
            source_system="IDS",
            timestamp=self._timestamp(hours_ago=2.0),
            rule_triggered="IDS_CRED_SPRAY_003",
            indicators={"ip": [pentest_ip], "user": [pentest_user]},
            raw_log_snippet=f"SRC={pentest_ip} TARGET=AD ATTEMPTS=500 USER={pentest_user}",
        )

        alert_backup = AlertMeta(
            alert_id=aids[1],
            title="Large Data Transfer to Backup Server -- Scheduled Job",
            description=(
                f"Service account {backup_user} transferred 450GB to {backup_host}. "
                f"This matches the nightly backup schedule (02:00-04:00 AM)."
            ),
            severity=AlertSeverity.MEDIUM,
            source_system="DLP",
            timestamp=self._timestamp(hours_ago=1.5),
            rule_triggered="DLP_LARGE_TRANSFER_001",
            indicators={"ip": [backup_ip], "user": [backup_user], "hostname": [backup_host]},
            raw_log_snippet=f"USER={backup_user} DST={backup_host} SIZE=450GB SCHEDULE=nightly",
        )

        alert_admin = AlertMeta(
            alert_id=aids[2],
            title="Admin Account Mass Group Policy Update",
            description=(
                f"Admin {admin_user} pushed GPO updates to 200 machines from {admin_host}. "
                f"Change ticket CHG-7291 approved for quarterly policy refresh."
            ),
            severity=AlertSeverity.MEDIUM,
            source_system="Active Directory Monitor",
            timestamp=self._timestamp(hours_ago=1.0),
            rule_triggered="AD_MASS_GPO_UPDATE_001",
            indicators={"ip": [admin_ip], "user": [admin_user], "hostname": [admin_host]},
            raw_log_snippet=f"USER={admin_user} HOST={admin_host} ACTION=gpo_update TARGETS=200 TICKET=CHG-7291",
        )

        alert_maint = AlertMeta(
            alert_id=aids[3],
            title="Scheduled Vulnerability Scan -- Internal Scanner",
            description=(
                f"IT ops account {it_user} triggered weekly vulnerability scan across "
                f"production subnet. Scan authorized per IT-SEC-POL-004."
            ),
            severity=AlertSeverity.LOW,
            source_system="IDS",
            timestamp=self._timestamp(hours_ago=0.5),
            rule_triggered="IDS_INTERNAL_SCAN_001",
            indicators={"user": [it_user]},
            raw_log_snippet=f"USER={it_user} ACTION=vuln_scan SUBNET=10.0.0.0/8 POLICY=IT-SEC-POL-004",
        )

        alert_deploy = AlertMeta(
            alert_id=aids[4],
            title="SSH Key Rotation -- DevOps Automation",
            description=(
                f"DevOps user {devops_user} rotated SSH keys across 50 production servers. "
                f"Matches quarterly key rotation schedule per compliance requirement."
            ),
            severity=AlertSeverity.LOW,
            source_system="SIEM",
            timestamp=self._timestamp(hours_ago=0.3),
            rule_triggered="AUTH_SSH_KEY_ROTATION_001",
            indicators={"user": [devops_user]},
            raw_log_snippet=f"USER={devops_user} ACTION=ssh_key_rotate SERVERS=50 SCHEDULE=quarterly",
        )

        enrichment_db = {
            pentest_ip: self._make_enrichment_result(
                pentest_ip, IndicatorType.IP, malicious=False, confidence=0.95,
                threat_score=0, tags=["internal", "authorized-red-team"],
            ),
        }

        log_db = {}
        for source in LogSource:
            log_db[source.value] = {aid: [] for aid in aids}

        log_db[LogSource.IDS.value][aids[0]] = [
            self._make_log_entry(LogSource.IDS, "credential_spray", hours_ago=2.0,
                src_ip=pentest_ip,
                details={"authorized": True, "ticket": "PENTEST-2024-Q4", "attempts": 500}),
        ]
        log_db[LogSource.FIREWALL.value][aids[1]] = [
            self._make_log_entry(LogSource.FIREWALL, "large_transfer", hours_ago=1.5,
                src_ip=backup_ip, user=backup_user, hostname=backup_host,
                action="allow",
                details={"size_gb": 450, "schedule": "nightly", "approved": True}),
        ]
        log_db[LogSource.AUTH.value][aids[2]] = [
            self._make_log_entry(LogSource.AUTH, "gpo_update", hours_ago=1.0,
                user=admin_user, hostname=admin_host, src_ip=admin_ip,
                details={"targets": 200, "ticket": "CHG-7291", "approved": True}),
        ]
        log_db[LogSource.IDS.value][aids[3]] = [
            self._make_log_entry(LogSource.IDS, "vulnerability_scan", hours_ago=0.5,
                user=it_user,
                details={"subnet": "10.0.0.0/8", "policy": "IT-SEC-POL-004", "authorized": True}),
        ]
        log_db[LogSource.AUTH.value][aids[4]] = [
            self._make_log_entry(LogSource.AUTH, "ssh_key_rotation", hours_ago=0.3,
                user=devops_user,
                details={"servers": 50, "schedule": "quarterly", "compliance": True}),
        ]

        user_db = {
            pentest_user: self._make_user(pentest_user, "Red Team Lead", "IT Security",
                risk_score=0.05, is_privileged=True, access_level="admin"),
            backup_user: self._make_user(backup_user, "Backup Service Account", "IT",
                risk_score=0.02, is_privileged=True, access_level="service"),
            admin_user: self._make_user(admin_user, "Systems Administrator", "IT",
                risk_score=0.08, is_privileged=True, access_level="admin"),
            it_user: self._make_user(it_user, "IT Operations", "IT",
                risk_score=0.05, is_privileged=True, access_level="admin"),
            devops_user: self._make_user(devops_user, "DevOps Engineer", "Engineering",
                risk_score=0.10, is_privileged=True, access_level="admin"),
        }

        return [alert_pentest, alert_backup, alert_admin, alert_maint, alert_deploy], {
            "enrichment_db": enrichment_db,
            "log_db": log_db,
            "asset_db": {},
            "user_db": user_db,
            "expected_techniques": {},
            "relevant_log_sources": {
                aids[0]: [LogSource.IDS],
                aids[1]: [LogSource.FIREWALL],
                aids[2]: [LogSource.AUTH],
                aids[3]: [LogSource.IDS],
                aids[4]: [LogSource.AUTH],
            },
        }

    # ------------------------------------------------------------------
    # False Positives (16 alerts)
    # ------------------------------------------------------------------

    def _build_false_positives(self):
        fps = []
        enrichment_db = {}
        log_db = {}

        fp_templates = [
            ("FW_GEO_BLOCK", "Geoblocked Connection from Sanctioned Country",
             AlertSeverity.LOW,
             "Firewall auto-blocked inbound connection from sanctioned country. No internal system reached.",
             LogSource.FIREWALL),
            ("FW_SCANNER", "External Vulnerability Scanner -- Shodan",
             AlertSeverity.INFO,
             "Internet-facing host probed by known security research scanner (Shodan). Auto-blocked.",
             LogSource.FIREWALL),
            ("DNS_NRD_001", "DNS Query to Newly-Registered Domain -- SaaS Vendor",
             AlertSeverity.INFO,
             "Internal host resolved newly-registered domain belonging to approved SaaS vendor.",
             LogSource.DNS),
            ("DNS_DGA_FP", "DNS Query Pattern Matches DGA -- False Positive",
             AlertSeverity.LOW,
             "DNS query pattern triggered DGA detector. Domain is legitimate CDN subdomain.",
             LogSource.DNS),
            ("DNS_TUN_FP", "DNS Tunneling Heuristic Triggered -- Antimalware Update",
             AlertSeverity.LOW,
             "High volume of DNS TXT queries from antimalware agent fetching signature updates.",
             LogSource.DNS),
            ("AV_FP_001", "Antivirus Alert -- Developer Tool Flagged",
             AlertSeverity.LOW,
             "AV flagged legitimate development tool (Postman binary). File in approved software list.",
             LogSource.ENDPOINT),
            ("AV_FP_002", "Antivirus Alert -- Python Script Heuristic",
             AlertSeverity.LOW,
             "AV heuristic triggered on legitimate Python automation script. SHA256 clean in threat feeds.",
             LogSource.ENDPOINT),
            ("AV_FP_003", "Antivirus Alert -- Installer Package",
             AlertSeverity.LOW,
             "AV flagged signed installer for approved application. Digital signature verified.",
             LogSource.ENDPOINT),
            ("CDN_ANOMALY", "CDN Traffic Spike -- Marketing Campaign",
             AlertSeverity.INFO,
             "Outbound CDN traffic spiked 400%. Caused by marketing email campaign with tracked links.",
             LogSource.PROXY),
            ("AUTH_SVC_001", "Service Account Authentication Burst -- Batch Job",
             AlertSeverity.LOW,
             "Service account performing scheduled batch authentication. Part of nightly ETL pipeline.",
             LogSource.AUTH),
            ("AUTH_SVC_002", "Service Account Lockout -- Password Rotation",
             AlertSeverity.LOW,
             "Service account locked out due to password rotation lag. IT ticket #INC-8812 open.",
             LogSource.AUTH),
            ("AUTH_GEO_001", "Off-Hours Login -- Remote Employee Different Timezone",
             AlertSeverity.LOW,
             "Employee account logging in at 03:00 AM local time. Employee is remote in UTC+9 timezone.",
             LogSource.AUTH),
            ("FW_CDN_FP", "Geoblocked Outbound to CDN Node -- Cloudflare",
             AlertSeverity.LOW,
             "Internal host connecting to Cloudflare CDN node incorrectly flagged as foreign IP.",
             LogSource.FIREWALL),
            ("IDS_FP_001", "IDS Signature Match -- HTTP Header False Positive",
             AlertSeverity.INFO,
             "IDS signature triggered on benign HTTP header containing SQL-like string in User-Agent.",
             LogSource.IDS),
            ("WAF_FP_001", "WAF Rule Triggered -- Automated Security Scanner",
             AlertSeverity.INFO,
             "WAF blocked automated pen test from authorized Qualys scanner. No successful exploitation.",
             LogSource.FIREWALL),
            ("PROXY_FP_001", "Proxy Alert -- Uncategorized Domain",
             AlertSeverity.INFO,
             "Proxy flagged access to uncategorized domain. Domain is new corporate partnership site.",
             LogSource.PROXY),
        ]

        for rule, title, severity, description, primary_source in fp_templates:
            aid = self._alert_id("FP")
            fps.append(AlertMeta(
                alert_id=aid,
                title=title,
                description=description,
                severity=severity,
                source_system="SIEM",
                timestamp=self._timestamp(hours_ago=self.rng.uniform(0.1, 10.0)),
                rule_triggered=rule,
                indicators={"ip": [self._public_ip()]},
                raw_log_snippet=f"Rule: {rule} | {description[:80]}",
            ))
            for source in LogSource:
                log_db.setdefault(source.value, {})[aid] = []
            log_db[primary_source.value][aid] = [
                self._make_log_entry(primary_source, "event",
                    hours_ago=self.rng.uniform(0.1, 10.0),
                    action="block" if "BLOCK" in rule or "FW" in rule else "allow",
                    details={"rule": rule, "false_positive": True}),
            ]

        return fps, {
            "enrichment_db": enrichment_db,
            "log_db": log_db,
            "asset_db": {},
            "user_db": {},
        }
