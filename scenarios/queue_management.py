"""
Queue Management Scenario Generator — Task 3 (Hard)
====================================================
Generates 20 alerts in shuffled order:
  - 2 real multi-stage attack chains (5 TP alerts total)
  - 3 benign true positives (legitimate activity triggering detection rules)
  - 12 false positives (realistic SOC noise)

The agent must triage all 20, dismiss FPs efficiently, and surface the real attacks.
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


class QueueManagementScenario(BaseScenario):
    """Hard task: alert queue with 20 mixed alerts."""

    MAX_STEPS = 60

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
        btp_alerts, btp_data = self._build_benign_tps()
        fp_alerts, fp_data = self._build_false_positives()

        all_alert_groups = chain_a_alerts + chain_b_alerts + btp_alerts + fp_alerts

        # Shuffle so alerts appear in random order
        self.rng.shuffle(all_alert_groups)

        for alert in all_alert_groups:
            all_alerts.append(alert)
            all_ids.append(alert.alert_id)

        # Initialize full log DB
        for source in LogSource:
            log_db[source.value] = {aid: [] for aid in all_ids}

        # Merge data from each group
        for data in [chain_a_data, chain_b_data, btp_data, fp_data]:
            enrichment_db.update(data.get("enrichment_db", {}))
            for source, alert_map in data.get("log_db", {}).items():
                for aid, entries in alert_map.items():
                    log_db[source][aid] = entries
            asset_db.update(data.get("asset_db", {}))
            user_db.update(data.get("user_db", {}))

        # Build ground truth
        chain_a_ids = [a.alert_id for a in chain_a_alerts]
        chain_b_ids = [a.alert_id for a in chain_b_alerts]
        btp_ids = [a.alert_id for a in btp_alerts]
        fp_ids = [a.alert_id for a in fp_alerts]
        true_positive_ids = chain_a_ids + chain_b_ids

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
        expected_techniques.update(btp_data.get("expected_techniques", {}))

        expected_response_actions = {}
        expected_response_actions.update(chain_a_data.get("expected_response_actions", {}))
        expected_response_actions.update(chain_b_data.get("expected_response_actions", {}))
        for aid in btp_ids + fp_ids:
            expected_response_actions[aid] = [ResponseActionType.NO_ACTION]

        relevant_log_sources = {}
        relevant_log_sources.update(chain_a_data.get("relevant_log_sources", {}))
        relevant_log_sources.update(chain_b_data.get("relevant_log_sources", {}))
        relevant_log_sources.update(btp_data.get("relevant_log_sources", {}))
        for aid in fp_ids:
            relevant_log_sources[aid] = [LogSource.EMAIL_GATEWAY, LogSource.FIREWALL]

        relevant_indicators = {}
        relevant_indicators.update(chain_a_data.get("relevant_indicators", {}))
        relevant_indicators.update(chain_b_data.get("relevant_indicators", {}))
        for aid in btp_ids + fp_ids:
            relevant_indicators[aid] = []

        ground_truth = GroundTruth(
            alert_classifications=alert_classifications,
            true_positive_ids=true_positive_ids,
            false_positive_ids=fp_ids,
            benign_tp_ids=btp_ids,
            expected_techniques=expected_techniques,
            expected_response_actions=expected_response_actions,
            kill_chain_order=chain_a_ids,  # Chain A is the ordered kill chain
            relevant_log_sources=relevant_log_sources,
            relevant_indicators=relevant_indicators,
            attack_chain_ids=[chain_a_ids, chain_b_ids],
        )

        return ScenarioConfig(
            scenario_id=f"queue-management-{self.seed}",
            task_id="queue_management",
            seed=self.seed,
            description=(
                "20-alert queue: 5 TPs (2 attack chains), 3 benign TPs, 12 FPs. "
                "Find the attacks hidden in the noise."
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
    # Chain A: Credential Stuffing → Account Takeover → Exfiltration (3 alerts)
    # ------------------------------------------------------------------

    def _build_chain_a(self):
        attacker_ip = self._tor_exit_ip()
        victim_user = self._username()
        # These draws are kept (not bound) so the deterministic RNG sequence
        # is preserved; this chain only surfaces attacker_ip / victim_user /
        # c2_domain in its alert payloads.
        self._hostname("WORKSTATION")
        self._private_ip()
        self._public_ip()
        c2_domain = self._malicious_domain()
        tool_hash = self._sha256()

        aid1 = self._alert_id("QA")
        aid2 = self._alert_id("QA")
        aid3 = self._alert_id("QA")

        alert1 = AlertMeta(
            alert_id=aid1,
            title="Credential Stuffing Attack Detected",
            description=f"Multiple failed login attempts from {attacker_ip} against user accounts. 847 failed attempts in 10 minutes.",
            severity=AlertSeverity.HIGH,
            source_system="Auth Intelligence",
            timestamp=self._timestamp(hours_ago=3.0),
            rule_triggered="AUTH_BRUTE_FORCE_003",
            indicators={"ip": [attacker_ip], "user": [victim_user]},
            raw_log_snippet=f"SRC={attacker_ip} FAILED_ATTEMPTS=847 TARGET_USER={victim_user} DURATION=600s",
        )

        alert2 = AlertMeta(
            alert_id=aid2,
            title="Successful Login After Multiple Failures — Possible Account Takeover",
            description=f"User {victim_user} successfully logged in from {attacker_ip} after 847 failures. First login from this IP.",
            severity=AlertSeverity.CRITICAL,
            source_system="SIEM Correlation",
            timestamp=self._timestamp(hours_ago=2.5),
            rule_triggered="AUTH_TAKEOVER_AFTER_BRUTE_001",
            indicators={"ip": [attacker_ip], "user": [victim_user]},
            raw_log_snippet=f"USER={victim_user} SRC={attacker_ip} AUTH=SUCCESS FIRST_SEEN_IP=true",
        )

        alert3 = AlertMeta(
            alert_id=aid3,
            title="Suspicious Cloud Storage Download — Mass File Access",
            description=f"User {victim_user} downloaded 4,200 files from SharePoint in 8 minutes using automated tooling.",
            severity=AlertSeverity.HIGH,
            source_system="Cloud Access Security Broker",
            timestamp=self._timestamp(hours_ago=2.0),
            rule_triggered="CASB_MASS_DOWNLOAD_001",
            indicators={"ip": [attacker_ip], "domain": [c2_domain], "user": [victim_user], "file_hash": [tool_hash]},
            raw_log_snippet=f"USER={victim_user} APP=SharePoint FILES_DOWNLOADED=4200 DURATION=480s SRC_IP={attacker_ip}",
        )

        enrichment_db = {
            attacker_ip: self._make_enrichment_result(
                attacker_ip, IndicatorType.IP, malicious=True, confidence=0.93, threat_score=90,
                threat_type="credential-stuffing", geo="Tor Exit Node",
                tags=["tor-exit", "credential-stuffing"],
            ),
            c2_domain: self._make_enrichment_result(
                c2_domain, IndicatorType.DOMAIN, malicious=True, confidence=0.88, threat_score=84,
                threat_type="data-exfiltration", tags=["c2", "data-theft"],
            ),
            tool_hash: self._make_enrichment_result(
                tool_hash, IndicatorType.FILE_HASH, malicious=True, confidence=0.91, threat_score=88,
                threat_type="rclone", tags=["data-theft-tool", "rclone"],
                malware=["RClone-variant"],
            ),
        }

        log_db = {}
        for source in LogSource:
            log_db[source.value] = {aid1: [], aid2: [], aid3: []}

        log_db[LogSource.AUTH.value][aid1] = [
            self._make_log_entry(LogSource.AUTH, "login_failure", hours_ago=3.0,
                src_ip=attacker_ip, user=victim_user,
                details={"failed_count": 847, "user_agent": "python-requests/2.28.0"}),
        ]
        log_db[LogSource.AUTH.value][aid2] = [
            self._make_log_entry(LogSource.AUTH, "login_success", hours_ago=2.5,
                src_ip=attacker_ip, user=victim_user,
                details={"first_seen_ip": True, "mfa_bypassed": False}),
        ]
        log_db[LogSource.CLOUD_TRAIL.value][aid3] = [
            self._make_log_entry(LogSource.CLOUD_TRAIL, "mass_download", hours_ago=2.0,
                user=victim_user, src_ip=attacker_ip,
                details={"app": "SharePoint", "file_count": 4200, "duration_seconds": 480}),
        ]

        user_db = {victim_user: self._make_user(victim_user, "Software Engineer", "Engineering", risk_score=0.3)}

        return [alert1, alert2, alert3], {
            "enrichment_db": enrichment_db,
            "log_db": log_db,
            "asset_db": {},
            "user_db": user_db,
            "expected_techniques": {
                aid1: ["T1110.003"],
                aid2: ["T1078"],
                aid3: ["T1567"],
            },
            "expected_response_actions": {
                aid1: [ResponseActionType.BLOCK_IP],
                aid2: [ResponseActionType.DISABLE_ACCOUNT, ResponseActionType.REVOKE_SESSIONS],
                aid3: [ResponseActionType.REVOKE_SESSIONS, ResponseActionType.RESET_PASSWORD],
            },
            "relevant_log_sources": {
                aid1: [LogSource.AUTH],
                aid2: [LogSource.AUTH],
                aid3: [LogSource.CLOUD_TRAIL],
            },
            "relevant_indicators": {
                aid1: [attacker_ip],
                aid2: [attacker_ip],
                aid3: [c2_domain, tool_hash],
            },
        }

    # ------------------------------------------------------------------
    # Chain B: Spearphishing → Persistence (2 alerts)
    # ------------------------------------------------------------------

    def _build_chain_b(self):
        ext_ip = self._public_ip()
        malicious_domain = self._malicious_domain()
        payload_hash = self._sha256()
        victim_user = self._username()
        victim_host = self._hostname("LAPTOP")
        victim_ip = self._private_ip()

        aid1 = self._alert_id("QB")
        aid2 = self._alert_id("QB")

        alert1 = AlertMeta(
            alert_id=aid1,
            title="Spearphishing Link Clicked — Credential Harvesting Page",
            description=f"User {victim_user} clicked a spearphishing link leading to a credential harvesting page on {malicious_domain}.",
            severity=AlertSeverity.HIGH,
            source_system="Email Security + Proxy",
            timestamp=self._timestamp(hours_ago=5.0),
            rule_triggered="PHISH_LINK_CLICK_002",
            indicators={"ip": [ext_ip], "domain": [malicious_domain], "user": [victim_user]},
            raw_log_snippet=f"USER={victim_user} URL=https://{malicious_domain}/login CATEGORY=phishing",
        )

        alert2 = AlertMeta(
            alert_id=aid2,
            title="Scheduled Task Created for Persistence",
            description=f"New scheduled task 'WindowsUpdateHelper' created on {victim_host} by {victim_user}. Runs PowerShell payload daily.",
            severity=AlertSeverity.HIGH,
            source_system="EDR",
            timestamp=self._timestamp(hours_ago=4.5),
            rule_triggered="PERSIST_SCHEDULED_TASK_001",
            indicators={"ip": [victim_ip], "file_hash": [payload_hash], "user": [victim_user]},
            raw_log_snippet=f"HOST={victim_host} USER={victim_user} TASK=WindowsUpdateHelper CMD=powershell -enc ...",
        )

        enrichment_db = {
            ext_ip: self._make_enrichment_result(
                ext_ip, IndicatorType.IP, malicious=True, confidence=0.87, threat_score=83,
                threat_type="phishing", geo="Romania", tags=["phishing-host"],
            ),
            malicious_domain: self._make_enrichment_result(
                malicious_domain, IndicatorType.DOMAIN, malicious=True, confidence=0.93, threat_score=88,
                threat_type="phishing", tags=["credential-harvesting", "newly-registered"],
                whois="Registered 1 day ago.",
            ),
            payload_hash: self._make_enrichment_result(
                payload_hash, IndicatorType.FILE_HASH, malicious=True, confidence=0.89, threat_score=86,
                threat_type="malware", tags=["powershell-backdoor"],
            ),
        }

        log_db = {}
        for source in LogSource:
            log_db[source.value] = {aid1: [], aid2: []}
        log_db[LogSource.PROXY.value][aid1] = [
            self._make_log_entry(LogSource.PROXY, "url_visited", hours_ago=5.0,
                user=victim_user, src_ip=victim_ip, dst_ip=ext_ip,
                details={"url": f"https://{malicious_domain}/login", "category": "phishing"}),
        ]
        log_db[LogSource.ENDPOINT.value][aid2] = [
            self._make_log_entry(LogSource.ENDPOINT, "scheduled_task_created", hours_ago=4.5,
                hostname=victim_host, user=victim_user,
                details={"task_name": "WindowsUpdateHelper", "hash": payload_hash}),
        ]

        user_db = {victim_user: self._make_user(victim_user, "HR Coordinator", "HR", risk_score=0.2)}
        asset_db = {victim_host: self._make_asset(victim_host, "laptop", victim_user, "HR", victim_ip)}

        return [alert1, alert2], {
            "enrichment_db": enrichment_db,
            "log_db": log_db,
            "asset_db": asset_db,
            "user_db": user_db,
            "expected_techniques": {
                aid1: ["T1566.002"],
                aid2: ["T1053.005", "T1059.001"],
            },
            "expected_response_actions": {
                aid1: [ResponseActionType.BLOCK_DOMAIN, ResponseActionType.BLOCK_IP],
                aid2: [ResponseActionType.ISOLATE_ENDPOINT, ResponseActionType.QUARANTINE_FILE],
            },
            "relevant_log_sources": {
                aid1: [LogSource.PROXY],
                aid2: [LogSource.ENDPOINT],
            },
            "relevant_indicators": {
                aid1: [ext_ip, malicious_domain],
                aid2: [payload_hash],
            },
        }

    # ------------------------------------------------------------------
    # Benign True Positives (3 alerts)
    # ------------------------------------------------------------------

    def _build_benign_tps(self):
        aids = [self._alert_id("BTP") for _ in range(3)]

        pentest_ip = self._private_ip()
        pentest_user = "pentest.internal"
        admin_user = self._username()
        admin_host = self._hostname("SERVER")
        admin_ip = self._private_ip()
        it_user = "it.admin"

        alert_pentest = AlertMeta(
            alert_id=aids[0],
            title="Port Scan Detected — Multiple Hosts",
            description=f"Internal host {pentest_ip} performing rapid port scan across subnet. Authorized penetration test in progress.",
            severity=AlertSeverity.MEDIUM,
            source_system="IDS",
            timestamp=self._timestamp(hours_ago=1.0),
            rule_triggered="IDS_PORTSCAN_002",
            indicators={"ip": [pentest_ip], "user": [pentest_user]},
            raw_log_snippet=f"SRC={pentest_ip} SCAN_TYPE=SYN PORT_RANGE=1-65535 RATE=10000pps",
        )

        alert_psexec = AlertMeta(
            alert_id=aids[1],
            title="PsExec Remote Execution — Admin Maintenance Activity",
            description=f"Admin {admin_user} used PsExec to execute a maintenance script on {admin_host}. Change ticket #CHG-4821 open.",
            severity=AlertSeverity.MEDIUM,
            source_system="EDR",
            timestamp=self._timestamp(hours_ago=0.5),
            rule_triggered="EXEC_PSEXEC_REMOTE_001",
            indicators={"ip": [admin_ip], "user": [admin_user], "hostname": [admin_host]},
            raw_log_snippet=f"USER={admin_user} TOOL=PsExec TARGET={admin_host} TICKET=CHG-4821",
        )

        alert_pwd_reset = AlertMeta(
            alert_id=aids[2],
            title="Bulk Password Reset — IT Maintenance Window",
            description=f"IT admin {it_user} reset 47 passwords in 3 minutes. Scheduled quarterly password policy enforcement.",
            severity=AlertSeverity.LOW,
            source_system="Identity Provider",
            timestamp=self._timestamp(hours_ago=0.2),
            rule_triggered="AUTH_BULK_PASSWORD_RESET_001",
            indicators={"user": [it_user]},
            raw_log_snippet=f"ADMIN={it_user} ACTION=reset_password COUNT=47 DURATION=180s",
        )

        # Clean enrichment for these
        enrichment_db = {
            pentest_ip: self._make_enrichment_result(
                pentest_ip, IndicatorType.IP, malicious=False, confidence=0.95, threat_score=0,
                tags=["internal", "authorized-scanner"],
            ),
        }

        log_db = {}
        for source in LogSource:
            log_db[source.value] = {aid: [] for aid in aids}

        log_db[LogSource.IDS.value][aids[0]] = [
            self._make_log_entry(LogSource.IDS, "port_scan", hours_ago=1.0,
                src_ip=pentest_ip, details={"authorized": True, "ticket": "PENTEST-2024-Q4"}),
        ]
        log_db[LogSource.ENDPOINT.value][aids[1]] = [
            self._make_log_entry(LogSource.ENDPOINT, "psexec_execution", hours_ago=0.5,
                src_ip=admin_ip, hostname=admin_host, user=admin_user,
                details={"ticket": "CHG-4821", "authorized": True}),
        ]
        log_db[LogSource.AUTH.value][aids[2]] = [
            self._make_log_entry(LogSource.AUTH, "bulk_password_reset", hours_ago=0.2,
                user=it_user, details={"count": 47, "scheduled": True}),
        ]

        user_db = {
            pentest_user: self._make_user(pentest_user, "Penetration Tester", "IT Security",
                risk_score=0.05, is_privileged=True, access_level="admin"),
            admin_user: self._make_user(admin_user, "Systems Administrator", "IT",
                risk_score=0.08, is_privileged=True, access_level="admin"),
            it_user: self._make_user(it_user, "IT Administrator", "IT",
                risk_score=0.05, is_privileged=True, access_level="admin"),
        }

        return [alert_pentest, alert_psexec, alert_pwd_reset], {
            "enrichment_db": enrichment_db,
            "log_db": log_db,
            "asset_db": {},
            "user_db": user_db,
            "expected_techniques": {},
            "relevant_log_sources": {
                aids[0]: [LogSource.IDS],
                aids[1]: [LogSource.ENDPOINT],
                aids[2]: [LogSource.AUTH],
            },
        }

    # ------------------------------------------------------------------
    # False Positives (15 alerts)
    # ------------------------------------------------------------------

    def _build_false_positives(self):
        fps = []
        enrichment_db = {}
        log_db = {}

        fp_templates = [
            ("FW_GEO_BLOCK", "Geoblocked Connection from High-Risk Country", AlertSeverity.LOW,
             "Firewall auto-blocked inbound connection from sanctioned country. No internal system reached.", LogSource.FIREWALL),
            ("FW_SCANNER", "Known Vulnerability Scanner — Shodan/Censys", AlertSeverity.INFO,
             "Internet-facing host probed by known security research scanner (Shodan). Auto-blocked.", LogSource.FIREWALL),
            ("FW_GEO_BLOCK", "Geoblocked Outbound to CDN — False Positive", AlertSeverity.LOW,
             "Internal host connecting to content delivery network incorrectly flagged as foreign IP.", LogSource.FIREWALL),
            ("AV_FP_001", "Antivirus Alert — Known False Positive on Dev Tool", AlertSeverity.LOW,
             "AV flagged legitimate development tool (npm package). File is in approved software list.", LogSource.ENDPOINT),
            ("AV_FP_002", "Antivirus Alert — Python Script Flagged as Heuristic", AlertSeverity.LOW,
             "AV heuristic triggered on legitimate Python automation script. SHA256 not in threat feeds.", LogSource.ENDPOINT),
            ("AV_FP_003", "Antivirus Alert — Compressed Archive Flagged", AlertSeverity.LOW,
             "AV flagged password-protected zip as suspicious. Contents verified clean by IT.", LogSource.ENDPOINT),
            ("AV_FP_004", "Antivirus Alert — PDF with JavaScript Flagged", AlertSeverity.LOW,
             "AV flagged PDF with embedded JavaScript. Document is legitimate marketing collateral.", LogSource.ENDPOINT),
            ("AUTH_SVC_001", "Service Account Authentication Alert — Expected Behavior", AlertSeverity.LOW,
             "Service account performing scheduled authentication. Part of automated monitoring job.", LogSource.AUTH),
            ("AUTH_SVC_002", "Service Account Multiple Auth Failures — Config Issue", AlertSeverity.LOW,
             "Service account locked out due to password policy change. IT ticket open, not an attack.", LogSource.AUTH),
            ("AUTH_SVC_003", "Off-Hours Login — Contractor in Different Timezone", AlertSeverity.LOW,
             "Contractor account logging in during off-hours — contractor is in APAC timezone (UTC+8).", LogSource.AUTH),
            ("DNS_CDN_001", "DNS Query to Newly-Registered Domain — Legitimate SaaS", AlertSeverity.INFO,
             "Internal host queried domain for new SaaS tool approved by IT. Domain is legitimate.", LogSource.DNS),
            ("FW_NOISE_001", "WAF Rule Triggered — Automated Vulnerability Scanner", AlertSeverity.INFO,
             "WAF blocked automated vulnerability scan from external scanner. No successful requests.", LogSource.FIREWALL),
        ]

        for i, (rule, title, severity, description, primary_source) in enumerate(fp_templates):
            aid = self._alert_id("FP")
            fps.append(AlertMeta(
                alert_id=aid,
                title=title,
                description=description,
                severity=severity,
                source_system="SIEM",
                timestamp=self._timestamp(hours_ago=self.rng.uniform(0.1, 6.0)),
                rule_triggered=rule,
                indicators={"ip": [self._public_ip()]},
                raw_log_snippet=f"Rule: {rule} | {description[:80]}",
            ))
            for source in LogSource:
                log_db.setdefault(source.value, {})[aid] = []
            # Add minimal supporting log
            log_db[primary_source.value][aid] = [
                self._make_log_entry(primary_source, "event",
                    hours_ago=self.rng.uniform(0.1, 6.0),
                    action="block" if "BLOCK" in rule or "FW" in rule else "allow",
                    details={"rule": rule, "false_positive": True}),
            ]

        return fps, {
            "enrichment_db": enrichment_db,
            "log_db": log_db,
            "asset_db": {},
            "user_db": {},
        }
