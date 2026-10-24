"""
Cloud Account Compromise Scenario Generator — single-alert triage
=================================================================
Generates one cloud identity alert with full supporting evidence.

Two variants depending on seed parity:
  - TRUE_POSITIVE: impossible-travel console login from a malicious IP, then
                   IAM abuse (new access key), a bucket made public, a security
                   group opened to 0.0.0.0/0, and CloudTrail logging disabled.
  - FALSE_POSITIVE: an SSO-authenticated admin working from a new-but-legitimate
                    location (travel/VPN), whose API calls match an approved IaC
                    deploy window; MFA present, no logging tampering.

Mirrors the single-alert scenarios so it plugs into the existing graders. All
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


class CloudCompromiseScenario(BaseScenario):
    """Medium task: single cloud identity-compromise alert triage."""

    MAX_STEPS = 15

    def generate(self) -> ScenarioConfig:
        is_tp = (self.seed % 2 == 0) or (self.seed % 7 < 4)
        return self._generate_true_positive() if is_tp else self._generate_false_positive()

    # ------------------------------------------------------------------
    # True Positive Variant — compromised cloud credentials
    # ------------------------------------------------------------------

    def _generate_true_positive(self) -> ScenarioConfig:
        alert_id = self._alert_id("CLOUD")

        attacker_ip = self._tor_exit_ip()
        username = self._username()
        home_ip = self._public_ip()
        new_key_id = "AKIA" + "".join(self.rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", k=16))
        bucket = f"acmecorp-{self.rng.choice(['backups', 'billing', 'customer-data', 'hr'])}"

        alert = AlertMeta(
            alert_id=alert_id,
            title="Cloud Identity Compromise — Impossible Travel and IAM Abuse",
            description=(
                f"CloudTrail flagged console access for '{username}' from {attacker_ip} "
                f"~8 minutes after a login from {home_ip}, followed by creation of a new "
                f"access key, a bucket policy making '{bucket}' public, and CloudTrail "
                f"StopLogging. Suspected credential compromise."
            ),
            severity=AlertSeverity.CRITICAL,
            source_system="CloudTrail",
            timestamp=self._timestamp(hours_ago=0.5),
            rule_triggered="CLOUD_IMPOSSIBLE_TRAVEL_IAM_ABUSE_001",
            indicators={
                "ip": [attacker_ip],
                "user": [username],
            },
            raw_log_snippet=(
                f"eventSource=signin.amazonaws.com eventName=ConsoleLogin "
                f"userIdentity={username} sourceIP={attacker_ip} MFAUsed=No "
                f"followedBy=CreateAccessKey,PutBucketPolicy(public),StopLogging"
            ),
        )

        enrichment_db = {
            attacker_ip: self._make_enrichment_result(
                attacker_ip, IndicatorType.IP, malicious=True,
                confidence=0.9, threat_score=86,
                threat_type="anonymizer",
                geo="Russia",
                tags=["tor-exit-node", "anonymizer", "credential-abuse"],
                whois="Tor exit node; no legitimate corporate use.",
            ),
            username: self._make_enrichment_result(
                username, IndicatorType.USER, malicious=False,
                confidence=0.8, threat_score=55,
                threat_type="compromised-account",
                tags=["impossible-travel", "no-mfa-on-anomalous-login"],
            ),
        }

        log_db = self._empty_log_db([alert_id])

        # Auth — the two logins that make travel impossible.
        log_db[LogSource.AUTH.value][alert_id] = [
            self._make_log_entry(
                LogSource.AUTH, "console_login", hours_ago=0.65,
                src_ip=home_ip, user=username, action="success",
                details={"mfa": True, "geo": "United States", "user_agent": "Chrome/normal"},
            ),
            self._make_log_entry(
                LogSource.AUTH, "console_login", hours_ago=0.52,
                src_ip=attacker_ip, user=username, action="success", severity="high",
                details={"mfa": False, "geo": "Russia", "user_agent": "python-requests/2.x",
                         "impossible_travel": True},
            ),
        ]

        # CloudTrail — IAM abuse, public bucket, SG open, logging disabled.
        log_db[LogSource.CLOUD_TRAIL.value][alert_id] = [
            self._make_log_entry(
                LogSource.CLOUD_TRAIL, "CreateAccessKey", hours_ago=0.5,
                src_ip=attacker_ip, user=username, action="allowed", severity="high",
                details={"event_source": "iam.amazonaws.com", "new_access_key_id": new_key_id},
            ),
            self._make_log_entry(
                LogSource.CLOUD_TRAIL, "PutBucketPolicy", hours_ago=0.48,
                src_ip=attacker_ip, user=username, action="allowed", severity="critical",
                details={"event_source": "s3.amazonaws.com", "bucket": bucket,
                         "effect": "Allow", "principal": "*", "made_public": True},
            ),
            self._make_log_entry(
                LogSource.CLOUD_TRAIL, "AuthorizeSecurityGroupIngress", hours_ago=0.46,
                src_ip=attacker_ip, user=username, action="allowed", severity="high",
                details={"event_source": "ec2.amazonaws.com", "cidr": "0.0.0.0/0", "port": 22},
            ),
            self._make_log_entry(
                LogSource.CLOUD_TRAIL, "StopLogging", hours_ago=0.44,
                src_ip=attacker_ip, user=username, action="allowed", severity="critical",
                details={"event_source": "cloudtrail.amazonaws.com", "trail": "org-audit-trail"},
            ),
        ]

        user_db = {username: self._make_user(
            username, "Cloud Engineer", "Engineering", risk_score=0.3,
            is_privileged=True, access_level="admin",
        )}

        ground_truth = GroundTruth(
            alert_classifications={alert_id: AlertClassification.TRUE_POSITIVE},
            true_positive_ids=[alert_id],
            false_positive_ids=[],
            benign_tp_ids=[],
            expected_techniques={
                alert_id: ["T1078.004", "T1098.001", "T1562.008", "T1530"]
            },
            expected_response_actions={
                alert_id: [
                    ResponseActionType.DISABLE_ACCOUNT,
                    ResponseActionType.REVOKE_SESSIONS,
                    ResponseActionType.RESET_PASSWORD,
                    ResponseActionType.BLOCK_IP,
                ]
            },
            relevant_log_sources={alert_id: [LogSource.AUTH, LogSource.CLOUD_TRAIL]},
            relevant_indicators={alert_id: [attacker_ip, username]},
        )

        return ScenarioConfig(
            scenario_id=f"cloud-compromise-tp-{self.seed}",
            task_id="cloud_compromise",
            seed=self.seed,
            description="Single cloud identity alert — TRUE POSITIVE. Impossible-travel login and IAM abuse.",
            max_steps=self.MAX_STEPS,
            alerts=[alert],
            enrichment_db=enrichment_db,
            log_db=log_db,
            asset_db={},
            user_db=user_db,
            ground_truth=ground_truth,
        )

    # ------------------------------------------------------------------
    # False Positive Variant — legitimate admin from a new location
    # ------------------------------------------------------------------

    def _generate_false_positive(self) -> ScenarioConfig:
        alert_id = self._alert_id("CLOUD")

        username = self._username()
        travel_ip = self._public_ip()
        bucket = f"acmecorp-{self.rng.choice(['static-site', 'public-assets', 'releases'])}"

        alert = AlertMeta(
            alert_id=alert_id,
            title="Cloud Console Access from a New Location",
            description=(
                f"CloudTrail flagged console access for '{username}' from a new IP "
                f"{travel_ip}. Rule fired on the location change; the session is SSO+MFA "
                f"authenticated and the API calls match an approved IaC deploy window."
            ),
            severity=AlertSeverity.LOW,
            source_system="CloudTrail",
            timestamp=self._timestamp(hours_ago=0.4),
            rule_triggered="CLOUD_NEW_LOCATION_LOGIN_002",
            indicators={"ip": [travel_ip], "user": [username]},
            raw_log_snippet=(
                f"eventSource=signin.amazonaws.com eventName=ConsoleLogin "
                f"userIdentity={username} sourceIP={travel_ip} MFAUsed=Yes SSO=Okta "
                f"changeWindow=approved"
            ),
        )

        enrichment_db = {
            travel_ip: self._make_enrichment_result(
                travel_ip, IndicatorType.IP, malicious=False,
                confidence=0.9, threat_score=3,
                geo="Germany",
                tags=["residential-isp", "corporate-vpn-egress"],
                whois="Residential ISP range; matches employee travel notice.",
            ),
            username: self._make_enrichment_result(
                username, IndicatorType.USER, malicious=False,
                confidence=0.95, threat_score=2,
                tags=["mfa-enabled", "sso", "known-admin"],
            ),
        }

        log_db = self._empty_log_db([alert_id])
        log_db[LogSource.AUTH.value][alert_id] = [
            self._make_log_entry(
                LogSource.AUTH, "console_login", hours_ago=0.45,
                src_ip=travel_ip, user=username, action="success",
                details={"mfa": True, "sso": "Okta", "geo": "Germany",
                         "user_agent": "Chrome/normal", "impossible_travel": False},
            ),
        ]
        log_db[LogSource.CLOUD_TRAIL.value][alert_id] = [
            self._make_log_entry(
                LogSource.CLOUD_TRAIL, "PutObject", hours_ago=0.4,
                src_ip=travel_ip, user=username, action="allowed",
                details={"event_source": "s3.amazonaws.com", "bucket": bucket,
                         "made_public": False, "change_window": "approved",
                         "pipeline": "terraform-apply"},
            ),
        ]

        user_db = {username: self._make_user(
            username, "Platform Engineer", "Engineering", risk_score=0.05,
            is_privileged=True, access_level="admin",
        )}

        ground_truth = GroundTruth(
            alert_classifications={alert_id: AlertClassification.FALSE_POSITIVE},
            true_positive_ids=[],
            false_positive_ids=[alert_id],
            benign_tp_ids=[],
            expected_techniques={},
            expected_response_actions={alert_id: [ResponseActionType.NO_ACTION]},
            relevant_log_sources={alert_id: [LogSource.AUTH, LogSource.CLOUD_TRAIL]},
            relevant_indicators={alert_id: [travel_ip, username]},
        )

        return ScenarioConfig(
            scenario_id=f"cloud-compromise-fp-{self.seed}",
            task_id="cloud_compromise",
            seed=self.seed,
            description="Single cloud identity alert — FALSE POSITIVE. SSO+MFA admin on approved travel.",
            max_steps=self.MAX_STEPS,
            alerts=[alert],
            enrichment_db=enrichment_db,
            log_db=log_db,
            asset_db={},
            user_db=user_db,
            ground_truth=ground_truth,
        )
