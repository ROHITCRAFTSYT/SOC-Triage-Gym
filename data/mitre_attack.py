"""
MITRE ATT&CK Technique Database
================================
Subset of MITRE ATT&CK v14.0 techniques relevant to SOC-Triage-Gym scenarios.
Covers Initial Access, Execution, Credential Access, Lateral Movement,
Collection, Command & Control, and Exfiltration tactics.
"""


TECHNIQUES: dict[str, dict[str, str]] = {
    # Initial Access
    "T1566": {
        "name": "Phishing",
        "tactic": "initial-access",
        "description": "Adversaries may send phishing messages to gain access to victim systems.",
    },
    "T1566.001": {
        "name": "Spearphishing Attachment",
        "tactic": "initial-access",
        "description": "Adversaries may send spearphishing emails with a malicious attachment.",
        "parent": "T1566",
    },
    "T1566.002": {
        "name": "Spearphishing Link",
        "tactic": "initial-access",
        "description": "Adversaries may send spearphishing emails with a malicious link.",
        "parent": "T1566",
    },
    "T1190": {
        "name": "Exploit Public-Facing Application",
        "tactic": "initial-access",
        "description": "Adversaries may attempt to exploit a weakness in an Internet-facing host.",
    },
    "T1078": {
        "name": "Valid Accounts",
        "tactic": "defense-evasion",
        "description": "Adversaries may obtain and abuse credentials of existing accounts.",
    },
    "T1078.002": {
        "name": "Domain Accounts",
        "tactic": "defense-evasion",
        "description": "Adversaries may obtain and abuse credentials of a domain account.",
        "parent": "T1078",
    },

    # Execution
    "T1059": {
        "name": "Command and Scripting Interpreter",
        "tactic": "execution",
        "description": "Adversaries may abuse command and script interpreters to execute commands.",
    },
    "T1059.001": {
        "name": "PowerShell",
        "tactic": "execution",
        "description": "Adversaries may abuse PowerShell commands and scripts for execution.",
        "parent": "T1059",
    },
    "T1059.003": {
        "name": "Windows Command Shell",
        "tactic": "execution",
        "description": "Adversaries may abuse the Windows command shell for execution.",
        "parent": "T1059",
    },
    "T1204": {
        "name": "User Execution",
        "tactic": "execution",
        "description": "An adversary may rely upon specific actions by a user in order to gain execution.",
    },
    "T1204.002": {
        "name": "Malicious File",
        "tactic": "execution",
        "description": "An adversary may rely upon a user opening a malicious file in order to gain execution.",
        "parent": "T1204",
    },

    # Persistence
    "T1547": {
        "name": "Boot or Logon Autostart Execution",
        "tactic": "persistence",
        "description": "Adversaries may configure system settings to automatically execute a program during system boot or logon.",
    },
    "T1547.001": {
        "name": "Registry Run Keys / Startup Folder",
        "tactic": "persistence",
        "description": "Adversaries may achieve persistence by adding a program to a commonly used registry run key.",
        "parent": "T1547",
    },
    "T1053": {
        "name": "Scheduled Task/Job",
        "tactic": "persistence",
        "description": "Adversaries may abuse task scheduling functionality to facilitate initial or recurring execution of malicious code.",
    },
    "T1053.005": {
        "name": "Scheduled Task",
        "tactic": "persistence",
        "description": "Adversaries may abuse the Windows Task Scheduler to perform task scheduling.",
        "parent": "T1053",
    },

    # Privilege Escalation
    "T1055": {
        "name": "Process Injection",
        "tactic": "privilege-escalation",
        "description": "Adversaries may inject code into processes to evade process-based defenses.",
    },

    # Defense Evasion
    "T1027": {
        "name": "Obfuscated Files or Information",
        "tactic": "defense-evasion",
        "description": "Adversaries may attempt to make an executable or file difficult to discover or analyze.",
    },
    "T1027.010": {
        "name": "Command Obfuscation",
        "tactic": "defense-evasion",
        "description": "Adversaries may obfuscate content during command execution to impede detection.",
        "parent": "T1027",
    },

    # Credential Access
    "T1110": {
        "name": "Brute Force",
        "tactic": "credential-access",
        "description": "Adversaries may use brute force techniques to gain access to accounts.",
    },
    "T1110.001": {
        "name": "Password Guessing",
        "tactic": "credential-access",
        "description": "Adversaries may use password guessing to access accounts.",
        "parent": "T1110",
    },
    "T1110.003": {
        "name": "Password Spraying",
        "tactic": "credential-access",
        "description": "Adversaries may use a single or small list of commonly used passwords against many accounts.",
        "parent": "T1110",
    },
    "T1003": {
        "name": "OS Credential Dumping",
        "tactic": "credential-access",
        "description": "Adversaries may attempt to dump credentials to obtain account login information.",
    },
    "T1003.001": {
        "name": "LSASS Memory",
        "tactic": "credential-access",
        "description": "Adversaries may attempt to access credential material stored in the process memory of LSASS.",
        "parent": "T1003",
    },
    "T1528": {
        "name": "Steal Application Access Token",
        "tactic": "credential-access",
        "description": "Adversaries can steal application access tokens as a means of acquiring credentials.",
    },

    # Discovery
    "T1046": {
        "name": "Network Service Discovery",
        "tactic": "discovery",
        "description": "Adversaries may attempt to get a listing of services running on remote hosts.",
    },
    "T1087": {
        "name": "Account Discovery",
        "tactic": "discovery",
        "description": "Adversaries may attempt to get a listing of accounts on a system or within an environment.",
    },

    # Lateral Movement
    "T1021": {
        "name": "Remote Services",
        "tactic": "lateral-movement",
        "description": "Adversaries may use valid accounts to log into a service specifically designed to accept remote connections.",
    },
    "T1021.001": {
        "name": "Remote Desktop Protocol",
        "tactic": "lateral-movement",
        "description": "Adversaries may use Valid Accounts to log into a computer using the Remote Desktop Protocol.",
        "parent": "T1021",
    },
    "T1021.002": {
        "name": "SMB/Windows Admin Shares",
        "tactic": "lateral-movement",
        "description": "Adversaries may use Valid Accounts to interact with a remote network share using Server Message Block.",
        "parent": "T1021",
    },
    "T1570": {
        "name": "Lateral Tool Transfer",
        "tactic": "lateral-movement",
        "description": "Adversaries may transfer tools or other files between systems in a compromised environment.",
    },

    # Collection
    "T1074": {
        "name": "Data Staged",
        "tactic": "collection",
        "description": "Adversaries may stage collected data in a central location or directory prior to Exfiltration.",
    },
    "T1074.001": {
        "name": "Local Data Staging",
        "tactic": "collection",
        "description": "Adversaries may stage collected data in a central location or directory on the local system.",
        "parent": "T1074",
    },
    "T1560": {
        "name": "Archive Collected Data",
        "tactic": "collection",
        "description": "Adversaries may compress and/or encrypt data that is collected prior to exfiltration.",
    },
    "T1560.001": {
        "name": "Archive via Utility",
        "tactic": "collection",
        "description": "Adversaries may use utilities to compress and/or encrypt collected data prior to exfiltration.",
        "parent": "T1560",
    },

    # Command and Control
    "T1071": {
        "name": "Application Layer Protocol",
        "tactic": "command-and-control",
        "description": "Adversaries may communicate using OSI application layer protocols to avoid detection.",
    },
    "T1071.001": {
        "name": "Web Protocols",
        "tactic": "command-and-control",
        "description": "Adversaries may communicate using application layer protocols associated with web traffic.",
        "parent": "T1071",
    },
    "T1105": {
        "name": "Ingress Tool Transfer",
        "tactic": "command-and-control",
        "description": "Adversaries may transfer tools or other files from an external system into a compromised environment.",
    },
    "T1132": {
        "name": "Data Encoding",
        "tactic": "command-and-control",
        "description": "Adversaries may encode data to make the content of command and control traffic more difficult to detect.",
    },
    "T1132.001": {
        "name": "Standard Encoding",
        "tactic": "command-and-control",
        "description": "Adversaries may encode data with a standard data encoding system to make the content undetectable.",
        "parent": "T1132",
    },

    # Exfiltration
    "T1041": {
        "name": "Exfiltration Over C2 Channel",
        "tactic": "exfiltration",
        "description": "Adversaries may steal data by exfiltrating it over an existing command and control channel.",
    },
    "T1048": {
        "name": "Exfiltration Over Alternative Protocol",
        "tactic": "exfiltration",
        "description": "Adversaries may steal data by exfiltrating it over a different protocol than that used for C2.",
    },
    "T1567": {
        "name": "Exfiltration Over Web Service",
        "tactic": "exfiltration",
        "description": "Adversaries may use an existing, legitimate external Web service to exfiltrate data.",
    },

    # Impact
    "T1486": {
        "name": "Data Encrypted for Impact",
        "tactic": "impact",
        "description": "Adversaries may encrypt data on target systems to interrupt availability.",
    },
    "T1490": {
        "name": "Inhibit System Recovery",
        "tactic": "impact",
        "description": "Adversaries may delete or remove built-in data and turn off services designed to aid in the recovery.",
    },
}

# Tactic ordering for kill chain display
TACTIC_ORDER = [
    "initial-access",
    "execution",
    "persistence",
    "privilege-escalation",
    "defense-evasion",
    "credential-access",
    "discovery",
    "lateral-movement",
    "collection",
    "command-and-control",
    "exfiltration",
    "impact",
]


def get_technique(technique_id: str) -> dict[str, str] | None:
    """Return technique metadata by ID, or None if not found."""
    return TECHNIQUES.get(technique_id)


def get_techniques_for_tactic(tactic: str) -> list[str]:
    """Return all technique IDs for a given tactic."""
    return [tid for tid, data in TECHNIQUES.items() if data.get("tactic") == tactic]


def is_valid_technique(technique_id: str) -> bool:
    """Return True if the technique ID exists in the database."""
    return technique_id in TECHNIQUES


def get_technique_name(technique_id: str) -> str:
    """Return technique name or the raw ID if not found."""
    tech = TECHNIQUES.get(technique_id)
    return tech["name"] if tech else technique_id
