"""
Shared Constants for SentinelIQ.
Centralized resource-department mappings and other shared configuration.
"""


# Resource to owning department mapping
# None means it's a shared resource with no department restriction
RESOURCE_DEPARTMENT_MAP = {
    "HRIS": "HR",
    "GL_System": "Finance",
    "BI_Tool": "Finance",
    "Admin_Console": "IT",
    "SIEM": "Security",
    "PROD_DB": "Engineering",
    "Data_Lake": "Engineering",
    "File_Share": None,  # Shared resource
    "Email_Archive": None,  # Shared resource
    "VPN": None,  # Shared resource
    "Customer_Vault": None,  # Shared resource
}

# Shared resources (no department restriction)
SHARED_RESOURCES = {
    resource for resource, dept in RESOURCE_DEPARTMENT_MAP.items() if dept is None
}

# High-sensitivity systems
HIGH_SENSITIVITY_SYSTEMS = {
    "PROD_DB", "ADMIN_SYS", "SIEM", "AWS_IAM", "GCP"
}
