from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class VerifierType(str, Enum):
    DATABASE_STATE = "database_state"
    RESPONSE_CHECKER = "response_check"
    TOOL_EXECUTION = "tool_execution"


@dataclass
class VerifierConfig:
    """Configuration for a verifier"""

    verifier_type: str
    validation_config: Dict[str, Any]
    name: Optional[str] = None
    description: Optional[str] = None
    gym_name: Optional[str] = None  # Which gym's database to query (for multi-gym)
