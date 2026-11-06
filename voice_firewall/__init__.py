"""Voice Firewall — public API.

    from voice_firewall import Firewall, FirewallConfig, TaggedSegment, Decision, RiskLevel

Everything the integrator needs is re-exported here.  Internal modules
(``_pipeline``, ``_types``, ``guards.*``, etc.) are implementation detail.
"""
from voice_firewall._config import FirewallConfig
from voice_firewall._pipeline import Firewall
from voice_firewall._types import (
    ActionRequest,
    ActionVerdict,
    Decision,
    GuardResult,
    GuardVerdict,
    PipelineContext,
    RawSegment,
    RiskLevel,
    TaggedSegment,
)

__all__ = [
    # Main class
    "Firewall",
    # Config
    "FirewallConfig",
    # Result types
    "TaggedSegment",
    "Decision",
    "ActionRequest",
    "GuardResult",
    "RawSegment",
    "PipelineContext",
    # Enums
    "RiskLevel",
    "GuardVerdict",
    "ActionVerdict",
]

__version__ = "0.1.0"
