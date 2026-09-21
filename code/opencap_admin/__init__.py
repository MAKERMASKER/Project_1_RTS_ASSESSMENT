"""Administrative OpenCap tooling: session download (+ later, kinematics repair).

Kept as a subpackage separate from the core analysis pipeline -- this is
occasional/interactive tooling (you run it once per new subject, or when
troubleshooting a session), not something the SLJ pipeline calls on every run.
"""
from opencap_admin.download import OpenCapDownloader

__all__ = ["OpenCapDownloader"]
