"""Zero-Copy AI package."""
from .config import CONFIG
from .services import CompressionService, QueryService

__all__ = ["CONFIG", "CompressionService", "QueryService"]
