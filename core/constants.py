# (constants)
from pathlib import Path
from typing import Any, Optional

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
CONFIDENCE_THRESHOLD = 0.55
SESSION_BASE_DIR = Path(__file__).parent.parent.parent / "sessions"  # ml_project/sessions
IMAGE_SIZE = 300
AGGREGATE_OUTPUT_DIR = "microservise/photo_aggregate"
# Buffer size for smoothing Model-1 (sliding window)
MODEL1_BUFFER_SIZE = 1

_model1_cache: Optional[Any] = None
_model2_cache: Optional[Any] = None
_models_initialized = False
_current_session_dir: Optional[Path] = None
