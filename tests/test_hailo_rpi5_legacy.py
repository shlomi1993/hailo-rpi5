"""
Legacy test file - Tests have been migrated to pytest format.

This file is kept for backward compatibility but the actual tests
are now in the unit/ and integration/ directories using pytest.

To run the new tests:
    pytest tests/unit/           # Unit tests
    pytest tests/integration/    # Integration tests (requires hardware)
    pytest tests/               # All tests
"""

# This file is deprecated - use pytest tests instead
import warnings

warnings.warn(
    "test_hailo_rpi5.py is deprecated. Use 'pytest tests/' to run the new test suite.",
    DeprecationWarning,
    stacklevel=2
)
