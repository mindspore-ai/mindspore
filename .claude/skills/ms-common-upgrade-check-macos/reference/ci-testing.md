# CI Test Suite for MindSpore macOS

This reference provides instructions for running the CI test suite after successful MindSpore installation on macOS Apple Silicon.

## When to Run CI Tests

- After successful installation to verify comprehensive functionality
- When troubleshooting specific operator or feature issues
- Before submitting patches or contributions
- Optional for basic usage (basic verification in Step 6 is sufficient)

## Prerequisites

```bash
# Install test dependencies
pip install pytest torch torchvision torchaudio

# Set environment variable to avoid OpenMP conflicts
export KMP_DUPLICATE_LIB_OK=TRUE
```

## Running CI Tests

### Full Test Suite

```bash
# Navigate to test directory
cd /path/to/mindspore/tests/st

# Collect tests with markers: (cpu_macos OR platform_arm_cpu) AND level0
pytest --collect-only -m "(cpu_macos or platform_arm_cpu) and level0" -q 2>/dev/null | grep "::" > /tmp/ci_tests.txt

# Run collected tests
while IFS= read -r test; do
    echo "Running: $test"
    if pytest "$test" -v; then
        echo "PASSED: $test"
    else
        echo "FAILED: $test"
    fi
done < /tmp/ci_tests.txt
```

### Run Specific Test

```bash
# Run a single test file
pytest tests/st/ops/test_ops_add.py -v

# Run a specific test function
pytest tests/st/ops/test_ops_add.py::test_add_forward -v
```

## Expected Results

**Success:** All tests pass with output like:
```
PASSED: tests/st/ops/test_ops_add.py::test_add_forward
PASSED: tests/st/ops/test_ops_mul.py::test_mul_forward
...
```

**Failure:** If tests fail, analyze root cause:
1. Check error message for missing dependencies
2. Verify MindSpore installation: `python -c "import mindspore;print(mindspore.__version__)"`
3. Check if specific operators are not supported on CPU
4. Consult `troubleshooting.md` for known issues

## Common Test Issues

### Issue: Many Tests Skipped

**Symptom:**
```
SKIPPED [X] tests/st/...
```

**Cause:** Missing PyTorch dependency

**Fix:**
```bash
pip install torch torchvision torchaudio
```

### Issue: OpenMP Error

**Symptom:**
```
OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
```

**Fix:**
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### Issue: Test Collection Fails

**Symptom:**
```
ERROR: file or directory not found: tests/st/
```

**Fix:**
```bash
# Ensure you're in the MindSpore source directory
cd /path/to/mindspore
# Verify tests directory exists
ls tests/st/
```

## Test Markers

MindSpore uses pytest markers to categorize tests:

- `cpu_macos`: Tests specifically for macOS CPU
- `platform_arm_cpu`: Tests for ARM CPU architecture
- `level0`: Basic functionality tests (fast)
- `level1`: Extended functionality tests (slower)

## Selective Testing

Run only specific categories:

```bash
# Only macOS-specific tests
pytest -m "cpu_macos" tests/st/

# Only level0 (fast) tests
pytest -m "level0" tests/st/

# Combine markers
pytest -m "(cpu_macos or platform_arm_cpu) and level0" tests/st/
```
