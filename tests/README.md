# Test Suite Documentation

Comprehensive testing for ReadyTensor RAG System - Module 3 Requirements

## Test Structure

```
tests/
├── __init__.py
├── conftest.py           # Shared fixtures and configuration
├── unit/                 # Unit tests for individual components
│   ├── test_data_collector.py
│   ├── test_vector_store.py
│   └── test_rag_pipeline.py
├── integration/          # Integration tests
│   └── test_rag_integration.py
└── e2e/                  # End-to-end tests
    └── test_complete_workflow.py
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# End-to-end tests only
pytest -m e2e

# Exclude slow tests
pytest -m "not slow"
```

### Run With Coverage Report

```bash
# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term-missing

# View HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Run Specific Test Files

```bash
# Single file
pytest tests/unit/test_data_collector.py

# Multiple files
pytest tests/unit/ tests/integration/
```

### Verbose Output

```bash
pytest -v
pytest -vv  # Very verbose
```

## Test Coverage Requirements

**Module 3 Requirement**: Minimum 70% code coverage

Current coverage targets:
- **src/data_collector.py**: 80%+
- **src/vector_store.py**: 75%+
- **src/rag_pipeline_gemini.py**: 70%+
- **Overall**: 70%+

## Test Markers

Tests are organized with pytest markers:

- `@pytest.mark.unit` - Unit tests for individual components
- `@pytest.mark.integration` - Integration tests between components
- `@pytest.mark.e2e` - End-to-end workflow tests
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.api` - Tests requiring external API calls

## Continuous Integration

Tests run automatically on:
- Pull requests
- Commits to main branch
- Manual workflow dispatch

CI Configuration: `.github/workflows/test.yml`

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test

```python
import pytest
from src.module import MyClass

@pytest.mark.unit
class TestMyClass:
    @pytest.fixture
    def instance(self):
        return MyClass()

    def test_my_function(self, instance):
        result = instance.my_function()
        assert result == expected_value
```

### Using Fixtures

See `conftest.py` for available shared fixtures:
- `test_data_dir` - Temporary test data directory
- `sample_articles` - Sample Wikipedia articles
- `mock_google_api_key` - Mocked API key
- etc.

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're running from project root
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **API key errors**: Tests use mocked APIs, no real keys needed
4. **Coverage too low**: Add more tests or exclude non-critical code

### Debug Mode

```bash
# Run with detailed output
pytest -vv --tb=long

# Stop on first failure
pytest -x

# Run specific test
pytest tests/unit/test_data_collector.py::TestWikipediaDataCollector::test_init
```

## Performance

### Test Execution Time

- Unit tests: ~10 seconds
- Integration tests: ~30 seconds
- End-to-end tests: ~60 seconds
- Full suite: ~90 seconds

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

## Quality Gates

Tests must pass these criteria:
- ✅ All tests pass
- ✅ Coverage >= 70%
- ✅ No critical warnings
- ✅ Performance within limits

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [ReadyTensor Module 3 Requirements](https://app.readytensor.ai/publications/aaidc-module-3-project-productionize-your-agentic-system-DSYotKAAvcxy)
