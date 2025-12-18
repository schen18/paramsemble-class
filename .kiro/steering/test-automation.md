---
inclusion: always
---

# Test Automation and Command Optimization

## Objective
Minimize commands that require manual review and avoid complex chained commands, especially when creating and reviewing tests.

## Guidelines

### Test Execution
- **Use simple pytest commands**: Always use `python -m pytest <path> -v` instead of chained commands
- **Avoid python -c commands**: Never use `python -c "import ..."` for testing imports - create proper test files instead
- **Single command execution**: Execute one command at a time rather than chaining with `&&` or `;`
- **Avoid complex shell operations**: Don't chain multiple operations in a single command line

### Test Creation Best Practices
- **Create comprehensive test files**: Write complete test suites that can be run independently
- **Use descriptive test names**: Make test purposes clear from the function names
- **Include setup and teardown**: Ensure tests clean up after themselves
- **Test edge cases**: Include both positive and negative test cases
- **Mock external dependencies**: Use mocks for external services or complex dependencies

### Command Structure
- **Preferred**: `python -m pytest src/data/test_processor.py -v`
- **Avoid**: `conda activate torch && python -c "from src.data import DataProcessor" && python -m pytest src/data/ -v`
- **Preferred**: `python -m pytest src/data/ --tb=short`
- **Avoid**: Complex shell scripting or multiple command chains

### Error Handling
- **Run tests incrementally**: Test individual modules before running full test suites
- **Use appropriate pytest flags**: 
  - `-v` for verbose output
  - `--tb=short` for concise error reporting
  - `-x` to stop on first failure when debugging
- **Handle import errors gracefully**: Include try/catch blocks for optional dependencies

### File Organization
- **Separate test files**: Create individual test files for each module (e.g., `test_processor.py`, `test_preprocessing.py`)
- **Clear test structure**: Organize tests into logical classes and methods
- **Consistent naming**: Use `test_` prefix for all test functions and files

### Environment Management
- **Assume environment is activated**: Don't repeatedly activate conda environments in commands
- **Use simple activation**: When environment activation is needed, do it as a separate step
- **Document dependencies**: Clearly state required packages in test files or documentation

## Examples

### Good Command Patterns
```bash
# Simple test execution
python -m pytest src/data/test_processor.py -v

# Test specific function
python -m pytest src/data/test_processor.py::TestDataProcessor::test_load_csv_data -v

# Run all tests in directory
python -m pytest src/data/ --tb=short
```

### Avoid These Patterns
```bash
# Don't chain commands
conda activate torch && python -c "import src.data" && python -m pytest src/data/ -v

# Don't use complex shell operations
python -c "from src.data.preprocessing import DataPreprocessor; print('Import successful')"

# Don't chain multiple test operations
python -m pytest src/data/test_processor.py -v && python -m pytest src/data/test_preprocessing.py -v
```

### Test File Structure
```python
# Good test file structure
import unittest
import tempfile
from pathlib import Path

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_specific_functionality(self):
        """Test with clear, descriptive name."""
        # Test implementation
        pass
```

## Benefits
- **Faster execution**: Single commands execute more quickly
- **Easier debugging**: Clear, simple commands are easier to troubleshoot
- **Better reliability**: Fewer points of failure in command execution
- **Improved readability**: Code and commands are easier to understand and maintain
- **Reduced manual intervention**: Tests run automatically without requiring user input
