---
inclusion: always
---

# Windows CMD Terminal Commands Only

## Terminal and Command Requirements

### REQUIRED: Windows CMD Only
- **ALWAYS use Windows CMD terminal commands**
- **NEVER use PowerShell commands** (no `Get-ChildItem`, `Remove-Item`, etc.)
- **NEVER use Bash/Linux commands** (no `ls`, `rm`, `grep`, etc.)
- **NEVER use Unix-style commands** (no `curl`, `wget`, `chmod`, etc.)

### CMD Command Examples
```cmd
REM Correct CMD commands to use:
dir                    REM List directory contents
cd /d C:\path          REM Change directory
copy file1.txt file2.txt  REM Copy files
move file1.txt newname.txt REM Move/rename files
del file.txt           REM Delete files
rmdir /s /q folder     REM Remove directory recursively
mkdir newfolder        REM Create directory
type file.txt          REM Display file contents
echo "text" > file.txt REM Write to file
findstr "pattern" *.txt REM Search in files
set VAR=value          REM Set environment variable
if exist file.txt echo Found  REM Check file existence
```

### FORBIDDEN Commands
```powershell
# NEVER use these PowerShell commands:
Get-ChildItem
Remove-Item
Copy-Item
Move-Item
New-Item
Get-Content
Set-Content
Test-Path
```

```bash
# NEVER use these Bash/Linux commands:
ls
rm
cp
mv
mkdir -p
cat
grep
find
chmod
chown
```

## Python Execution Requirements

### REQUIRED: Use "python -m" for Script Testing
- **ALWAYS use `python -m module_name`** when testing or running Python modules
- **NEVER use `python -c "code"`** for inline code execution
- **Use proper module execution** for testing and validation

### Correct Python Execution Examples
```cmd
REM Correct ways to test Python scripts:
python -m pytest tests/
python -m unittest test_module
python -m mypy src/
python -m black src/
python -m flake8 src/
python -m pip install package
python -m venv venv_name
python script.py
python -m package.module
```

### FORBIDDEN Python Execution
```cmd
REM NEVER use these patterns:
python -c "import sys; print(sys.version)"
python -c "print('hello')"
python -c "import os; os.system('command')"
```

### Alternative Approaches for Testing
Instead of `python -c`, create proper test scripts:

```cmd
REM Instead of: python -c "import module"
REM Create: test_import.py
python test_import.py

REM Instead of: python -c "print(os.getcwd())"
REM Create: show_cwd.py
python show_cwd.py
```

## Batch Script Requirements

### Use .bat Files for Automation
- **Create .bat files** for complex command sequences
- **Use CMD syntax** in batch files
- **Test with CMD terminal** before deployment

### Batch Script Example
```cmd
@echo off
REM Windows CMD batch script example
set PYTHON_PATH=python
set PROJECT_DIR=%CD%

echo Starting application tests...
%PYTHON_PATH% -m pytest tests/
if errorlevel 1 (
    echo Tests failed!
    exit /b 1
)

echo Running code quality checks...
%PYTHON_PATH% -m black --check src/
%PYTHON_PATH% -m flake8 src/

echo All checks passed!
```

## File Path Handling

### Windows Path Conventions
```cmd
REM Use Windows path separators
cd /d C:\Users\Username\Project
copy "C:\path with spaces\file.txt" "D:\destination\"

REM Use quotes for paths with spaces
dir "C:\Program Files\Python\"

REM Use environment variables
echo %USERPROFILE%
echo %PROGRAMFILES%
```

## Environment Variable Management

### CMD Environment Variables
```cmd
REM Set environment variables
set PYTHONPATH=%CD%\src
set ENVIRONMENT=development
set DEBUG=true

REM Use environment variables
echo Current path: %PATH%
python -m myapp --env=%ENVIRONMENT%
```

## Error Handling in Batch Scripts

### Proper Error Checking
```cmd
@echo off
python -m pytest tests/
if errorlevel 1 (
    echo ERROR: Tests failed
    pause
    exit /b 1
)

python -m mypy src/
if errorlevel 1 (
    echo ERROR: Type checking failed
    pause
    exit /b 1
)

echo All checks passed successfully!
```

## Integration with Development Tools

### IDE and Editor Integration
- **Configure IDEs** to use CMD as default terminal
- **Set Python interpreter** to use `python -m` for module execution
- **Configure build tools** to use CMD commands

### Git Integration
```cmd
REM Use git with CMD (git is available in CMD)
git status
git add .
git commit -m "Update with CMD commands"
git push origin main
```

## Testing and Validation Commands

### Module Testing
```cmd
REM Test Python modules properly
python -m pytest tests/test_auth.py
python -m pytest tests/test_api.py -v
python -m unittest discover tests/
python -m doctest src/module.py
```

### Code Quality
```cmd
REM Code quality checks
python -m black src/ tests/
python -m isort src/ tests/
python -m flake8 src/ tests/
python -m mypy src/
python -m bandit -r src/
```

### Package Management
```cmd
REM Package installation and management
python -m pip install -r requirements.txt
python -m pip install --upgrade pip
python -m pip list
python -m pip show package_name
python -m pip freeze > requirements.txt
```

## Documentation and Help

### Getting Help
```cmd
REM Get help for CMD commands
help dir
help copy
help set

REM Get help for Python modules
python -m pytest --help
python -m pip --help
python -m mypy --help
```

This steering ensures all terminal operations use Windows CMD syntax and all Python module testing uses the proper `python -m` pattern instead of inline code execution.
