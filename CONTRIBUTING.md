# Contributing to Agentic RAG

Thank you for your interest in contributing! This guide will help you get started.

## :wave: Welcome

We welcome contributions of all kinds:
- Bug reports and fixes
- New features and enhancements
- Documentation improvements
- Test coverage
- Performance optimizations

## :memo: Development Setup

### Prerequisites
- Python 3.11+
- Ollama installed and running
- Git

### Steps

```bash
# 1. Fork and clone the repository
git clone https://github.com/yourusername/agentic-rag.git
cd agentic-rag

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install pytest ruff black

# 4. Create a feature branch
git checkout -b feature/your-feature-name

# 5. Make your changes and test
pytest tests/ -v

# 6. Run linters
ruff check .
black --check .

# 7. Commit and push
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name

# 8. Open a Pull Request
```

## :triangular_ruler: Code Style

- **Formatter**: Black (line length 88)
- **Linter**: Ruff
- **Type Hints**: Required for all public functions
- **Docstrings**: Google style for all modules and public functions

```bash
# Auto-format
black .

# Auto-fix linting
ruff check --fix .
```

## :test_tube: Testing

All contributions must include tests:

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing
```

### Test Categories

| Category | Marker | Description |
|----------|--------|-------------|
| Unit | *(default)* | Individual function tests |
| Integration | *(default)* | End-to-end pipeline tests |
| Slow | `@pytest.mark.slow` | Tests requiring Ollama |

## :bug: Bug Reports

When filing a bug report, please include:
1. **Environment**: OS, Python version, Ollama version
2. **Steps to reproduce**: Minimal code to trigger the issue
3. **Expected behavior**: What you expected
4. **Actual behavior**: What actually happened
5. **Logs**: Relevant error messages or stack traces

## :bulb: Feature Requests

For feature requests, please open an issue with:
- Clear description of the use case
- How it benefits the project
- Possible implementation approach

## :label: Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new feature
fix: bug fix
docs: documentation only
style: code style changes
refactor: code refactoring
test: adding tests
chore: maintenance tasks
```

## :question: Questions?

Open a GitHub Discussion or reach out on Discord.

Thank you for contributing! :heart:
