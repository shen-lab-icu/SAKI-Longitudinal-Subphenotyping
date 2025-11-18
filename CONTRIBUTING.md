# Contributing to SAKI-Phenotyping

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🤝 Ways to Contribute

- **Bug Reports**: Open an issue describing the bug, including steps to reproduce
- **Feature Requests**: Suggest new features or enhancements via issues
- **Code Contributions**: Submit pull requests for bug fixes or new features
- **Documentation**: Improve README, docstrings, or add examples
- **Testing**: Add unit tests or report test failures

## 🔧 Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/shen-lab-icu/SAKI-Longitudinal-Subphenotyping.git
   cd SAKI-Longitudinal-Subphenotyping
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv ../.venv
   source ../.venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8  # Development tools
   ```

4. **Run tests to ensure everything works**:
   ```bash
   PYTHONPATH=src pytest tests -v
   ```

## 📝 Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write docstrings for all public functions (Google style preferred)
- Keep functions focused and under 50 lines when possible
- Add unit tests for new functionality

**Formatting**:
```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/ --max-line-length=100
```

## 🧪 Testing

- All new features must include unit tests
- Tests should be placed in `tests/` directory
- Run full test suite before submitting PR:
  ```bash
  PYTHONPATH=src pytest tests -v --cov=sa_aki_pipeline
  ```

## 📤 Submitting Changes

1. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clear, concise commit messages
   - Keep commits focused on single changes
   - Add tests for new functionality

3. **Test your changes**:
   ```bash
   PYTHONPATH=src pytest tests -v
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**:
   - Provide a clear description of changes
   - Reference any related issues
   - Ensure CI tests pass

## 🔒 Data Privacy

**CRITICAL**: Never commit patient data to the repository!

- Review `.gitignore` before committing
- No `.csv`, `.parquet`, `.pkl` files with patient data
- No database credentials or API keys
- Test with synthetic/mock data only

## 📋 Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated (README, docstrings)
- [ ] No patient data included
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

## 🐛 Bug Reports

When reporting bugs, please include:

- Python version and OS
- Steps to reproduce the issue
- Expected vs. actual behavior
- Error messages (full traceback)
- Relevant code snippets (without patient data)

## 💡 Feature Requests

For feature requests, describe:

- Use case and motivation
- Proposed solution or API
- Any alternatives considered
- Potential impact on existing code

## ❓ Questions

For questions:
- Check existing documentation (README, REFACTOR_PROGRESS.md)
- Search existing issues
- Open a new issue with "Question:" prefix

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing!** 🙌
