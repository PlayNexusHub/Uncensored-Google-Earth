# Contributing to PlayNexus Satellite Toolkit

Thank you for your interest in contributing to the PlayNexus Satellite Toolkit! This document provides guidelines for contributing to this educational project.

## 🎯 Project Mission

This toolkit is designed for educational use, making satellite imagery analysis accessible to students, researchers, and educators worldwide. All contributions should align with this educational mission.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+ (3.10+ recommended)
- Git
- Virtual environment (recommended)

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/satellite-toolkit.git
   cd satellite-toolkit
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   .\venv\Scripts\activate   # Windows
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```

4. **Run tests to verify setup**
   ```bash
   pytest
   ```

## 🏗️ Architecture Overview

The project follows a modular MVC (Model-View-Controller) architecture:

```
gui/
├── components/
│   ├── controllers/     # Business logic controllers
│   ├── models/          # Data models with change notifications
│   ├── views/           # UI views and interfaces
│   └── widgets/         # Custom widget components
└── utils/               # Reusable UI and animation utilities

scripts/                 # Core satellite processing functionality
tests/                   # Comprehensive test suite
```

## 📝 Code Style Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Include type hints for all function signatures
- 100 character line length maximum
- Use Black for code formatting

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
pre-commit install
```

This will automatically run:
- Black (code formatting)
- Flake8 (linting)
- mypy (type checking)
- pytest (tests)

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=playnexus_satellite_toolkit

# Run specific test file
pytest tests/test_specific_module.py

# Run type checking
mypy .

# Run linting
flake8 .
```

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Include both positive and negative test cases
- Mock external dependencies (API calls, file operations)
- Aim for >90% test coverage

## 🔄 Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   pytest
   mypy .
   flake8 .
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📋 Types of Contributions

### 🐛 Bug Reports
- Use the bug report template
- Include steps to reproduce
- Provide system information
- Include error messages and logs

### ✨ Feature Requests
- Use the feature request template
- Explain the educational value
- Provide use cases and examples
- Consider implementation complexity

### 📚 Documentation
- Improve existing documentation
- Add examples and tutorials
- Fix typos and formatting
- Translate documentation

### 🔧 Code Contributions
- Bug fixes
- New features
- Performance improvements
- Code refactoring
- Test improvements

## 🎓 Educational Focus

When contributing, consider:
- **Clarity**: Code should be easy to understand for students
- **Documentation**: Include clear explanations of concepts
- **Examples**: Provide practical, real-world examples
- **Error Handling**: User-friendly error messages
- **Performance**: Reasonable performance for educational use

## 📖 Documentation Standards

- Use clear, concise language
- Include code examples
- Explain remote sensing concepts when relevant
- Provide links to educational resources
- Update README.md for significant changes

## 🔒 Security Considerations

- Validate all user inputs
- Use secure file handling practices
- Don't hardcode API keys or sensitive data
- Follow security best practices for data processing

## 📄 License and Legal

- All contributions must be compatible with our educational use license
- By contributing, you agree to license your work under the same terms
- Respect data source licenses (Sentinel-2, Landsat)
- Don't include copyrighted material without permission

## 🤝 Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Provide constructive feedback
- Focus on educational value
- Collaborate openly and transparently

## 📞 Getting Help

- Open an issue for questions
- Join our discussions
- Check existing documentation
- Contact: playnexushq@gmail.com

## 🏷️ Commit Message Format

Use conventional commits:
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `test:` test additions/changes
- `refactor:` code refactoring
- `style:` formatting changes
- `chore:` maintenance tasks

Example: `feat: add NDVI calculation with cloud masking`

## 🎉 Recognition

Contributors will be recognized in:
- README.md acknowledgments
- Release notes
- Project documentation

Thank you for helping make satellite imagery analysis more accessible to everyone! 🛰️🌍
