# Contributing to WhoLLM

Thank you for your interest in contributing to WhoLLM!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/WhoLLM.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest tests/`
6. Commit your changes
7. Push to your fork and submit a Pull Request

## Development Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements_dev.txt

# Install pre-commit hooks
pre-commit install
```

## Code Style

- We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Run `ruff check .` to lint
- Run `ruff format .` to format
- All code must pass CI checks before merging

## Testing

- Write tests for new features
- Ensure all tests pass: `pytest tests/ -v`
- Aim for good coverage of edge cases

## Pull Request Guidelines

1. **One feature per PR** - Keep PRs focused and reviewable
2. **Update documentation** - If you change behavior, update the docs
3. **Add tests** - New features should include tests
4. **Follow existing patterns** - Match the code style of the project
5. **Write clear commit messages** - Explain what and why

## Reporting Issues

- Check existing issues before creating a new one
- Include Home Assistant version, WhoLLM version, and relevant logs
- Provide steps to reproduce the issue

## Areas for Contribution

- **Prompt engineering** - Improve LLM prompts for better accuracy
- **New providers** - Add support for OpenAI, Anthropic, etc.
- **Testing** - Improve test coverage
- **Documentation** - Examples, troubleshooting guides
- **Translations** - Help translate strings

## Code of Conduct

Be respectful and constructive. We're all here to make smart homes smarter.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
