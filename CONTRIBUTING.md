# Contributing to Variational QEC Decoder

Thank you for your interest in contributing to this project! This document outlines the rules and guidelines for participation, with a specific focus on how we handle the MIT License.

## Licensing Rules (MIT)

This project is licensed under the **MIT License**. By contributing, you agree that your contributions will be licensed under the same terms.

### What the MIT License Means for You
- **Permission**: You can use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.
- **Requirement**: You must include the original copyright notice and the license text in any substantial portion of the Software or derivative works.
- **No Warranty**: The software is provided "as is", without warranty of any kind.

### Contribution Rules
1. **Attribution**: If you add significant new logic or entire files, you may add your name to the `LICENSE` file or include a copyright line at the top of the new files.
2. **Third-Party Code**: Do not include code that you do not own or that is under a restrictive license (like GPL) without explicit approval. All dependencies must be compatible with the MIT License.

## How to Contribute

1. **Bug Reports**: Open an issue describing the bug, including a traceback if possible.
2. **Feature Requests**: Open an issue to discuss the proposed feature before implementation.
3. **Pull Requests**:
   - Create a new branch for each feature or bug fix.
   - Ensure all unit tests pass (`python -m pytest tests/`).
   - Follow the existing code style (clean, documented, and typed).
   - Link the PR to any relevant issues.

## Code Style

- Use **type hints** for all function signatures.
- Provide **numpy-style docstrings** for every public function and class.
- Keep components modular and avoid hardcoding parameters.
- Run `black` or `flake8` if available to maintain formatting.

## Code of Conduct

Please be respectful and professional in all interactions. We aim to foster an inclusive and welcoming research environment.
