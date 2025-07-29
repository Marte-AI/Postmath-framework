# Contributing to PostMath Framework

Thank you for your interest in contributing to PostMath! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our principles of respectful and constructive collaboration.

## How to Contribute

### Reporting Issues

1. **Check existing issues** first to avoid duplicates
2. Use the appropriate issue template:
   - 🐛 Bug Report
   - ✨ Feature Request
   - 📚 Documentation
3. Provide clear, detailed information:
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - System information (Python version, OS)

### Good First Issues

Look for issues labeled `good first issue`:

- **Add cascade graph PNG export** - Hook into existing cascade analysis
- **Export uncertainty heatmap** - Visualize uncertainty maps
- **Add domain vocabularies** - Expand semantic understanding
- **Improve cascade metrics** - Enhance coherence calculations

### Pull Requests

1. **Fork** the repository
2. Create a **feature branch** (`git checkout -b feature/amazing-feature`)
3. Make your changes:
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed
4. **Test** your changes:
   ```bash
   make test
   make lint
   ```
5. **Commit** with clear messages:
   ```bash
   git commit -m "Add cascade visualization export"
   ```
6. **Push** to your fork and submit a PR

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Postmath-framework.git
cd Postmath-framework

# Install in development mode
make install-dev

# Run tests
make test

# Run linting
make lint
```

## Code Style

- Follow PEP 8
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use descriptive variable names
- Add docstrings to all public functions/classes

### Example Code Style

```python
def process_semantic_cascade(
    self, 
    trigger_word: str, 
    max_depth: int = 5
) -> List[Dict[str, Any]]:
    """
    Simulate semantic cascade from trigger word.
    
    Args:
        trigger_word: Initial word to start cascade
        max_depth: Maximum cascade depth
        
    Returns:
        List of cascade steps with activation info
    """
    # Implementation here
```

## Testing

- Write tests for new features
- Maintain test coverage above 80%
- Use pytest for testing
- Place tests in `tests/` directory

### Test Example

```python
def test_cascade_simulation():
    translator = PracticalTranslator()
    result = translator.semantics.simulate_cascade("creativity", max_depth=3)
    
    assert len(result) > 0
    assert result[0]['word'] == 'creativity'
    assert all('activation' in step for step in result)
```

## Documentation

- Update docstrings for API changes
- Add examples for new features
- Update README if adding major features
- Consider adding tutorial notebooks

## License Compliance

- All contributions must comply with the PostMath Public Research License v1.0
- Do not add dependencies with incompatible licenses
- Maintain copyright notices

## Areas for Contribution

### Core Framework
- Enhance dual-mode processing algorithms
- Improve uncertainty calibration
- Add new semantic relationship types

### Visualization
- Create interactive cascade graphs
- Build uncertainty heatmap tools
- Design semantic network visualizers

### Performance
- Optimize cascade simulation
- Improve vocabulary loading
- Add caching mechanisms

### Documentation
- Write tutorial notebooks
- Create API examples
- Improve mathematical explanations

### Testing
- Add edge case tests
- Create benchmark suites
- Build integration tests

## Questions?

- Open a [Discussion](https://github.com/Marte-AI/Postmath-framework/discussions)
- Email: jesussoledadt@gmail.com (for licensing questions)

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- Academic papers (where applicable)

Thank you for helping make PostMath better! 🚀