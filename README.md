# PostMath Framework

<div align="center">

**PostMath™** - Dual-mode semantic engine fusing linear NLP with non-linear, cascade-oriented operators

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/Marte-AI/Postmath-framework/releases)
[![License](https://img.shields.io/badge/license-PostMath%20Public%20Research-orange.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/downloads/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Marte-AI/Postmath-framework/blob/main/examples/01_quickstart.ipynb)

[Demo](#quick-start) • [Features](#key-features) • [Benchmarks](#performance) • [Documentation](#documentation) • [License](#license)

</div>

---

## What's New (2025-01-14)

- ✨ **v1.0.0** - First public release of PostMath Framework
- 🚀 Dual-mode semantic processing (linear + non-linear)
- ⚡ Cascade dynamics operator (⇝cascade)
- 🌀 Uncertainty mapping (Ψ∞^void)
- 📊 Built-in benchmarking against linear baselines

## Quick Start

### Installation

```bash
pip install -e .
```

### CLI Usage

```bash
postmath                # Run comprehensive demo with benchmarks
postmath-interactive    # Interactive semantic analysis
```

### Python API

```python
from postmath.translator import PracticalTranslator

# Initialize translator
translator = PracticalTranslator()

# Analyze text with dual-mode processing
result = translator.translate_text(
    "Consciousness emerges from complex neural interactions", 
    mode="dual"
)

# Access nonlinear insights
print(f"Uncertainty Level: {result['nonlinear_analysis']['uncertainty_level']:.3f}")
print(f"Cascade Potential: {result['nonlinear_analysis']['cascade_potential']:.3f}")
print(f"Emergence Likelihood: {result['nonlinear_analysis']['emergence_likelihood']:.3f}")
```

## Key Features

### 🔄 Dual-Mode Processing
Combines traditional linear NLP with PostMath's non-linear operators for deeper semantic understanding.

### ⚡ Cascade Dynamics (⇝cascade)
Simulates semantic cascades showing how concepts trigger chain reactions of meaning.

![Cascade Example](assets/cascade_example.png)
*Example cascade from "Creativity sparks innovation through imagination"*

### 🌀 Uncertainty Mapping (Ψ∞^void)
Maps uncertainty levels across concepts and identifies bridges between unknown territories.

![Uncertainty Map](assets/uncertainty_map.png)
*Uncertainty analysis of "Consciousness transcends physical reality through mysterious void"*

### 📊 Measurable Improvements
Built-in benchmarking shows consistent improvements over linear-only approaches.

## Performance

Results from built-in benchmark suite:

| Metric | Linear Baseline | PostMath Dual-Mode | Improvement |
|--------|----------------|-------------------|-------------|
| Understanding Score | 0.542 | 0.687 | **+26.8%** |
| Emotional Content | 0.493 | 0.651 | +32.0% |
| Technical Content | 0.612 | 0.698 | +14.1% |
| Philosophical Content | 0.421 | 0.743 | +76.5% |
| Scientific Content | 0.589 | 0.671 | +13.9% |

Key improvements in:
- ✅ Abstract concept handling
- ✅ Creative content detection
- ✅ Uncertainty calibration
- ✅ Emergence prediction

## Examples

### Domain-Specific Analysis

```python
# Technical domain
tech_result = translator.translate_text(
    "Algorithms process data to generate insights"
)
# High reality grounding, moderate cascade potential

# Philosophical domain  
phil_result = translator.translate_text(
    "Consciousness emerges from complex neural interactions"
)
# High uncertainty, high emergence likelihood

# Creative domain
creative_result = translator.translate_text(
    "Imagination sparks innovation through creative exploration"
)
# High creativity factor, high cascade potential
```

### Cascade Simulation

```python
# Simulate semantic cascade
cascade_results = translator.semantics.simulate_cascade(
    trigger_word="creativity",
    max_depth=5
)

for step in cascade_results:
    print(f"Depth {step['depth']}: {step['word']} "
          f"(activation: {step['activation']:.3f})")
```

### Uncertainty Bridge Detection

```python
# Find connections between uncertain concepts
bridges = translator.semantics.find_uncertainty_bridges(threshold=0.5)

for bridge in bridges[:3]:
    print(f"{bridge['source']} ↔ {bridge['target']}: "
          f"uncertainty bridge strength {bridge['uncertainty_bridge']:.3f}")
```

## Documentation

- 📚 [Full API Reference](docs/api.md)
- 🎯 [Tutorial Notebooks](examples/)
- 🧮 [Mathematical Foundation](docs/theory.md)
- 🔧 [Configuration Guide](docs/configuration.md)

## License

This project is licensed under the **PostMath Public Research License v1.0**.

### Allowed Uses
- ✅ Academic research
- ✅ Educational purposes  
- ✅ Non-commercial projects

### Commercial Use
For commercial licensing inquiries, please contact: jesussoledadt@gmail.com

See [LICENSE](LICENSE) and [License FAQ](docs/LICENSE_FAQ.md) for details.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Good First Issues
- [ ] Add cascade graph PNG export
- [ ] Export uncertainty heatmap visualization
- [ ] Add more domain vocabularies
- [ ] Improve cascade coherence metrics

## Citation

If you use PostMath in your research, please cite:

```bibtex
@software{postmath2025,
  author = {Terrazas, Jesús Manuel Soledad},
  title = {PostMath Framework: Dual-Mode Semantic Engine},
  year = {2025},
  version = {1.0.0},
  url = {https://github.com/Marte-AI/Postmath-framework}
}
```

## Contact

**Author**: Jesús Manuel Soledad Terrazas  
**Email**: jesussoledadt@gmail.com  
**Website**: [www.marteai.com/postmath](https://www.marteai.com/postmath)

---

<div align="center">

© 2025 Jesús Manuel Soledad Terrazas. All rights reserved.

</div>