"""
PostMath Framework
==================

A dual-mode semantic engine that fuses linear NLP with non-linear, 
cascade-oriented operators.

© 2025 Jesús Manuel Soledad Terrazas. All rights reserved.
Licensed under the PostMath Public Research License v1.0
"""

__version__ = "1.0.0"
__author__ = "Jesús Manuel Soledad Terrazas"
__email__ = "jesussoledadt@gmail.com"
__license__ = "PostMath Public Research License v1.0"
__copyright__ = "© 2025 Jesús Manuel Soledad Terrazas"

from .core import (
    SemanticNode,
    SemanticRelation,
    DualModeSemantics,
    LicenseError,
    PostMathLicense
)

from .evaluator import SemanticEvaluator

from .translator import PracticalTranslator

__all__ = [
    # Core classes
    'SemanticNode',
    'SemanticRelation', 
    'DualModeSemantics',
    'SemanticEvaluator',
    'PracticalTranslator',
    
    # License
    'LicenseError',
    'PostMathLicense',
    
    # Metadata
    '__version__',
    '__author__',
    '__email__',
    '__license__',
    '__copyright__'
]