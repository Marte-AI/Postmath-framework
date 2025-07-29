"""
Basic tests for PostMath Framework
"""

import pytest
from postmath.translator import PracticalTranslator
from postmath.core import SemanticNode, DualModeSemantics, PostMathLicense, LicenseError


class TestLicense:
    """Test license validation"""
    
    def test_valid_license_uses(self):
        """Test that valid use cases pass"""
        PostMathLicense.validate_use('research')
        PostMathLicense.validate_use('education')
        PostMathLicense.validate_use('non-commercial')
    
    def test_invalid_license_use(self):
        """Test that commercial use raises error"""
        with pytest.raises(LicenseError):
            PostMathLicense.validate_use('commercial')
    
    def test_usage_logging(self):
        """Test that usage can be logged"""
        log = PostMathLicense.log_usage()
        assert 'timestamp' in log
        assert 'version' in log


class TestSemanticNode:
    """Test semantic node functionality"""
    
    def test_node_creation(self):
        """Test basic node creation"""
        node = SemanticNode('test', 'general', 'concept')
        assert node.word == 'test'
        assert node.domain == 'general'
        assert node.node_type == 'concept'
    
    def test_uncertainty_computation(self):
        """Test uncertainty level computation"""
        # High uncertainty word
        node1 = SemanticNode('consciousness', 'philosophical', 'concept')
        assert node1.uncertainty_level > 0
        
        # Low uncertainty word
        node2 = SemanticNode('table', 'general', 'concept')
        assert node2.uncertainty_level == 0
    
    def test_cascade_potential(self):
        """Test cascade potential computation"""
        node = SemanticNode('cause', 'general', 'relation')
        assert node.cascade_potential > 0


class TestDualModeSemantics:
    """Test dual-mode semantics processor"""
    
    def test_initialization(self):
        """Test processor initialization"""
        processor = DualModeSemantics()
        assert len(processor.domain_contexts) > 0
        assert 'technical' in processor.domain_contexts
    
    def test_add_node(self):
        """Test adding nodes"""
        processor = DualModeSemantics()
        node = processor.add_node('test', 'general', 'concept')
        
        assert 'test' in processor.nodes
        assert processor.nodes['test'] == node
    
    def test_domain_detection(self):
        """Test domain detection"""
        processor = DualModeSemantics()
        
        assert processor.detect_domain("algorithm process data") == 'technical'
        assert processor.detect_domain("consciousness and existence") == 'philosophical'
        assert processor.detect_domain("art beauty imagination") == 'creative'
    
    def test_process_linear(self):
        """Test linear processing"""
        processor = DualModeSemantics()
        result = processor.process_linear("test text processing")
        
        assert 'tokens' in result
        assert 'domain' in result
        assert 'semantic_density' in result
    
    def test_process_nonlinear(self):
        """Test nonlinear processing"""
        processor = DualModeSemantics()
        result = processor.process_nonlinear("consciousness emerges mysteriously")
        
        assert 'uncertainty_level' in result
        assert 'cascade_potential' in result
        assert 'creativity_factor' in result
        assert result['uncertainty_level'] > 0  # Should be high for "consciousness"


class TestPracticalTranslator:
    """Test the main translator interface"""
    
    def test_translator_initialization(self):
        """Test translator creation"""
        translator = PracticalTranslator()
        assert translator.semantics is not None
        assert translator.evaluator is not None
    
    def test_translate_text_linear(self):
        """Test linear mode translation"""
        translator = PracticalTranslator()
        result = translator.translate_text("test text", mode='linear')
        
        assert result['mode'] == 'linear'
        assert 'result' in result
        assert 'interpretation' in result
    
    def test_translate_text_nonlinear(self):
        """Test nonlinear mode translation"""
        translator = PracticalTranslator()
        result = translator.translate_text("consciousness emerges", mode='nonlinear')
        
        assert result['mode'] == 'nonlinear'
        assert 'result' in result
        assert 'interpretation' in result
    
    def test_translate_text_dual(self):
        """Test dual mode translation"""
        translator = PracticalTranslator()
        result = translator.translate_text("algorithms process data", mode='dual')
        
        assert result['mode'] == 'dual'
        assert 'linear_analysis' in result
        assert 'nonlinear_analysis' in result
        assert 'synthesis' in result
        assert 'practical_insights' in result
    
    def test_cascade_simulation(self):
        """Test cascade simulation"""
        translator = PracticalTranslator()
        translator.semantics.add_node('creativity', 'creative', 'concept')
        translator.semantics.add_node('innovation', 'creative', 'concept')
        translator.semantics.add_relation('creativity', 'innovation', 'causes', 0.8)
        
        cascade = translator.semantics.simulate_cascade('creativity', max_depth=3)
        assert len(cascade) > 0
        assert cascade[0]['word'] == 'creativity'
    
    def test_evaluate_translation_quality(self):
        """Test quality evaluation"""
        translator = PracticalTranslator()
        quality = translator.evaluate_translation_quality(
            "love creates connection",
            expected_concepts=['love', 'connection']
        )
        
        assert 'understanding_quality' in quality
        assert 'overall_score' in quality
        assert quality['overall_score'] >= 0 and quality['overall_score'] <= 1


class TestEvaluator:
    """Test the semantic evaluator"""
    
    def test_benchmark_against_baseline(self):
        """Test benchmarking functionality"""
        translator = PracticalTranslator()
        test_cases = [
            {
                'text': "algorithms process data",
                'expected_concepts': ['algorithms', 'data'],
                'category': 'technical'
            }
        ]
        
        results = translator.evaluator.benchmark_against_baseline(test_cases)
        
        assert 'dual_mode_scores' in results
        assert 'linear_only_scores' in results
        assert len(results['dual_mode_scores']) == 1


@pytest.mark.parametrize("text,expected_domain", [
    ("machine learning algorithms", "technical"),
    ("consciousness and awareness", "philosophical"),
    ("artistic creative expression", "creative"),
    ("quantum physics particles", "scientific")
])
def test_domain_detection_parametrized(text, expected_domain):
    """Test domain detection with multiple examples"""
    processor = DualModeSemantics()
    detected = processor.detect_domain(text)
    assert detected == expected_domain


@pytest.mark.parametrize("word,min_uncertainty", [
    ("consciousness", 0.1),
    ("infinity", 0.1),
    ("table", 0.0),
    ("chair", 0.0)
])
def test_uncertainty_levels(word, min_uncertainty):
    """Test uncertainty computation for different words"""
    node = SemanticNode(word, 'general', 'concept')
    assert node.uncertainty_level >= min_uncertainty