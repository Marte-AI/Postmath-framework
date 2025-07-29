#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# © 2025 Jesús Manuel Soledad Terrazas. All rights reserved.
# Licensed under the PostMath Public Research License v1.0
#
# This file implements the practical translation engine of PostMath™.
# For non-commercial research and educational use only.
#
"""
PostMath Framework: Practical Translator
========================================

Streamlined translator focusing on measurable improvements with PostMath operators.
"""

__author__ = "Jesús Manuel Soledad Terrazas"
__copyright__ = "© 2025 Jesús Manuel Soledad Terrazas"
__license__ = "PostMath Public Research License v1.0"
__version__ = "1.0.0"

import re
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

from .core import DualModeSemantics, SemanticNode
from .evaluator import SemanticEvaluator


class PracticalTranslator:
    """Streamlined translator focusing on measurable improvements with PostMath operators"""
    
    def __init__(self):
        self.semantics = DualModeSemantics()
        self.evaluator = SemanticEvaluator(self.semantics)
        self._load_common_vocabulary()
        
    def _load_common_vocabulary(self):
        """Load common vocabulary with relationships"""
        # Core concepts
        vocabulary = {
            # Abstract concepts (high uncertainty, high creativity potential)
            'consciousness': ('philosophical', 'concept'),
            'love': ('emotional', 'concept'),
            'creativity': ('creative', 'concept'),
            'beauty': ('creative', 'concept'),
            'meaning': ('philosophical', 'concept'),
            'purpose': ('philosophical', 'concept'),
            
            # Concrete concepts (low uncertainty, high grounding)
            'data': ('technical', 'concept'),
            'algorithm': ('technical', 'concept'),
            'system': ('technical', 'concept'),
            'process': ('technical', 'concept'),
            'network': ('technical', 'concept'),
            
            # Process words (high cascade potential)
            'create': ('general', 'relation'),
            'generate': ('general', 'relation'),
            'cause': ('general', 'relation'),
            'lead': ('general', 'relation'),
            'result': ('general', 'relation'),
            'flow': ('general', 'relation'),
            'connect': ('general', 'relation'),
            'transform': ('general', 'relation'),
            
            # Scientific concepts
            'energy': ('scientific', 'concept'),
            'matter': ('scientific', 'concept'),
            'particle': ('scientific', 'concept'),
            'wave': ('scientific', 'concept'),
            'field': ('scientific', 'concept'),
        }
        
        # Add vocabulary to semantic processor
        for word, (domain, node_type) in vocabulary.items():
            self.semantics.add_node(word, domain, node_type)
        
        # Add common relationships
        relationships = [
            ('create', 'generate', 'similar', 0.8),
            ('cause', 'lead', 'similar', 0.9),
            ('data', 'information', 'similar', 0.9),
            ('algorithm', 'process', 'part_of', 0.7),
            ('consciousness', 'mind', 'similar', 0.8),
            ('love', 'emotion', 'part_of', 0.6),
            ('energy', 'matter', 'related', 0.8),
            ('particle', 'wave', 'similar', 0.7),
            ('beauty', 'art', 'related', 0.6),
            ('meaning', 'purpose', 'related', 0.7),
        ]
        
        for source, target, rel_type, strength in relationships:
            self.semantics.add_relation(source, target, rel_type, strength)
    
    def translate_text(self, text: str, mode: str = 'dual') -> Dict[str, Any]:
        """Main translation interface with mode selection"""
        if mode == 'linear':
            return {
                'mode': 'linear',
                'result': self.semantics.process_linear(text),
                'interpretation': self._interpret_linear(text)
            }
        elif mode == 'nonlinear':
            return {
                'mode': 'nonlinear', 
                'result': self.semantics.process_nonlinear(text),
                'interpretation': self._interpret_nonlinear(text)
            }
        else:  # dual mode
            linear_result = self.semantics.process_linear(text)
            nonlinear_result = self.semantics.process_nonlinear(text)
            
            return {
                'mode': 'dual',
                'linear_analysis': linear_result,
                'nonlinear_analysis': nonlinear_result,
                'synthesis': self._synthesize_understanding(linear_result, nonlinear_result),
                'cascade_potential': self._analyze_cascade_potential(text),
                'uncertainty_map': self._map_uncertainties(text),
                'practical_insights': self._extract_practical_insights(linear_result, nonlinear_result)
            }
    
    def _interpret_linear(self, text: str) -> Dict[str, Any]:
        """Interpret linear analysis results"""
        linear_result = self.semantics.process_linear(text)
        
        return {
            'summary': f"Text contains {linear_result['unique_tokens']} unique concepts in {linear_result['domain']} domain",
            'complexity_level': 'high' if linear_result['complexity'] > 1.5 else 'medium' if linear_result['complexity'] > 1.2 else 'low',
            'semantic_richness': 'rich' if linear_result['semantic_density'] > 0.7 else 'moderate' if linear_result['semantic_density'] > 0.4 else 'sparse',
            'domain_confidence': linear_result['domain']
        }
    
    def _interpret_nonlinear(self, text: str) -> Dict[str, Any]:
        """Interpret nonlinear analysis results"""
        nonlinear_result = self.semantics.process_nonlinear(text)
        
        interpretation = {
            'uncertainty_level': self._categorize_level(nonlinear_result['uncertainty_level']),
            'creativity_potential': self._categorize_level(nonlinear_result['creativity_factor']),
            'cascade_readiness': self._categorize_level(nonlinear_result['cascade_potential']),
            'reality_grounding': self._categorize_level(nonlinear_result['reality_grounding']),
            'emergence_likelihood': self._categorize_level(nonlinear_result['emergence_likelihood'])
        }
        
        # Add insights
        if nonlinear_result['uncertainty_level'] > 0.7:
            interpretation['insight'] = "High uncertainty - text touches unknown/transcendent concepts"
        elif nonlinear_result['creativity_factor'] > 0.6:
            interpretation['insight'] = "High creativity potential - text enables novel combinations"
        elif nonlinear_result['cascade_potential'] > 0.6:
            interpretation['insight'] = "High cascade potential - text likely to trigger chain reactions"
        elif nonlinear_result['emergence_likelihood'] > 0.5:
            interpretation['insight'] = "High emergence likelihood - new meanings may arise from combinations"
        else:
            interpretation['insight'] = "Stable semantic content with predictable interpretations"
        
        return interpretation
    
    def _categorize_level(self, value: float) -> str:
        """Categorize a 0-1 value into descriptive levels"""
        if value > 0.8:
            return 'very_high'
        elif value > 0.6:
            return 'high'
        elif value > 0.4:
            return 'medium'
        elif value > 0.2:
            return 'low'
        else:
            return 'very_low'
    
    def _synthesize_understanding(self, linear: Dict, nonlinear: Dict) -> Dict[str, Any]:
        """Synthesize linear and nonlinear understanding"""
        return {
            'overall_complexity': (linear['complexity'] + nonlinear['cascade_potential']) / 2,
            'interpretation_confidence': linear['semantic_density'] * nonlinear['reality_grounding'],
            'novelty_potential': nonlinear['creativity_factor'] * nonlinear['uncertainty_level'],
            'practical_utility': linear['semantic_density'] * (1 - nonlinear['uncertainty_level']),
            'exploration_value': nonlinear['uncertainty_level'] * nonlinear['emergence_likelihood'],
            'domain_coherence': linear['domain']
        }
    
    def _analyze_cascade_potential(self, text: str) -> Dict[str, Any]:
        """Analyze cascade potential for key concepts in text"""
        words = re.findall(r'\w+', text.lower())
        cascade_analysis = {}
        
        for word in set(words):
            if word in self.semantics.nodes:
                cascade_result = self.semantics.simulate_cascade(word, max_depth=3)
                if cascade_result:
                    cascade_analysis[word] = {
                        'cascade_length': len(cascade_result),
                        'max_depth': max(step['depth'] for step in cascade_result),
                        'avg_uncertainty': sum(step['uncertainty'] for step in cascade_result) / len(cascade_result),
                        'cascade_path': [step['word'] for step in cascade_result[:5]]  # First 5 steps
                    }
        
        return cascade_analysis
    
    def _map_uncertainties(self, text: str) -> Dict[str, Any]:
        """Map uncertainty levels across the text"""
        words = re.findall(r'\w+', text.lower())
        uncertainty_map = {}
        
        for word in set(words):
            if word in self.semantics.nodes:
                node = self.semantics.nodes[word]
                uncertainty_map[word] = {
                    'uncertainty_level': node.uncertainty_level,
                    'creativity_factor': node.creativity_factor,
                    'reality_grounding': node.reality_grounding
                }
        
        # Find uncertainty bridges
        bridges = self.semantics.find_uncertainty_bridges(threshold=0.4)
        
        return {
            'word_uncertainties': uncertainty_map,
            'uncertainty_bridges': bridges[:5],  # Top 5 bridges
            'overall_uncertainty': sum(info['uncertainty_level'] for info in uncertainty_map.values()) / len(uncertainty_map) if uncertainty_map else 0
        }
    
    def _extract_practical_insights(self, linear: Dict, nonlinear: Dict) -> List[str]:
        """Extract practical insights for AI system improvement"""
        insights = []
        
        # Semantic density insights
        if linear['semantic_density'] < 0.3:
            insights.append("Low semantic density - consider expanding vocabulary or improving concept recognition")
        
        # Uncertainty handling insights
        if nonlinear['uncertainty_level'] > 0.7:
            insights.append("High uncertainty detected - may require human clarification or creative interpretation")
        
        # Cascade potential insights
        if nonlinear['cascade_potential'] > 0.6:
            insights.append("High cascade potential - responses may trigger complex chains of associations")
        
        # Creativity insights
        if nonlinear['creativity_factor'] > 0.5:
            insights.append("Creative content detected - system may generate novel or unexpected responses")
        
        # Grounding insights
        if nonlinear['reality_grounding'] < 0.3:
            insights.append("Low reality grounding - verify outputs against factual knowledge")
        
        # Emergence insights
        if nonlinear['emergence_likelihood'] > 0.5:
            insights.append("High emergence potential - monitor for unexpected concept combinations")
        
        return insights
    
    def evaluate_translation_quality(self, text: str, expected_concepts: List[str] = None, 
                                   expected_cascades: List[str] = None) -> Dict[str, Any]:
        """Evaluate translation quality with optional expected results"""
        expected_concepts = expected_concepts or []
        expected_cascades = expected_cascades or []
        
        # Basic understanding evaluation
        understanding_eval = self.evaluator.evaluate_understanding(text, expected_concepts)
        
        # Cascade evaluation if we have expected cascades
        cascade_eval = {}
        if expected_cascades:
            words = re.findall(r'\w+', text.lower())
            if words:
                primary_word = max(words, key=lambda w: self.semantics.nodes.get(w, SemanticNode(w, 'general', 'concept')).cascade_potential)
                cascade_eval = self.evaluator.evaluate_cascade_quality(primary_word, expected_cascades)
        
        return {
            'understanding_quality': understanding_eval,
            'cascade_quality': cascade_eval,
            'overall_score': understanding_eval['understanding_score'],
            'recommendations': self._generate_recommendations(understanding_eval, cascade_eval)
        }
    
    def _generate_recommendations(self, understanding_eval: Dict, cascade_eval: Dict) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        if understanding_eval['concept_coverage'] < 0.5:
            recommendations.append("Expand vocabulary coverage for better concept recognition")
        
        if understanding_eval['uncertainty_appropriateness'] < 0.6:
            recommendations.append("Improve uncertainty calibration for abstract vs concrete concepts")
        
        if understanding_eval['creativity_detection'] < 0.4:
            recommendations.append("Enhance creative content detection capabilities")
        
        if cascade_eval.get('cascade_coherence', 1.0) < 0.5:
            recommendations.append("Improve cascade coherence by strengthening semantic relationships")
        
        if understanding_eval['reality_grounding'] < 0.4:
            recommendations.append("Strengthen reality grounding for better factual accuracy")
        
        return recommendations

__all__ = ['PracticalTranslator']