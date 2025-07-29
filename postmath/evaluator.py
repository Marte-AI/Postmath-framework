#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# © 2025 Jesús Manuel Soledad Terrazas. All rights reserved.
# Licensed under the PostMath Public Research License v1.0
#
# This file implements the evaluation metrics for PostMath™.
# For non-commercial research and educational use only.
#
"""
PostMath Framework: Semantic Evaluator
======================================

Evaluation metrics and benchmarking for semantic processing quality.
"""

__author__ = "Jesús Manuel Soledad Terrazas"
__copyright__ = "© 2025 Jesús Manuel Soledad Terrazas"
__license__ = "PostMath Public Research License v1.0"
__version__ = "1.0.0"

from typing import Dict, List, Any
from collections import defaultdict

from .core import DualModeSemantics, SemanticNode


class SemanticEvaluator:
    """Evaluation metrics for semantic processing quality"""
    
    def __init__(self, semantic_processor: DualModeSemantics):
        self.processor = semantic_processor
        
    def evaluate_understanding(self, text: str, expected_concepts: List[str]) -> Dict[str, float]:
        """Evaluate how well the system understands given text"""
        linear_result = self.processor.process_linear(text)
        nonlinear_result = self.processor.process_nonlinear(text)
        
        # Concept coverage - how many expected concepts were recognized
        recognized_concepts = sum(1 for concept in expected_concepts 
                                if concept.lower() in [token.lower() for token in linear_result['tokens']])
        concept_coverage = recognized_concepts / len(expected_concepts) if expected_concepts else 0
        
        # Semantic density - how much meaning was extracted
        semantic_density = linear_result['semantic_density']
        
        # Uncertainty handling - appropriate uncertainty for abstract concepts
        uncertainty_appropriateness = self._evaluate_uncertainty_appropriateness(text, nonlinear_result)
        
        # Creativity detection - ability to identify creative/generative potential
        creativity_detection = nonlinear_result['creativity_factor']
        
        # Overall understanding score
        understanding_score = (concept_coverage * 0.3 + 
                             semantic_density * 0.2 + 
                             uncertainty_appropriateness * 0.2 + 
                             creativity_detection * 0.2 + 
                             nonlinear_result['reality_grounding'] * 0.1)
        
        return {
            'concept_coverage': concept_coverage,
            'semantic_density': semantic_density,
            'uncertainty_appropriateness': uncertainty_appropriateness,
            'creativity_detection': creativity_detection,
            'reality_grounding': nonlinear_result['reality_grounding'],
            'understanding_score': understanding_score,
            'emergence_likelihood': nonlinear_result['emergence_likelihood']
        }
    
    def _evaluate_uncertainty_appropriateness(self, text: str, nonlinear_result: Dict) -> float:
        """Evaluate whether uncertainty level is appropriate for the text"""
        text_lower = text.lower()
        
        # Abstract concepts should have higher uncertainty
        abstract_indicators = ['consciousness', 'love', 'beauty', 'meaning', 'purpose', 
                              'soul', 'spirit', 'essence', 'truth', 'reality']
        has_abstract = any(indicator in text_lower for indicator in abstract_indicators)
        
        # Concrete concepts should have lower uncertainty
        concrete_indicators = ['table', 'chair', 'car', 'house', 'book', 'water',
                              'tree', 'rock', 'computer', 'phone']
        has_concrete = any(indicator in text_lower for indicator in concrete_indicators)
        
        uncertainty = nonlinear_result['uncertainty_level']
        
        if has_abstract and not has_concrete:
            # Abstract text should have moderate to high uncertainty
            return 1.0 if uncertainty > 0.3 else uncertainty / 0.3
        elif has_concrete and not has_abstract:
            # Concrete text should have low uncertainty
            return 1.0 if uncertainty < 0.3 else (1.0 - uncertainty) / 0.7
        else:
            # Mixed or neutral text
            return 0.7  # Moderate score for mixed content
    
    def evaluate_cascade_quality(self, trigger_word: str, expected_cascades: List[str]) -> Dict[str, float]:
        """Evaluate cascade generation quality"""
        cascade_result = self.processor.simulate_cascade(trigger_word)
        
        if not cascade_result:
            return {
                'cascade_coverage': 0.0,
                'cascade_depth': 0.0,
                'cascade_coherence': 0.0,
                'cascade_novelty': 0.0
            }
        
        # Coverage - how many expected cascades were found
        cascade_words = [step['word'] for step in cascade_result]
        found_expected = sum(1 for expected in expected_cascades 
                           if expected.lower() in [word.lower() for word in cascade_words])
        cascade_coverage = found_expected / len(expected_cascades) if expected_cascades else 0
        
        # Depth - how deep the cascade went
        max_depth = max(step['depth'] for step in cascade_result)
        cascade_depth = min(1.0, max_depth / 5.0)  # Normalize to 0-1
        
        # Coherence - how related the cascade steps are
        coherence_scores = []
        for i in range(1, len(cascade_result)):
            prev_word = cascade_result[i-1]['word']
            curr_word = cascade_result[i]['word']
            coherence = self._calculate_word_similarity(prev_word, curr_word)
            coherence_scores.append(coherence)
        
        cascade_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
        
        # Novelty - how unexpected the cascade path is
        novelty_scores = [step['creativity'] * step['uncertainty'] for step in cascade_result]
        cascade_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0
        
        return {
            'cascade_coverage': cascade_coverage,
            'cascade_depth': cascade_depth,
            'cascade_coherence': cascade_coherence,
            'cascade_novelty': cascade_novelty,
            'cascade_steps': len(cascade_result)
        }
    
    def _calculate_word_similarity(self, word1: str, word2: str) -> float:
        """Simple word similarity calculation"""
        # Placeholder - in practice would use embeddings or WordNet
        if word1 == word2:
            return 1.0
        
        # Check if they share a domain
        node1 = self.processor.nodes.get(word1)
        node2 = self.processor.nodes.get(word2)
        
        if node1 and node2 and node1.domain == node2.domain:
            return 0.7
        
        # Check for common prefixes/suffixes
        common_chars = sum(1 for a, b in zip(word1, word2) if a == b)
        max_len = max(len(word1), len(word2))
        
        return common_chars / max_len if max_len > 0 else 0
    
    def benchmark_against_baseline(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """Benchmark against baseline traditional NLP"""
        results = {
            'dual_mode_scores': [],
            'linear_only_scores': [],
            'improvement_cases': [],
            'degradation_cases': []
        }
        
        for case in test_cases:
            text = case['text']
            expected_concepts = case.get('expected_concepts', [])
            expected_cascades = case.get('expected_cascades', [])
            
            # Dual-mode evaluation
            dual_eval = self.evaluate_understanding(text, expected_concepts)
            results['dual_mode_scores'].append(dual_eval['understanding_score'])
            
            # Linear-only evaluation (simulate traditional NLP)
            linear_score = dual_eval['concept_coverage'] * 0.7 + dual_eval['semantic_density'] * 0.3
            results['linear_only_scores'].append(linear_score)
            
            # Track improvements and degradations
            improvement = dual_eval['understanding_score'] - linear_score
            if improvement > 0.1:
                results['improvement_cases'].append({
                    'text': text,
                    'improvement': improvement,
                    'key_factors': {
                        'uncertainty_handling': dual_eval['uncertainty_appropriateness'],
                        'creativity_detection': dual_eval['creativity_detection'],
                        'emergence_likelihood': dual_eval['emergence_likelihood']
                    }
                })
            elif improvement < -0.1:
                results['degradation_cases'].append({
                    'text': text,
                    'degradation': abs(improvement),
                    'potential_causes': ['over_complexity', 'poor_grounding']
                })
        
        # Summary statistics
        if results['dual_mode_scores'] and results['linear_only_scores']:
            dual_avg = sum(results['dual_mode_scores']) / len(results['dual_mode_scores'])
            linear_avg = sum(results['linear_only_scores']) / len(results['linear_only_scores'])
            
            results['summary'] = {
                'dual_mode_average': dual_avg,
                'linear_average': linear_avg,
                'overall_improvement': dual_avg - linear_avg,
                'improvement_percentage': (dual_avg - linear_avg) / linear_avg * 100 if linear_avg > 0 else 0,
                'cases_improved': len(results['improvement_cases']),
                'cases_degraded': len(results['degradation_cases'])
            }
        
        return results

__all__ = ['SemanticEvaluator']