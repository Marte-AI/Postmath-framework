#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# © 2025 Jesús Manuel Soledad Terrazas. All rights reserved.
# Licensed under the PostMath Public Research License v1.0
#
# This file implements the core components of PostMath™.
# For non-commercial research and educational use only.
#
# PostMath™ is a trademark of Jesús Manuel Soledad Terrazas.
#
"""
PostMath Framework: Core Components
===================================

Core semantic processing components including nodes, relations, and 
dual-mode semantic processor.
"""

__author__ = "Jesús Manuel Soledad Terrazas"
__copyright__ = "© 2025 Jesús Manuel Soledad Terrazas"
__license__ = "PostMath Public Research License v1.0"
__version__ = "1.0.0"
__email__ = "jesussoledadt@gmail.com"

import re
import math
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Any, Optional, Union, Callable
from collections import defaultdict, Counter
from enum import Enum
import numpy as np

# =============================================================================
# LICENSE VALIDATION
# =============================================================================

class LicenseError(Exception):
    """Raised when PostMath license terms are violated"""
    pass

class PostMathLicense:
    """Validate PostMath license compliance"""
    
    ALLOWED_USES = ['research', 'education', 'non-commercial']
    
    @staticmethod
    def validate_use(use_case: str):
        """Validate that usage complies with license terms"""
        if use_case.lower() not in PostMathLicense.ALLOWED_USES:
            raise LicenseError(
                f"Commercial use prohibited. Contact {__email__} for licensing."
            )
    
    @staticmethod
    def log_usage():
        """Log usage for audit purposes"""
        import datetime
        import os
        usage_log = {
            'timestamp': datetime.datetime.now().isoformat(),
            'user': os.getenv('USER', 'unknown'),
            'file': __file__,
            'version': __version__,
            'copyright': __copyright__
        }
        # In production, this would log to file or send to server
        return usage_log

# =============================================================================
# CORE SEMANTIC COMPONENTS
# =============================================================================

@dataclass
class SemanticNode:
    """Simplified semantic node with dual-mode properties"""
    word: str
    domain: str
    node_type: str  # 'concept', 'relation', 'state', 'unknown'
    
    # Linear properties (traditional NLP)
    linear_embedding: List[float] = field(default_factory=lambda: [0.0] * 50)
    pos_tag: str = "NOUN"
    frequency: float = 1.0
    
    # Non-linear properties (PostMath inspired)
    cascade_potential: float = 0.0  # How likely to trigger cascades
    uncertainty_level: float = 0.0  # Void resonance - how much is unknown
    creativity_factor: float = 0.0  # How much this enables novel combinations
    reality_grounding: float = 1.0  # How well grounded in measurable reality
    
    # Dynamic properties
    activation: float = 0.0
    connections: Dict[str, float] = field(default_factory=dict)
    cascade_history: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize dynamic properties based on word characteristics"""
        self._compute_intrinsic_properties()
    
    def _compute_intrinsic_properties(self):
        """Compute properties based on word characteristics"""
        word_lower = self.word.lower()
        
        # High uncertainty for abstract/philosophical concepts
        uncertainty_triggers = ['consciousness', 'infinity', 'nothing', 'impossible', 
                              'transcendent', 'void', 'mystery', 'unknown', 'why']
        self.uncertainty_level = sum(0.2 for trigger in uncertainty_triggers 
                                   if trigger in word_lower)
        
        # High creativity for generative/dynamic concepts  
        creativity_triggers = ['create', 'generate', 'imagine', 'dream', 'invent',
                             'discover', 'emerge', 'evolve', 'transform']
        self.creativity_factor = sum(0.15 for trigger in creativity_triggers
                                   if trigger in word_lower)
        
        # High cascade potential for process/relationship words
        cascade_triggers = ['cause', 'lead', 'result', 'flow', 'spread', 'influence',
                          'connect', 'relate', 'link', 'bridge', 'trigger']
        self.cascade_potential = sum(0.1 for trigger in cascade_triggers
                                   if trigger in word_lower)
        
        # Lower reality grounding for abstract concepts
        abstract_triggers = ['love', 'beauty', 'justice', 'consciousness', 'soul',
                           'mind', 'spirit', 'meaning', 'purpose', 'essence']
        abstractness = sum(0.15 for trigger in abstract_triggers if trigger in word_lower)
        self.reality_grounding = max(0.1, 1.0 - abstractness)

@dataclass
class SemanticRelation:
    """Dynamic relationship between semantic nodes"""
    source: str
    target: str
    relation_type: str  # 'causal', 'similarity', 'part_of', 'opposite', etc.
    strength: float
    
    # Non-linear properties
    bidirectional: bool = False
    cascade_multiplier: float = 1.0  # How much this amplifies cascades
    uncertainty_bridge: float = 0.0  # How much this connects unknowns
    emergence_potential: float = 0.0  # How likely to create new meanings
    
    # Dynamic properties
    activation_count: int = 0
    last_activated: float = 0.0
    context_sensitivity: float = 0.5  # How much context affects this relation

class DualModeSemantics:
    """Core dual-mode semantic processor with PostMath operators"""
    
    def __init__(self):
        self.nodes: Dict[str, SemanticNode] = {}
        self.relations: Dict[Tuple[str, str], SemanticRelation] = {}
        self.domain_contexts: Dict[str, Dict] = self._initialize_domains()
        self.cascade_threshold = 0.3
        self.uncertainty_threshold = 0.5
        
    def _initialize_domains(self) -> Dict[str, Dict]:
        """Initialize domain-specific processing rules"""
        return {
            'technical': {
                'keywords': ['algorithm', 'system', 'process', 'function', 'data', 'network'],
                'cascade_modifier': 1.2,
                'uncertainty_modifier': 0.8,
                'grounding_requirement': 0.8
            },
            'creative': {
                'keywords': ['art', 'beauty', 'imagination', 'dream', 'story', 'music'],
                'cascade_modifier': 1.5,
                'uncertainty_modifier': 1.3,
                'grounding_requirement': 0.3
            },
            'philosophical': {
                'keywords': ['consciousness', 'existence', 'meaning', 'truth', 'reality'],
                'cascade_modifier': 1.1,
                'uncertainty_modifier': 1.8,
                'grounding_requirement': 0.2
            },
            'scientific': {
                'keywords': ['energy', 'matter', 'force', 'particle', 'wave', 'field'],
                'cascade_modifier': 1.0,
                'uncertainty_modifier': 0.6,
                'grounding_requirement': 0.9
            }
        }
    
    def add_node(self, word: str, domain: str = 'general', node_type: str = 'concept') -> SemanticNode:
        """Add or update a semantic node"""
        if word in self.nodes:
            return self.nodes[word]
        
        node = SemanticNode(word, domain, node_type)
        
        # Apply domain modifiers
        if domain in self.domain_contexts:
            domain_config = self.domain_contexts[domain]
            node.cascade_potential *= domain_config['cascade_modifier']
            node.uncertainty_level *= domain_config['uncertainty_modifier']
            node.reality_grounding = max(node.reality_grounding, 
                                       domain_config['grounding_requirement'])
        
        self.nodes[word] = node
        return node
    
    def add_relation(self, source: str, target: str, relation_type: str, 
                    strength: float = 1.0) -> SemanticRelation:
        """Add or update a semantic relation"""
        key = (source, target)
        
        if key in self.relations:
            # Update existing relation
            rel = self.relations[key]
            rel.strength = (rel.strength + strength) / 2  # Average
            return rel
        
        # Create new relation
        rel = SemanticRelation(source, target, relation_type, strength)
        
        # Compute dynamic properties
        source_node = self.nodes.get(source)
        target_node = self.nodes.get(target)
        
        if source_node and target_node:
            # Relations between uncertain concepts can bridge unknowns
            rel.uncertainty_bridge = (source_node.uncertainty_level + 
                                    target_node.uncertainty_level) / 2
            
            # Relations between creative concepts enable emergence
            rel.emergence_potential = (source_node.creativity_factor + 
                                     target_node.creativity_factor) / 2
            
            # Cascade multiplier based on node properties
            rel.cascade_multiplier = 1.0 + (source_node.cascade_potential * 0.5)
        
        self.relations[key] = rel
        
        # Update node connections
        if source in self.nodes:
            self.nodes[source].connections[target] = strength
        if target in self.nodes and rel.bidirectional:
            self.nodes[target].connections[source] = strength
            
        return rel
    
    def detect_domain(self, text: str) -> str:
        """Detect primary domain of text"""
        text_lower = text.lower()
        domain_scores = defaultdict(float)
        
        for domain, config in self.domain_contexts.items():
            for keyword in config['keywords']:
                if keyword in text_lower:
                    domain_scores[domain] += 1.0
        
        return max(domain_scores.items(), key=lambda x: x[1])[0] if domain_scores else 'general'
    
    def process_linear(self, text: str) -> Dict[str, Any]:
        """Traditional linear semantic processing"""
        words = re.findall(r'\w+', text.lower())
        
        # Basic linguistic analysis
        result = {
            'tokens': words,
            'unique_tokens': len(set(words)),
            'complexity': len(words) / len(set(words)) if words else 0,
            'domain': self.detect_domain(text),
            'pos_tags': [(word, self._simple_pos_tag(word)) for word in words[:10]]
        }
        
        # Semantic density
        known_words = sum(1 for word in words if word in self.nodes)
        result['semantic_density'] = known_words / len(words) if words else 0
        
        return result
    
    def process_nonlinear(self, text: str) -> Dict[str, Any]:
        """Non-linear semantic processing with PostMath insights"""
        words = re.findall(r'\w+', text.lower())
        domain = self.detect_domain(text)
        
        # Add nodes for unknown words
        for word in words:
            if word not in self.nodes:
                self.add_node(word, domain)
        
        # Calculate aggregate properties
        nodes = [self.nodes[word] for word in words if word in self.nodes]
        
        if not nodes:
            return {
                'cascade_potential': 0.0,
                'uncertainty_level': 1.0,  # Unknown text has high uncertainty
                'creativity_factor': 0.0,
                'reality_grounding': 0.0,
                'emergence_likelihood': 0.0
            }
        
        # Aggregate node properties
        avg_cascade = sum(n.cascade_potential for n in nodes) / len(nodes)
        avg_uncertainty = sum(n.uncertainty_level for n in nodes) / len(nodes)
        avg_creativity = sum(n.creativity_factor for n in nodes) / len(nodes)
        avg_grounding = sum(n.reality_grounding for n in nodes) / len(nodes)
        
        # Calculate emergence likelihood
        # High when creative concepts combine with uncertain concepts
        emergence = avg_creativity * avg_uncertainty * len(nodes) / 10.0
        emergence = min(1.0, emergence)
        
        # Apply domain modifiers
        if domain in self.domain_contexts:
            domain_config = self.domain_contexts[domain]
            avg_cascade *= domain_config['cascade_modifier']
            avg_uncertainty *= domain_config['uncertainty_modifier']
        
        return {
            'cascade_potential': min(1.0, avg_cascade),
            'uncertainty_level': min(1.0, avg_uncertainty),
            'creativity_factor': min(1.0, avg_creativity),
            'reality_grounding': avg_grounding,
            'emergence_likelihood': emergence,
            'domain_context': domain
        }
    
    def simulate_cascade(self, trigger_word: str, max_depth: int = 5) -> List[Dict]:
        """Simulate semantic cascade from trigger word (PostMath ⇝cascade operator)"""
        if trigger_word not in self.nodes:
            return []
        
        cascade_path = []
        current_activation = 1.0
        visited = set()
        
        def _cascade_step(word: str, activation: float, depth: int) -> List[Dict]:
            if depth >= max_depth or word in visited or activation < self.cascade_threshold:
                return []
            
            visited.add(word)
            node = self.nodes[word]
            
            step_info = {
                'word': word,
                'depth': depth,
                'activation': activation,
                'uncertainty': node.uncertainty_level,
                'creativity': node.creativity_factor,
                'cascade_potential': node.cascade_potential
            }
            
            results = [step_info]
            
            # Find next cascade targets
            for target, strength in node.connections.items():
                if target in self.nodes:
                    target_node = self.nodes[target]
                    
                    # Calculate cascade activation
                    cascade_activation = (activation * strength * 
                                        node.cascade_potential * 
                                        target_node.cascade_potential)
                    
                    # Add uncertainty amplification (void resonance)
                    if target_node.uncertainty_level > 0.5:
                        cascade_activation *= (1.0 + target_node.uncertainty_level * 0.5)
                    
                    if cascade_activation > self.cascade_threshold:
                        results.extend(_cascade_step(target, cascade_activation, depth + 1))
            
            return results
        
        return _cascade_step(trigger_word, current_activation, 0)
    
    def find_uncertainty_bridges(self, threshold: float = 0.5) -> List[Dict]:
        """Find relations that bridge between unknown/uncertain concepts (PostMath ΞΩ nexus)"""
        bridges = []
        
        for (source, target), relation in self.relations.items():
            if relation.uncertainty_bridge > threshold:
                source_node = self.nodes.get(source)
                target_node = self.nodes.get(target)
                
                if source_node and target_node:
                    bridge_info = {
                        'source': source,
                        'target': target,
                        'relation_type': relation.relation_type,
                        'uncertainty_bridge': relation.uncertainty_bridge,
                        'emergence_potential': relation.emergence_potential,
                        'source_uncertainty': source_node.uncertainty_level,
                        'target_uncertainty': target_node.uncertainty_level
                    }
                    bridges.append(bridge_info)
        
        return sorted(bridges, key=lambda x: x['uncertainty_bridge'], reverse=True)
    
    def _simple_pos_tag(self, word: str) -> str:
        """Simple POS tagging"""
        if word in ['the', 'a', 'an']:
            return 'DET'
        elif word in ['is', 'are', 'was', 'were', 'be', 'being', 'been']:
            return 'VERB'
        elif word in ['and', 'or', 'but', 'so', 'yet']:
            return 'CONJ'
        elif word.endswith('ly'):
            return 'ADV'
        elif word.endswith('ing'):
            return 'VERB'
        elif word.endswith('ed'):
            return 'VERB'
        else:
            return 'NOUN'

__all__ = [
    'LicenseError',
    'PostMathLicense',
    'SemanticNode',
    'SemanticRelation',
    'DualModeSemantics'
]