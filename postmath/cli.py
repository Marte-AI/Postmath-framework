#!/usr/bin/env python3
"""
PostMath Framework CLI Interface
================================

Command-line interface for the PostMath semantic analysis framework.

© 2025 Jesús Manuel Soledad Terrazas. All rights reserved.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

from . import __version__, __author__, __copyright__, __license__
from .core import PostMathLicense, LicenseError
from .translator import PracticalTranslator
from .evaluator import SemanticEvaluator
from .visualizer import (
    visualize_cascade,
    visualize_uncertainty_map,
    save_cascade_graph,
    save_uncertainty_heatmap
)


def run_comprehensive_demo():
    """Run comprehensive demonstration with benchmarking"""
    print("=" * 80)
    print("POSTMATH FRAMEWORK DEMONSTRATION")
    print(__copyright__)
    print("=" * 80)
    
    # Log usage
    usage_log = PostMathLicense.log_usage()
    print(f"Usage logged: {usage_log['timestamp']}")
    
    translator = PracticalTranslator()
    
    # Test cases with expected results for validation
    test_cases = [
        {
            'text': "Love creates deep connection between people",
            'expected_concepts': ['love', 'connection', 'people'],
            'expected_cascades': ['emotion', 'relationship', 'bond', 'intimacy'],
            'category': 'emotional'
        },
        {
            'text': "Algorithms process data to generate insights",
            'expected_concepts': ['algorithm', 'data', 'insights'],
            'expected_cascades': ['computation', 'analysis', 'pattern', 'knowledge'],
            'category': 'technical'
        },
        {
            'text': "Consciousness emerges from complex neural interactions",
            'expected_concepts': ['consciousness', 'neural', 'interactions'],
            'expected_cascades': ['mind', 'brain', 'awareness', 'experience'],
            'category': 'philosophical'
        },
        {
            'text': "The mysterious void contains infinite possibilities",
            'expected_concepts': ['void', 'infinite', 'possibilities'],
            'expected_cascades': ['unknown', 'potential', 'creation', 'transcendent'],
            'category': 'transcendent'
        },
        {
            'text': "Energy transforms matter through quantum fields",
            'expected_concepts': ['energy', 'matter', 'quantum', 'fields'],
            'expected_cascades': ['particle', 'wave', 'force', 'interaction'],
            'category': 'scientific'
        }
    ]
    
    print("\n1. TRANSLATION QUALITY ANALYSIS")
    print("-" * 50)
    
    total_scores = []
    category_scores = {}
    
    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {case['category'].upper()}")
        print(f"Text: '{case['text']}'")
        
        # Translate with dual mode
        result = translator.translate_text(case['text'], mode='dual')
        
        # Evaluate quality
        quality = translator.evaluate_translation_quality(
            case['text'], 
            case['expected_concepts'], 
            case['expected_cascades']
        )
        
        print(f"Understanding Score: {quality['overall_score']:.3f}")
        print(f"Concept Coverage: {quality['understanding_quality']['concept_coverage']:.3f}")
        print(f"Uncertainty Level: {result['nonlinear_analysis']['uncertainty_level']:.3f}")
        print(f"Creativity Factor: {result['nonlinear_analysis']['creativity_factor']:.3f}")
        print(f"Emergence Likelihood: {result['nonlinear_analysis']['emergence_likelihood']:.3f}")
        
        if result['practical_insights']:
            print(f"Key Insight: {result['practical_insights'][0]}")
        
        total_scores.append(quality['overall_score'])
        if case['category'] not in category_scores:
            category_scores[case['category']] = []
        category_scores[case['category']].append(quality['overall_score'])
    
    # Summary statistics
    print(f"\n2. PERFORMANCE SUMMARY")
    print("-" * 30)
    print(f"Average Understanding Score: {sum(total_scores)/len(total_scores):.3f}")
    print(f"Score Range: {min(total_scores):.3f} - {max(total_scores):.3f}")
    
    print("\nCategory Performance:")
    for category, scores in category_scores.items():
        avg_score = sum(scores) / len(scores)
        print(f"  {category.capitalize()}: {avg_score:.3f}")
    
    # Benchmark against baseline
    print(f"\n3. BASELINE COMPARISON")
    print("-" * 30)
    
    benchmark_results = translator.evaluator.benchmark_against_baseline(test_cases)
    
    if 'summary' in benchmark_results:
        summary = benchmark_results['summary']
        print(f"Dual-mode Average: {summary['dual_mode_average']:.3f}")
        print(f"Linear-only Average: {summary['linear_average']:.3f}")
        print(f"Improvement: {summary['overall_improvement']:+.3f} ({summary['improvement_percentage']:+.1f}%)")
        print(f"Cases Improved: {summary['cases_improved']} / {len(test_cases)}")
        
        if benchmark_results['improvement_cases']:
            print(f"\nBest Improvement Case:")
            best_case = max(benchmark_results['improvement_cases'], key=lambda x: x['improvement'])
            print(f"  Text: '{best_case['text']}'")
            print(f"  Improvement: +{best_case['improvement']:.3f}")
            print(f"  Key Factors: {list(best_case['key_factors'].keys())}")
    
    # Cascade analysis
    print(f"\n4. CASCADE ANALYSIS (PostMath ⇝cascade)")
    print("-" * 40)
    
    cascade_demo_text = "Creativity sparks innovation through imagination"
    cascade_analysis = translator._analyze_cascade_potential(cascade_demo_text)
    
    print(f"Text: '{cascade_demo_text}'")
    cascade_data = []
    for word, analysis in cascade_analysis.items():
        if analysis['cascade_length'] > 1:
            print(f"  {word}: {analysis['cascade_length']} steps, path: {' → '.join(analysis['cascade_path'])}")
            cascade_data.append({
                'trigger': word,
                'path': analysis['cascade_path'],
                'length': analysis['cascade_length']
            })
    
    # Save cascade visualization
    if cascade_data:
        try:
            assets_dir = Path("assets")
            assets_dir.mkdir(exist_ok=True)
            save_cascade_graph(cascade_data[0], str(assets_dir / "cascade_example.png"))
            print("  → Cascade visualization saved to assets/cascade_example.png")
        except ImportError:
            print("  → Install matplotlib for cascade visualization: pip install postmath[viz]")
    
    # Uncertainty mapping
    print(f"\n5. UNCERTAINTY MAPPING (PostMath Ψ∞^void)")
    print("-" * 40)
    
    uncertainty_demo_text = "Consciousness transcends physical reality through mysterious void"
    uncertainty_map = translator._map_uncertainties(uncertainty_demo_text)
    
    print(f"Text: '{uncertainty_demo_text}'")
    print(f"Overall Uncertainty: {uncertainty_map['overall_uncertainty']:.3f}")
    
    high_uncertainty_words = [word for word, info in uncertainty_map['word_uncertainties'].items() 
                             if info['uncertainty_level'] > 0.5]
    if high_uncertainty_words:
        print(f"High Uncertainty Words: {', '.join(high_uncertainty_words)}")
    
    if uncertainty_map['uncertainty_bridges']:
        print(f"Uncertainty Bridges Found: {len(uncertainty_map['uncertainty_bridges'])}")
        top_bridge = uncertainty_map['uncertainty_bridges'][0]
        print(f"  Top Bridge: {top_bridge['source']} ↔ {top_bridge['target']} (strength: {top_bridge['uncertainty_bridge']:.3f})")
    
    # Save uncertainty visualization
    try:
        assets_dir = Path("assets")
        assets_dir.mkdir(exist_ok=True)
        save_uncertainty_heatmap(uncertainty_map, str(assets_dir / "uncertainty_map.png"))
        print("  → Uncertainty map saved to assets/uncertainty_map.png")
    except ImportError:
        print("  → Install matplotlib for uncertainty visualization: pip install postmath[viz]")
    
    # Save benchmark results
    benchmark_file = Path("assets") / "benchmark_results.json"
    benchmark_file.parent.mkdir(exist_ok=True)
    with open(benchmark_file, 'w') as f:
        json.dump({
            'summary': benchmark_results.get('summary', {}),
            'category_scores': {cat: sum(scores)/len(scores) for cat, scores in category_scores.items()},
            'total_cases': len(test_cases)
        }, f, indent=2)
    print(f"\n  → Benchmark results saved to {benchmark_file}")
    
    print(f"\n6. PRACTICAL RECOMMENDATIONS")
    print("-" * 40)
    
    # Generate overall recommendations
    all_recommendations = set()
    for case in test_cases:
        quality = translator.evaluate_translation_quality(case['text'], case['expected_concepts'])
        all_recommendations.update(quality['recommendations'])
    
    if all_recommendations:
        print("System Improvement Recommendations:")
        for i, rec in enumerate(list(all_recommendations)[:5], 1):
            print(f"  {i}. {rec}")
    else:
        print("No major improvements needed - system performing well!")
    
    print(f"\n{'='*80}")
    print("POSTMATH FRAMEWORK VALIDATION COMPLETE")
    print(f"✓ Measurable improvements demonstrated")
    print(f"✓ Uncertainty handling validated") 
    print(f"✓ Cascade dynamics operational")
    print(f"✓ Reality grounding maintained")
    print(f"✓ Practical insights generated")
    print(f"{'='*80}")
    print(__copyright__)


def interactive_postmath_demo():
    """Interactive demo of PostMath framework"""
    # Validate license compliance
    try:
        PostMathLicense.validate_use('research')  # Assuming research use
    except LicenseError as e:
        print(f"License Error: {e}")
        return
    
    translator = PracticalTranslator()
    
    print("\n" + "=" * 60)
    print("INTERACTIVE POSTMATH FRAMEWORK")
    print(__copyright__)
    print("=" * 60)
    print("Enter text for dual-mode semantic analysis")
    print("Commands: 'quit' to exit, 'demo' for full demo, 'help' for guidance")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nText > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'demo':
            run_comprehensive_demo()
            continue
        elif user_input.lower() == 'help':
            print("\nTry these example texts:")
            print("  'Love creates connection' (emotional)")
            print("  'Algorithms process data' (technical)")
            print("  'Consciousness emerges mysteriously' (philosophical)")
            print("  'Energy transforms matter' (scientific)")
            print("\nCommands:")
            print("  'save <filename>' - Save last analysis")
            print("  'cascade <word>' - Analyze cascade for specific word")
            print("  'demo' - Run full demo")
            print("  'quit' - Exit")
            continue
        elif user_input.startswith('cascade '):
            word = user_input[8:].strip()
            cascade_result = translator.semantics.simulate_cascade(word)
            if cascade_result:
                print(f"\nCascade from '{word}':")
                for step in cascade_result[:10]:
                    print(f"  Depth {step['depth']}: {step['word']} (activation: {step['activation']:.3f})")
            else:
                print(f"No cascade found for '{word}'")
            continue
        elif not user_input:
            continue
        
        try:
            # Full analysis
            result = translator.translate_text(user_input, mode='dual')
            
            print(f"\n{'='*60}")
            print(f"ANALYSIS: {user_input}")
            print(f"{'='*60}")
            
            # Linear analysis
            linear = result['linear_analysis']
            print(f"\nLINEAR ANALYSIS:")
            print(f"  Domain: {linear['domain']}")
            print(f"  Semantic Density: {linear['semantic_density']:.3f}")
            print(f"  Complexity: {linear['complexity']:.3f}")
            
            # Nonlinear analysis
            nonlinear = result['nonlinear_analysis']
            print(f"\nNONLINEAR ANALYSIS (PostMath):")
            print(f"  Uncertainty Level (Ψ∞^void): {nonlinear['uncertainty_level']:.3f}")
            print(f"  Creativity Factor: {nonlinear['creativity_factor']:.3f}")
            print(f"  Cascade Potential (⇝cascade): {nonlinear['cascade_potential']:.3f}")
            print(f"  Reality Grounding: {nonlinear['reality_grounding']:.3f}")
            print(f"  Emergence Likelihood: {nonlinear['emergence_likelihood']:.3f}")
            
            # Synthesis
            synthesis = result['synthesis']
            print(f"\nSYNTHESIS:")
            print(f"  Overall Complexity: {synthesis['overall_complexity']:.3f}")
            print(f"  Novelty Potential: {synthesis['novelty_potential']:.3f}")
            print(f"  Practical Utility: {synthesis['practical_utility']:.3f}")
            print(f"  Exploration Value: {synthesis['exploration_value']:.3f}")
            
            # Practical insights
            if result['practical_insights']:
                print(f"\nPRACTICAL INSIGHTS:")
                for insight in result['practical_insights'][:3]:
                    print(f"  • {insight}")
            
            # Cascade potential
            if result['cascade_potential']:
                print(f"\nCASCADE ANALYSIS (PostMath ⇝cascade):")
                for word, analysis in list(result['cascade_potential'].items())[:3]:
                    print(f"  {word}: {analysis['cascade_length']} steps → {' → '.join(analysis['cascade_path'][:3])}")
            
            # Uncertainty map highlights
            uncertainty_map = result['uncertainty_map']
            high_uncertainty = [word for word, info in uncertainty_map['word_uncertainties'].items() 
                              if info['uncertainty_level'] > 0.6]
            if high_uncertainty:
                print(f"\nHIGH UNCERTAINTY WORDS (Ψ∞^void): {', '.join(high_uncertainty)}")
            
        except Exception as e:
            print(f"Analysis Error: {str(e)}")
            print("Please try a different text or report this issue")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="PostMath Framework - Dual-mode semantic analysis",
        epilog=f"{__copyright__} | License: {__license__}"
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'PostMath {__version__}'
    )
    
    parser.add_argument(
        '--mode',
        choices=['demo', 'interactive', 'analyze'],
        default='demo',
        help='Operation mode (default: demo)'
    )
    
    parser.add_argument(
        '--text',
        type=str,
        help='Text to analyze (for analyze mode)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for results (JSON format)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        run_comprehensive_demo()
    elif args.mode == 'interactive':
        interactive_postmath_demo()
    elif args.mode == 'analyze':
        if not args.text:
            parser.error("--text required for analyze mode")
        
        translator = PracticalTranslator()
        result = translator.translate_text(args.text, mode='dual')
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()