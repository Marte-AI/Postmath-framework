"""
Visualization utilities for PostMath Framework
© 2025 Jesús Manuel Soledad Terrazas. All rights reserved.
"""

from typing import Dict, List, Any
import json


def visualize_cascade(cascade_data: Dict[str, Any]) -> None:
    """
    Visualize cascade data (requires matplotlib)
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Create directed graph
        G = nx.DiGraph()
        path = cascade_data['path']
        
        for i in range(len(path) - 1):
            G.add_edge(path[i], path[i + 1])
        
        # Draw graph
        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=3000, font_size=10, font_weight='bold',
                arrows=True, arrowsize=20, edge_color='gray')
        plt.title(f"Cascade from '{cascade_data['trigger']}'")
        plt.axis('off')
        
    except ImportError:
        print("Matplotlib not installed. Install with: pip install postmath[viz]")


def visualize_uncertainty_map(uncertainty_data: Dict[str, Any]) -> None:
    """
    Visualize uncertainty map (requires matplotlib)
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Extract word uncertainties
        words = list(uncertainty_data['word_uncertainties'].keys())
        uncertainties = [info['uncertainty_level'] 
                        for info in uncertainty_data['word_uncertainties'].values()]
        
        # Create heatmap data
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        
        # Bar plot of uncertainties
        plt.bar(range(len(words)), uncertainties, color='darkblue', alpha=0.7)
        plt.xticks(range(len(words)), words, rotation=45, ha='right')
        plt.ylabel('Uncertainty Level')
        plt.title('Word Uncertainty Levels')
        
        # Bridges visualization
        plt.subplot(1, 2, 2)
        bridges = uncertainty_data['uncertainty_bridges'][:5]  # Top 5
        if bridges:
            bridge_labels = [f"{b['source']}↔{b['target']}" for b in bridges]
            bridge_strengths = [b['uncertainty_bridge'] for b in bridges]
            
            plt.barh(range(len(bridge_labels)), bridge_strengths, color='darkred', alpha=0.7)
            plt.yticks(range(len(bridge_labels)), bridge_labels)
            plt.xlabel('Bridge Strength')
            plt.title('Top Uncertainty Bridges')
        
        plt.tight_layout()
        
    except ImportError:
        print("Matplotlib/Seaborn not installed. Install with: pip install postmath[viz]")


def save_cascade_graph(cascade_data: Dict[str, Any], filename: str) -> None:
    """
    Save cascade visualization to file
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        
        plt.figure(figsize=(10, 8))
        
        # Create graph
        G = nx.DiGraph()
        path = cascade_data['path']
        
        # Add edges with increasing weight for visual effect
        for i in range(len(path) - 1):
            G.add_edge(path[i], path[i + 1], weight=len(path) - i)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, 
                              edge_color='blue', arrows=True, arrowsize=20)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=3000, alpha=0.9)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        plt.title(f"Semantic Cascade: {cascade_data['trigger']} → ... ({cascade_data['length']} steps)", 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Add annotation
        plt.text(0.02, 0.02, "PostMath Framework ⇝cascade", 
                transform=plt.gca().transAxes, fontsize=10, 
                alpha=0.7, ha='left', va='bottom')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
    except ImportError:
        # Create a simple JSON representation instead
        with open(filename.replace('.png', '.json'), 'w') as f:
            json.dump(cascade_data, f, indent=2)
        print(f"Saved cascade data as JSON (install matplotlib for PNG: pip install postmath[viz])")


def save_uncertainty_heatmap(uncertainty_data: Dict[str, Any], filename: str) -> None:
    """
    Save uncertainty heatmap to file
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Word uncertainties
        words = list(uncertainty_data['word_uncertainties'].keys())
        uncertainties = [info['uncertainty_level'] 
                        for info in uncertainty_data['word_uncertainties'].values()]
        
        # Create gradient effect
        colors = plt.cm.Reds(np.array(uncertainties))
        
        bars1 = ax1.bar(range(len(words)), uncertainties, color=colors, edgecolor='black', linewidth=1)
        ax1.set_xticks(range(len(words)))
        ax1.set_xticklabels(words, rotation=45, ha='right')
        ax1.set_ylabel('Uncertainty Level', fontsize=12)
        ax1.set_title('Word Uncertainty Levels (Ψ∞^void)', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, val in zip(bars1, uncertainties):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Uncertainty bridges
        bridges = uncertainty_data['uncertainty_bridges'][:5]
        if bridges:
            bridge_labels = [f"{b['source']} ↔ {b['target']}" for b in bridges]
            bridge_strengths = [b['uncertainty_bridge'] for b in bridges]
            
            colors2 = plt.cm.Blues(np.array(bridge_strengths))
            bars2 = ax2.barh(range(len(bridge_labels)), bridge_strengths, 
                            color=colors2, edgecolor='black', linewidth=1)
            ax2.set_yticks(range(len(bridge_labels)))
            ax2.set_yticklabels(bridge_labels)
            ax2.set_xlabel('Bridge Strength', fontsize=12)
            ax2.set_title('Uncertainty Bridges (ΞΩ nexus)', fontsize=14, fontweight='bold')
            ax2.set_xlim(0, 1.1)
            
            # Add value labels
            for bar, val in zip(bars2, bridge_strengths):
                ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                        f'{val:.2f}', ha='left', va='center', fontsize=9)
        
        # Overall annotation
        fig.text(0.5, 0.02, f"PostMath Framework | Overall Uncertainty: {uncertainty_data['overall_uncertainty']:.3f}", 
                ha='center', fontsize=11, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
    except ImportError:
        # Create a simple JSON representation instead
        with open(filename.replace('.png', '.json'), 'w') as f:
            json.dump({
                'overall_uncertainty': uncertainty_data['overall_uncertainty'],
                'high_uncertainty_words': [w for w, info in uncertainty_data['word_uncertainties'].items() 
                                         if info['uncertainty_level'] > 0.5],
                'top_bridges': uncertainty_data['uncertainty_bridges'][:3]
            }, f, indent=2)
        print(f"Saved uncertainty data as JSON (install matplotlib for PNG: pip install postmath[viz])")