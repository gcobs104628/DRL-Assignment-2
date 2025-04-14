import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def count_states_in_weights(weights_file):
    """
    Count the number of states (patterns) in a weights file.
    
    Args:
        weights_file: Path to the weights file
        
    Returns:
        Dictionary with counts for each type of lookup table
    """
    try:
        with open(weights_file, 'rb') as f:
            weights = pickle.load(f)
            
        counts = {
            'n_tuples': sum(len(lut) for lut in weights['luts']),
            'large_tiles': len(weights['large_tile_lut']),
            'empty_cells': len(weights['empty_cells_lut']),
            'tile_types': len(weights['tile_types_lut']),
            'mergeable_pairs': len(weights['mergeable_pairs_lut']),
            'v_2v_pairs': len(weights['v_2v_pairs_lut']),
        }
        
        counts['total'] = sum(counts.values())
        
        return counts
    except Exception as e:
        print(f"Error analyzing {weights_file}: {e}")
        return None

def analyze_all_stages():
    """
    Analyze all stage weight files and print statistics.
    """
    weights_dir = os.path.join(os.path.dirname(__file__), "weights")
    results = {}
    
    # Analyze each stage
    for stage in range(1, 6):
        weights_file = os.path.join(weights_dir, f"stage{stage}_weights.pkl")
        if os.path.exists(weights_file):
            print(f"\nAnalyzing Stage {stage} weights...")
            counts = count_states_in_weights(weights_file)
            if counts:
                results[f"Stage {stage}"] = counts
                print(f"  N-Tuple patterns: {counts['n_tuples']}")
                print(f"  Large tile patterns: {counts['large_tiles']}")
                print(f"  Empty cells patterns: {counts['empty_cells']}")
                print(f"  Tile types patterns: {counts['tile_types']}")
                print(f"  Mergeable pairs patterns: {counts['mergeable_pairs']}")
                print(f"  Value-2×Value pairs patterns: {counts['v_2v_pairs']}")
                print(f"  Total patterns: {counts['total']}")
        else:
            print(f"Stage {stage} weights file not found at {weights_file}")
    
    return results

def plot_results(results):
    """
    Plot the results as a bar chart.
    
    Args:
        results: Dictionary of results from analyze_all_stages
    """
    if not results:
        print("No results to plot")
        return
    
    stages = list(results.keys())
    totals = [results[stage]['total'] for stage in stages]
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot total patterns
    ax1.bar(stages, totals, color='blue')
    ax1.set_title('Total Patterns by Stage')
    ax1.set_ylabel('Number of Patterns')
    ax1.set_xlabel('Training Stage')
    
    # Add values on top of bars
    for i, v in enumerate(totals):
        ax1.text(i, v + 0.05 * max(totals), f"{v:,}", 
                 ha='center', va='bottom', fontweight='bold')
    
    # Plot breakdown by pattern type
    pattern_types = ['n_tuples', 'large_tiles', 'empty_cells', 
                     'tile_types', 'mergeable_pairs', 'v_2v_pairs']
    
    # Prepare data for stacked bar chart
    data = {}
    for pattern_type in pattern_types:
        data[pattern_type] = [results[stage][pattern_type] for stage in stages]
    
    # Create stacked bar chart
    bottom = np.zeros(len(stages))
    for pattern_type in pattern_types:
        ax2.bar(stages, data[pattern_type], bottom=bottom, label=pattern_type)
        bottom += np.array(data[pattern_type])
    
    ax2.set_title('Pattern Types by Stage')
    ax2.set_ylabel('Number of Patterns')
    ax2.set_xlabel('Training Stage')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('weight_analysis.png')
    print("\nPlot saved as 'weight_analysis.png'")
    plt.show()

if __name__ == "__main__":
    print("Analyzing weights files for all stages...")
    results = analyze_all_stages()
    
    if results:
        print("\nPlotting results...")
        plot_results(results)
    else:
        print("No results to plot")
