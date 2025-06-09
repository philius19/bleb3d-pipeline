import numpy as np
from scipy import stats as scipy_stats
from scipy.spatial import distance
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union
import pandas as pd
import seaborn as sns

# ========================================
# FEATURE EXTRACTION FUNCTIONS
# ========================================

def calculate_surface_roughness(curvature: np.ndarray) -> float:
    """
    Calculate surface roughness as coefficient of variation of curvature.
    
    Higher values indicate more complex, rough morphology.
    
    Parameters:
    -----------
    curvature : np.ndarray
        Array of curvature values
        
    Returns:
    --------
    float : Surface roughness metric
    """
    mean_abs_curv = np.mean(np.abs(curvature))
    if mean_abs_curv > 0:
        return np.std(curvature) / mean_abs_curv
    return 0


def calculate_curvature_entropy(curvature: np.ndarray, n_bins: int = 50) -> float:
    """
    Calculate Shannon entropy of curvature distribution.
    
    Higher entropy = more diverse curvature values = complex shape
    
    Parameters:
    -----------
    curvature : np.ndarray
        Array of curvature values
    n_bins : int
        Number of bins for histogram
        
    Returns:
    --------
    float : Entropy value
    """
    # Create histogram
    hist, _ = np.histogram(curvature, bins=n_bins)
    
    # Normalize to probability distribution
    hist = hist / np.sum(hist)
    
    # Remove zeros (log(0) is undefined)
    hist = hist[hist > 0]
    
    # Calculate entropy
    entropy = -np.sum(hist * np.log(hist))
    
    return entropy


def calculate_spatial_heterogeneity(mesh, curvature: np.ndarray, 
                                  grid_size: int = 10) -> float:
    """
    Calculate how heterogeneous curvature is across the surface.
    
    Divides surface into grid and calculates variance of regional means.
    
    Parameters:
    -----------
    mesh : trimesh object
        The mesh with face centers
    curvature : np.ndarray
        Curvature values for each face
    grid_size : int
        Number of grid divisions per axis
        
    Returns:
    --------
    float : Spatial heterogeneity score
    """
    # Get face centers
    face_centers = mesh.triangles_center
    
    # Get bounds
    bounds = mesh.bounds
    x_bins = np.linspace(bounds[0][0], bounds[1][0], grid_size)
    y_bins = np.linspace(bounds[0][1], bounds[1][1], grid_size)
    
    region_means = []
    
    # Calculate mean curvature in each grid cell
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            # Find faces in this grid cell
            mask = ((face_centers[:, 0] >= x_bins[i]) & 
                   (face_centers[:, 0] < x_bins[i+1]) &
                   (face_centers[:, 1] >= y_bins[j]) & 
                   (face_centers[:, 1] < y_bins[j+1]))
            
            # Need minimum number of faces for reliable mean
            if np.sum(mask) > 10:
                region_means.append(np.mean(curvature[mask]))
    
    # Heterogeneity = standard deviation of regional means
    if len(region_means) > 1:
        return np.std(region_means)
    return 0


def calculate_high_curvature_clustering(mesh, curvature: np.ndarray, 
                                      percentile: float = 90) -> float:
    """
    Calculate clustering index of high curvature regions.
    
    Lower values = more clustered high curvature regions
    
    Parameters:
    -----------
    mesh : trimesh object
        The mesh with face centers
    curvature : np.ndarray
        Curvature values
    percentile : float
        Percentile threshold for "high" curvature
        
    Returns:
    --------
    float : Clustering index (mean nearest neighbor distance)
    """
    # Get face centers
    face_centers = mesh.triangles_center
    
    # Find high curvature regions
    threshold = np.percentile(np.abs(curvature), percentile)
    high_curv_mask = np.abs(curvature) > threshold
    
    if np.sum(high_curv_mask) < 10:
        return np.nan
    
    high_curv_positions = face_centers[high_curv_mask]
    
    # Sample if too many points (for computational efficiency)
    if len(high_curv_positions) > 100:
        idx = np.random.choice(len(high_curv_positions), 100, replace=False)
        high_curv_positions = high_curv_positions[idx]
    
    # Calculate nearest neighbor distances
    nbr_distances = []
    for i, pos in enumerate(high_curv_positions):
        # Distance to all other high curvature points
        dists = distance.cdist([pos], high_curv_positions)[0]
        dists[i] = np.inf  # Exclude self
        
        if len(dists) > 1:
            nbr_distances.append(np.min(dists))
    
    return np.mean(nbr_distances) if nbr_distances else np.nan


def extract_morphological_features(analyzer) -> Dict[str, float]:
    """
    Extract comprehensive morphological features from a single neuron.
    
    Parameters:
    -----------
    analyzer : MeshAnalyzer object
        Analyzer with loaded mesh and calculated curvature
        
    Returns:
    --------
    dict : Dictionary of feature_name -> value
    """
    curvature = analyzer.curvature
    mesh = analyzer.mesh
    
    # Basic curvature statistics
    features = {
        # Central tendency
        'mean_curvature': np.mean(curvature),
        'median_curvature': np.median(curvature),
        'trimmed_mean_curvature': scipy_stats.trim_mean(curvature, 0.1),
        
        # Variability
        'curvature_std': np.std(curvature),
        'curvature_iqr': np.percentile(curvature, 75) - np.percentile(curvature, 25),
        'curvature_mad': np.median(np.abs(curvature - np.median(curvature))),
        
        # Distribution shape
        'curvature_skewness': scipy_stats.skew(curvature),
        'curvature_kurtosis': scipy_stats.kurtosis(curvature),
        
        # Shape characteristics
        'percent_convex': np.sum(curvature > 0) / len(curvature) * 100,
        'percent_concave': np.sum(curvature < 0) / len(curvature) * 100,
        'percent_flat': np.sum(np.abs(curvature) < 0.01) / len(curvature) * 100,
        
        # Extreme features
        'max_positive_curvature': np.max(curvature),
        'max_negative_curvature': np.min(curvature),
        'curvature_range': np.max(curvature) - np.min(curvature),
        
        # Percentiles
        'curvature_p05': np.percentile(curvature, 5),
        'curvature_p95': np.percentile(curvature, 95),
        
        # Absolute curvature features
        'mean_abs_curvature': np.mean(np.abs(curvature)),
        'total_abs_curvature': np.sum(np.abs(curvature)),
        
        # Normalized features
        'total_curvature_per_area': np.sum(np.abs(curvature)) / mesh.area,
        'high_curvature_density': np.sum(np.abs(curvature) > np.percentile(np.abs(curvature), 95)) / mesh.area,
        
        # Complexity metrics
        'surface_roughness': calculate_surface_roughness(curvature),
        'curvature_entropy': calculate_curvature_entropy(curvature),
        
        # Spatial features
        'spatial_heterogeneity': calculate_spatial_heterogeneity(mesh, curvature),
        'high_curvature_clustering': calculate_high_curvature_clustering(mesh, curvature),
        
        # Size features
        'surface_area_um2': mesh.area * analyzer.pixel_size_xy**2,
        'volume_um3': mesh.volume * analyzer.pixel_size_xy**2 * analyzer.pixel_size_z
    }
    
    return features


# ========================================
# STATISTICAL ANALYSIS FUNCTIONS
# ========================================

def test_normality(data: np.ndarray) -> Tuple[float, bool]:
    """
    Test if data follows normal distribution using Shapiro-Wilk test.
    
    Parameters:
    -----------
    data : np.ndarray
        Data to test
        
    Returns:
    --------
    tuple : (p_value, is_normal)
    """
    if len(data) < 3:
        return 0, False
    
    stat, p_value = scipy_stats.shapiro(data)
    is_normal = p_value > 0.05
    
    return p_value, is_normal


def calculate_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    
    Parameters:
    -----------
    group1, group2 : np.ndarray
        Data from two groups
        
    Returns:
    --------
    float : Cohen's d
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return d


def compare_feature_between_groups(feature_name: str, 
                                 features_2d: List[Dict], 
                                 features_3d: List[Dict]) -> Dict:
    """
    Compare a single feature between 2D and 3D groups.
    
    Parameters:
    -----------
    feature_name : str
        Name of feature to compare
    features_2d : List[Dict]
        List of feature dictionaries for 2D neurons
    features_3d : List[Dict]
        List of feature dictionaries for 3D neurons
        
    Returns:
    --------
    dict : Statistical comparison results
    """
    # Extract feature values
    values_2d = np.array([f[feature_name] for f in features_2d])
    values_3d = np.array([f[feature_name] for f in features_3d])
    
    # Remove any NaN values
    values_2d = values_2d[~np.isnan(values_2d)]
    values_3d = values_3d[~np.isnan(values_3d)]
    
    # Test normality
    _, normal_2d = test_normality(values_2d)
    _, normal_3d = test_normality(values_3d)
    
    # Choose appropriate test
    if normal_2d and normal_3d and len(values_2d) >= 5 and len(values_3d) >= 5:
        # Parametric test
        stat, p_value = scipy_stats.ttest_ind(values_2d, values_3d)
        test_used = 't-test'
    else:
        # Non-parametric test
        stat, p_value = scipy_stats.mannwhitneyu(values_2d, values_3d, alternative='two-sided')
        test_used = 'Mann-Whitney U'
    
    # Calculate effect size
    cohens_d = calculate_effect_size(values_2d, values_3d)
    
    # Create results dictionary
    results = {
        'feature': feature_name,
        'n_2d': len(values_2d),
        'n_3d': len(values_3d),
        'mean_2d': np.mean(values_2d),
        'std_2d': np.std(values_2d),
        'sem_2d': np.std(values_2d) / np.sqrt(len(values_2d)),
        'mean_3d': np.mean(values_3d),
        'std_3d': np.std(values_3d),
        'sem_3d': np.std(values_3d) / np.sqrt(len(values_3d)),
        'test': test_used,
        'statistic': stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'percent_change': ((np.mean(values_3d) - np.mean(values_2d)) / np.mean(values_2d) * 100 
                          if np.mean(values_2d) != 0 else np.nan)
    }
    
    return results


def perform_multiple_testing_correction(p_values: List[float], 
                                      method: str = 'fdr_bh') -> np.ndarray:
    """
    Correct p-values for multiple testing.
    
    Parameters:
    -----------
    p_values : List[float]
        Original p-values
    method : str
        Correction method ('fdr_bh', 'bonferroni', etc.)
        
    Returns:
    --------
    np.ndarray : Adjusted p-values
    """
    from statsmodels.stats.multitest import multipletests
    
    if not p_values:
        return np.array([])
    
    _, p_adjusted, _, _ = multipletests(p_values, method=method)
    
    return p_adjusted


# ========================================
# VISUALIZATION FUNCTIONS
# ========================================

def plot_feature_distributions(feature_name: str,
                             features_2d: List[Dict],
                             features_3d: List[Dict],
                             stats: Dict = None):
    """
    Create violin plot comparing feature distributions between groups.
    
    Parameters:
    -----------
    feature_name : str
        Feature to plot
    features_2d, features_3d : List[Dict]
        Feature dictionaries
    stats : Dict
        Optional statistics to display
    """
    # Extract values
    values_2d = [f[feature_name] for f in features_2d if not np.isnan(f[feature_name])]
    values_3d = [f[feature_name] for f in features_3d if not np.isnan(f[feature_name])]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Create violin plot
    parts = ax.violinplot([values_2d, values_3d], positions=[0, 1], 
                         showmeans=True, showmedians=True)
    
    # Color the violins
    colors = ['#E63946', '#2E86AB']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    
    # Add individual points
    np.random.seed(42)
    x_2d = np.random.normal(0, 0.04, size=len(values_2d))
    x_3d = np.random.normal(1, 0.04, size=len(values_3d))
    
    ax.scatter(x_2d, values_2d, alpha=0.6, s=30, color='#E63946', 
               edgecolor='black', linewidth=0.5)
    ax.scatter(x_3d, values_3d, alpha=0.6, s=30, color='#2E86AB', 
               edgecolor='black', linewidth=0.5)
    
    # Styling
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['2D', '3D'])
    ax.set_ylabel(feature_name.replace('_', ' ').title())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add statistics if provided
    if stats:
        y_max = max(max(values_2d), max(values_3d))
        y_range = y_max - min(min(values_2d), min(values_3d))
        
        # Add significance line
        ax.plot([0, 1], [y_max + 0.1*y_range, y_max + 0.1*y_range], 
                'k-', linewidth=1)
        
        # Add significance stars
        if stats['p_value'] < 0.001:
            sig_text = '***'
        elif stats['p_value'] < 0.01:
            sig_text = '**'
        elif stats['p_value'] < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        
        ax.text(0.5, y_max + 0.12*y_range, sig_text, 
                ha='center', va='bottom')
        
        # Add effect size
        ax.text(0.02, 0.98, f"Cohen's d = {stats['cohens_d']:.2f}\np = {stats['p_value']:.3f}",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax


def plot_effect_sizes(feature_stats: List[Dict], top_n: int = 15):
    """
    Create forest plot of effect sizes for all features.
    
    Parameters:
    -----------
    feature_stats : List[Dict]
        List of statistical results for each feature
    top_n : int
        Number of top features to show
    """
    # Sort by absolute effect size
    sorted_stats = sorted(feature_stats, key=lambda x: abs(x['cohens_d']), reverse=True)[:top_n]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract data
    features = [s['feature'].replace('_', ' ') for s in sorted_stats]
    effect_sizes = [s['cohens_d'] for s in sorted_stats]
    significant = [s.get('p_adjusted', s['p_value']) < 0.05 for s in sorted_stats]
    
    # Colors based on significance
    colors = ['#E63946' if sig else '#CCCCCC' for sig in significant]
    
    # Create horizontal bar plot
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, effect_sizes, color=colors, alpha=0.8)
    
    # Add reference lines
    ax.axvline(0, color='black', linewidth=1)
    ax.axvline(-0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(-0.8, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.axvline(0.8, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
    
    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel("Cohen's d (Effect Size)")
    ax.set_title('Morphological Differences: 2D vs 3D Cultivation')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add interpretation guide
    ax.text(0.98, 0.02, 'Small: |d|=0.2\nMedium: |d|=0.5\nLarge: |d|=0.8', 
            transform=ax.transAxes, va='bottom', ha='right', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add direction labels
    ax.text(0.02, 0.98, '← More in 2D', transform=ax.transAxes, 
            va='top', ha='left', fontsize=9, style='italic')
    ax.text(0.98, 0.98, 'More in 3D →', transform=ax.transAxes, 
            va='top', ha='right', fontsize=9, style='italic')
    
    plt.tight_layout()
    return fig, ax


def create_feature_heatmap(features_2d: List[Dict], features_3d: List[Dict], 
                          normalize: bool = True):
    """
    Create heatmap showing all features for all neurons.
    
    Parameters:
    -----------
    features_2d, features_3d : List[Dict]
        Feature dictionaries
    normalize : bool
        Whether to z-score normalize features
    """
    # Create DataFrames
    df_2d = pd.DataFrame(features_2d)
    df_2d['condition'] = '2D'
    df_2d['neuron_id'] = [f'2D_{i+1}' for i in range(len(df_2d))]
    
    df_3d = pd.DataFrame(features_3d)
    df_3d['condition'] = '3D'
    df_3d['neuron_id'] = [f'3D_{i+1}' for i in range(len(df_3d))]
    
    # Combine
    df_all = pd.concat([df_2d, df_3d], ignore_index=True)
    
    # Select numeric features only
    numeric_features = df_all.select_dtypes(include=[np.number]).columns
    numeric_features = [f for f in numeric_features if f not in ['neuron_id']]
    
    # Create matrix for heatmap
    data_matrix = df_all[numeric_features].values.T
    
    if normalize:
        # Z-score normalize each feature
        data_matrix = (data_matrix - np.mean(data_matrix, axis=1, keepdims=True)) / np.std(data_matrix, axis=1, keepdims=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap
    im = ax.imshow(data_matrix, aspect='auto', cmap='RdBu_r', 
                   vmin=-2 if normalize else None, 
                   vmax=2 if normalize else None)
    
    # Set ticks
    ax.set_xticks(range(len(df_all)))
    ax.set_xticklabels(df_all['neuron_id'], rotation=90)
    ax.set_yticks(range(len(numeric_features)))
    ax.set_yticklabels([f.replace('_', ' ') for f in numeric_features])
    
    # Add condition separator
    separator_pos = len(df_2d) - 0.5
    ax.axvline(separator_pos, color='black', linewidth=2)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Z-score' if normalize else 'Feature value')
    
    # Add labels
    ax.set_xlabel('Neurons')
    ax.set_ylabel('Features')
    ax.set_title('Morphological Feature Heatmap')
    
    plt.tight_layout()
    return fig, ax


# ========================================
# MAIN ANALYSIS PIPELINE
# ========================================

def analyze_neuron_populations(neurons_2d: List, neurons_3d: List, 
                             output_dir: Path = None) -> Dict:
    """
    Main analysis pipeline for comparing neuron populations.
    
    Parameters:
    -----------
    neurons_2d : List[MeshAnalyzer]
        List of 2D cultivated neurons
    neurons_3d : List[MeshAnalyzer]
        List of 3D cultivated neurons
    output_dir : Path
        Directory to save results (optional)
        
    Returns:
    --------
    dict : Complete analysis results
    """
    print(f"Analyzing {len(neurons_2d)} 2D neurons vs {len(neurons_3d)} 3D neurons")
    
    # Step 1: Extract features
    print("Extracting morphological features...")
    features_2d = [extract_morphological_features(n) for n in neurons_2d]
    features_3d = [extract_morphological_features(n) for n in neurons_3d]
    
    # Step 2: Statistical comparison for each feature
    print("Performing statistical comparisons...")
    feature_names = list(features_2d[0].keys())
    all_stats = []
    
    for feature in feature_names:
        stats = compare_feature_between_groups(feature, features_2d, features_3d)
        all_stats.append(stats)
    
    # Step 3: Multiple testing correction
    print("Applying multiple testing correction...")
    p_values = [s['p_value'] for s in all_stats]
    p_adjusted = perform_multiple_testing_correction(p_values)
    
    for i, stats in enumerate(all_stats):
        stats['p_adjusted'] = p_adjusted[i]
    
    # Step 4: Identify significant features
    significant_features = [s for s in all_stats if s['p_adjusted'] < 0.05]
    significant_features.sort(key=lambda x: abs(x['cohens_d']), reverse=True)
    
    # Step 5: Create summary
    summary = {
        'n_neurons_2d': len(neurons_2d),
        'n_neurons_3d': len(neurons_3d),
        'n_features_tested': len(feature_names),
        'n_significant': len(significant_features),
        'top_features': significant_features[:5],
        'largest_effects': sorted(all_stats, key=lambda x: abs(x['cohens_d']), reverse=True)[:5]
    }
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Neurons analyzed: {summary['n_neurons_2d']} (2D) vs {summary['n_neurons_3d']} (3D)")
    print(f"Features tested: {summary['n_features_tested']}")
    print(f"Significant after correction: {summary['n_significant']}")
    
    if significant_features:
        print("\nTop significant features:")
        for i, feat in enumerate(summary['top_features']):
            print(f"{i+1}. {feat['feature']}: d={feat['cohens_d']:.2f}, p={feat['p_adjusted']:.3f}")
    
    # Compile results
    results = {
        'features_2d': features_2d,
        'features_3d': features_3d,
        'statistics': all_stats,
        'summary': summary
    }
    
    # Save results if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results as CSV
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(output_dir / 'feature_statistics.csv', index=False)
        
        # Save summary
        with open(output_dir / 'analysis_summary.txt', 'w') as f:
            f.write(f"NEURON MORPHOLOGY ANALYSIS SUMMARY\n")
            f.write(f"==================================\n\n")
            f.write(f"Sample size: {summary['n_neurons_2d']} (2D), {summary['n_neurons_3d']} (3D)\n")
            f.write(f"Features tested: {summary['n_features_tested']}\n")
            f.write(f"Significant features: {summary['n_significant']}\n\n")
            
            if significant_features:
                f.write("SIGNIFICANT FEATURES (sorted by effect size):\n")
                f.write("-" * 50 + "\n")
                for feat in significant_features:
                    f.write(f"{feat['feature']}:\n")
                    f.write(f"  Effect size: {feat['cohens_d']:.3f}\n")
                    f.write(f"  P-value (adjusted): {feat['p_adjusted']:.3f}\n")
                    f.write(f"  Mean 2D: {feat['mean_2d']:.3f} ± {feat['sem_2d']:.3f}\n")
                    f.write(f"  Mean 3D: {feat['mean_3d']:.3f} ± {feat['sem_3d']:.3f}\n")
                    f.write(f"  Change: {feat['percent_change']:.1f}%\n\n")
    
    return results


# ========================================
# EXAMPLE USAGE IN JUPYTER
# ========================================

def example_single_neuron_exploration(analyzer):
    """
    Example of exploring features for a single neuron.
    
    Usage in Jupyter:
    -----------------
    # Load your neuron
    analyzer = MeshAnalyzer()
    analyzer.load_data('path/to/neuron.pkl')
    
    # Explore features
    features = extract_morphological_features(analyzer)
    
    # Print key features
    for key, value in features.items():
        print(f"{key}: {value:.3f}")
    """
    features = extract_morphological_features(analyzer)
    
    # Create visualization of feature values
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot 1: Curvature distribution
    ax = axes[0, 0]
    ax.hist(analyzer.curvature, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(features['mean_curvature'], color='red', linestyle='--', label='Mean')
    ax.axvline(features['median_curvature'], color='green', linestyle='--', label='Median')
    ax.set_xlabel('Curvature')
    ax.set_ylabel('Count')
    ax.set_title('Curvature Distribution')
    ax.legend()
    
    # Plot 2: Shape composition
    ax = axes[0, 1]
    sizes = [features['percent_convex'], features['percent_concave'], features['percent_flat']]
    labels = ['Convex', 'Concave', 'Flat']
    colors = ['red', 'blue', 'gray']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    ax.set_title('Surface Shape Composition')
    
    # Plot 3: Feature values bar chart
    ax = axes[1, 0]
    selected_features = ['surface_roughness', 'curvature_entropy', 
                        'spatial_heterogeneity', 'mean_abs_curvature']
    values = [features[f] for f in selected_features]
    ax.bar(range(len(selected_features)), values, color='purple', alpha=0.7)
    ax.set_xticks(range(len(selected_features)))
    ax.set_xticklabels([f.replace('_', '\n') for f in selected_features], rotation=45)
    ax.set_ylabel('Value')
    ax.set_title('Complexity Metrics')
    
    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"NEURON SUMMARY\n" + "-"*30 + "\n"
    summary_text += f"Surface area: {features['surface_area_um2']:.1f} μm²\n"
    summary_text += f"Volume: {features['volume_um3']:.1f} μm³\n"
    summary_text += f"Mean curvature: {features['mean_curvature']:.3f}\n"
    summary_text += f"Surface roughness: {features['surface_roughness']:.3f}\n"
    summary_text += f"High curv. clustering: {features['high_curvature_clustering']:.3f}"
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontfamily='monospace', fontsize=10, va='top')
    
    plt.tight_layout()
    plt.show()
    
    return features