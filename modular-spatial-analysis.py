import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.spatial import distance
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import pandas as pd

# ============================================
# SECTION 1: BASIC VISUALIZATION FUNCTIONS
# ============================================

def visualize_curvature_distribution(curvature: np.ndarray, title: str = "Curvature Distribution"):
    """
    Visualize the distribution of curvature values with statistics.
    
    This is your starting point to understand the data.
    
    Parameters:
    -----------
    curvature : np.ndarray
        Array of curvature values
    title : str
        Title for the plot
    
    Example:
    --------
    >>> visualize_curvature_distribution(analyzer_2d.curvature, "2D Neuron Curvature")
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Histogram
    ax = axes[0]
    n, bins, patches = ax.hist(curvature, bins=50, density=True, alpha=0.7, color='blue')
    ax.axvline(np.mean(curvature), color='red', linestyle='--', label=f'Mean: {np.mean(curvature):.3f}')
    ax.axvline(np.median(curvature), color='green', linestyle='--', label=f'Median: {np.median(curvature):.3f}')
    ax.set_xlabel('Curvature (1/pixels)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Distribution')
    ax.legend()
    
    # 2. Box plot
    ax = axes[1]
    box = ax.boxplot(curvature, vert=True, patch_artist=True)
    box['boxes'][0].set_facecolor('lightblue')
    ax.set_ylabel('Curvature (1/pixels)')
    ax.set_title('Box Plot')
    ax.grid(True, alpha=0.3)
    
    # 3. Statistics summary
    ax = axes[2]
    ax.axis('off')
    stats_text = f"""Statistics Summary
{'='*25}
Count: {len(curvature)}
Mean: {np.mean(curvature):.4f}
Std Dev: {np.std(curvature):.4f}
Median: {np.median(curvature):.4f}
IQR: {np.percentile(curvature, 75) - np.percentile(curvature, 25):.4f}
Skewness: {scipy_stats.skew(curvature):.4f}
Kurtosis: {scipy_stats.kurtosis(curvature):.4f}

Percentiles:
  5th: {np.percentile(curvature, 5):.4f}
 25th: {np.percentile(curvature, 25):.4f}
 50th: {np.percentile(curvature, 50):.4f}
 75th: {np.percentile(curvature, 75):.4f}
 95th: {np.percentile(curvature, 95):.4f}
"""
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
            fontfamily='monospace', fontsize=10, va='top')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def create_basic_spatial_plot(mesh, curvature, title: str = "Spatial Curvature Distribution"):
    """
    Create a simple spatial visualization of curvature.
    
    This shows you where high/low curvatures are located.
    
    Parameters:
    -----------
    mesh : trimesh object
        The mesh with face centers
    curvature : np.ndarray
        Curvature values for each face
    
    Example:
    --------
    >>> create_basic_spatial_plot(analyzer_2d.mesh, analyzer_2d.curvature, "2D Neuron")
    """
    face_centers = mesh.triangles_center
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(face_centers[:, 0], face_centers[:, 1], 
                        c=curvature, s=0.5, cmap='RdBu_r',
                        vmin=np.percentile(curvature, 5),
                        vmax=np.percentile(curvature, 95))
    
    ax.set_xlabel('X position (pixels)')
    ax.set_ylabel('Y position (pixels)')
    ax.set_title(title)
    ax.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mean Curvature (1/pixels)')
    
    # Add some statistics to the plot
    stats_text = f'Mean: {np.mean(curvature):.3f}\nStd: {np.std(curvature):.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax


# ============================================
# SECTION 2: GRID-BASED ANALYSIS FUNCTIONS
# ============================================

def calculate_regional_statistics(mesh, curvature, grid_size: int = 10):
    """
    Calculate statistics for each region in a grid.
    
    This is the foundation for spatial heterogeneity analysis.
    
    Parameters:
    -----------
    mesh : trimesh object
    curvature : np.ndarray
    grid_size : int
        Number of grid divisions per axis
    
    Returns:
    --------
    dict : Contains region centers, means, counts, etc.
    
    Example:
    --------
    >>> region_stats = calculate_regional_statistics(analyzer_2d.mesh, analyzer_2d.curvature, 10)
    >>> print(f"Found {len(region_stats['means'])} valid regions")
    """
    face_centers = mesh.triangles_center
    bounds = mesh.bounds
    
    # Create grid
    x_bins = np.linspace(bounds[0][0], bounds[1][0], grid_size)
    y_bins = np.linspace(bounds[0][1], bounds[1][1], grid_size)
    
    # Storage for results
    region_centers = []
    region_means = []
    region_stds = []
    region_counts = []
    
    # Calculate for each grid cell
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            # Find faces in this cell
            mask = ((face_centers[:, 0] >= x_bins[i]) & 
                   (face_centers[:, 0] < x_bins[i+1]) &
                   (face_centers[:, 1] >= y_bins[j]) & 
                   (face_centers[:, 1] < y_bins[j+1]))
            
            n_faces = np.sum(mask)
            
            if n_faces > 10:  # Minimum faces for reliable statistics
                region_curvatures = curvature[mask]
                
                # Store region info
                region_centers.append([
                    (x_bins[i] + x_bins[i+1]) / 2,
                    (y_bins[j] + y_bins[j+1]) / 2
                ])
                region_means.append(np.mean(region_curvatures))
                region_stds.append(np.std(region_curvatures))
                region_counts.append(n_faces)
    
    return {
        'centers': np.array(region_centers),
        'means': np.array(region_means),
        'stds': np.array(region_stds),
        'counts': np.array(region_counts),
        'grid_size': grid_size,
        'x_bins': x_bins,
        'y_bins': y_bins
    }


def visualize_regional_statistics(mesh, curvature, region_stats):
    """
    Visualize the regional statistics calculated above.
    
    Parameters:
    -----------
    mesh : trimesh object
    curvature : np.ndarray
    region_stats : dict
        Output from calculate_regional_statistics
    
    Example:
    --------
    >>> region_stats = calculate_regional_statistics(analyzer_2d.mesh, analyzer_2d.curvature)
    >>> visualize_regional_statistics(analyzer_2d.mesh, analyzer_2d.curvature, region_stats)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Get face centers for background
    face_centers = mesh.triangles_center
    
    # 1. Regional means
    ax = axes[0, 0]
    # Background scatter
    ax.scatter(face_centers[:, 0], face_centers[:, 1], 
              c='lightgray', s=0.1, alpha=0.5)
    # Regional means
    scatter = ax.scatter(region_stats['centers'][:, 0], 
                        region_stats['centers'][:, 1],
                        c=region_stats['means'], 
                        s=region_stats['counts']/10,  # Size by number of faces
                        cmap='RdBu_r', edgecolor='black', linewidth=0.5)
    ax.set_title(f"Regional Mean Curvature (grid={region_stats['grid_size']})")
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Mean Curvature')
    
    # Add grid lines
    for x in region_stats['x_bins']:
        ax.axvline(x, color='gray', alpha=0.3, linewidth=0.5)
    for y in region_stats['y_bins']:
        ax.axhline(y, color='gray', alpha=0.3, linewidth=0.5)
    
    # 2. Regional standard deviations
    ax = axes[0, 1]
    ax.scatter(face_centers[:, 0], face_centers[:, 1], 
              c='lightgray', s=0.1, alpha=0.5)
    scatter = ax.scatter(region_stats['centers'][:, 0], 
                        region_stats['centers'][:, 1],
                        c=region_stats['stds'], 
                        s=region_stats['counts']/10,
                        cmap='viridis', edgecolor='black', linewidth=0.5)
    ax.set_title("Regional Curvature Std Dev")
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Std Dev')
    
    # 3. Distribution of regional means
    ax = axes[1, 0]
    ax.hist(region_stats['means'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(np.mean(region_stats['means']), color='red', linestyle='--',
              label=f"Mean: {np.mean(region_stats['means']):.3f}")
    ax.set_xlabel('Regional Mean Curvature')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Regional Means')
    ax.legend()
    
    # Calculate spatial heterogeneity
    heterogeneity = np.std(region_stats['means'])
    ax.text(0.02, 0.95, f'Spatial Heterogeneity: {heterogeneity:.3f}',
           transform=ax.transAxes, va='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 4. Face density map
    ax = axes[1, 1]
    ax.scatter(region_stats['centers'][:, 0], 
              region_stats['centers'][:, 1],
              c=region_stats['counts'], 
              s=100, cmap='plasma', edgecolor='black', linewidth=0.5)
    ax.set_title("Face Count per Region")
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, label='Number of Faces')
    
    plt.suptitle('Regional Curvature Analysis', fontsize=14)
    plt.tight_layout()
    return fig


# ============================================
# SECTION 3: SPATIAL HETEROGENEITY ANALYSIS
# ============================================

def test_grid_sizes(mesh, curvature, grid_sizes=None):
    """
    Test how spatial heterogeneity changes with grid size.
    
    This helps you find the optimal grid resolution.
    
    Parameters:
    -----------
    mesh : trimesh object
    curvature : np.ndarray
    grid_sizes : list
        Grid sizes to test (default: [5, 8, 10, 15, 20, 30, 40, 50])
    
    Returns:
    --------
    results : list of dicts
    
    Example:
    --------
    >>> results = test_grid_sizes(analyzer_2d.mesh, analyzer_2d.curvature)
    >>> for r in results[:3]:
    ...     print(f"Grid {r['grid_size']}: Heterogeneity = {r['heterogeneity']:.3f}")
    """
    if grid_sizes is None:
        grid_sizes = [5, 8, 10, 15, 20, 30, 40, 50]
    
    results = []
    
    for gs in grid_sizes:
        region_stats = calculate_regional_statistics(mesh, curvature, gs)
        
        if len(region_stats['means']) > 1:
            heterogeneity = np.std(region_stats['means'])
        else:
            heterogeneity = 0
        
        # Calculate coverage
        total_faces_in_regions = np.sum(region_stats['counts'])
        total_faces = len(curvature)
        coverage = total_faces_in_regions / total_faces * 100
        
        results.append({
            'grid_size': gs,
            'heterogeneity': heterogeneity,
            'n_valid_regions': len(region_stats['means']),
            'total_regions': (gs-1)**2,
            'coverage': coverage,
            'avg_faces_per_region': np.mean(region_stats['counts']) if len(region_stats['counts']) > 0 else 0
        })
        
        print(f"Grid {gs}: Het={heterogeneity:.3f}, Regions={len(region_stats['means'])}/{(gs-1)**2}, Coverage={coverage:.1f}%")
    
    return results


def plot_grid_size_analysis(results):
    """
    Visualize the results from test_grid_sizes.
    
    Parameters:
    -----------
    results : list
        Output from test_grid_sizes
    
    Example:
    --------
    >>> results = test_grid_sizes(analyzer_2d.mesh, analyzer_2d.curvature)
    >>> plot_grid_size_analysis(results)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    grid_sizes = [r['grid_size'] for r in results]
    
    # 1. Heterogeneity vs grid size
    ax = axes[0, 0]
    heterogeneities = [r['heterogeneity'] for r in results]
    ax.plot(grid_sizes, heterogeneities, 'o-', markersize=8, linewidth=2)
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Spatial Heterogeneity')
    ax.set_title('Heterogeneity vs Grid Resolution')
    ax.grid(True, alpha=0.3)
    
    # 2. Coverage
    ax = axes[0, 1]
    coverages = [r['coverage'] for r in results]
    ax.plot(grid_sizes, coverages, 'o-', markersize=8, linewidth=2, color='orange')
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Mesh Coverage by Valid Regions')
    ax.grid(True, alpha=0.3)
    ax.axhline(95, color='red', linestyle='--', label='95% threshold')
    ax.legend()
    
    # 3. Valid regions percentage
    ax = axes[1, 0]
    valid_pcts = [r['n_valid_regions']/r['total_regions']*100 for r in results]
    ax.plot(grid_sizes, valid_pcts, 'o-', markersize=8, linewidth=2, color='green')
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Valid Regions (%)')
    ax.set_title('Percentage of Grid Cells with >10 Faces')
    ax.grid(True, alpha=0.3)
    
    # 4. Average faces per region
    ax = axes[1, 1]
    avg_faces = [r['avg_faces_per_region'] for r in results]
    ax.plot(grid_sizes, avg_faces, 'o-', markersize=8, linewidth=2, color='purple')
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Average Faces per Region')
    ax.set_title('Region Density')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.suptitle('Grid Size Analysis', fontsize=14)
    plt.tight_layout()
    return fig


# ============================================
# SECTION 4: INTERPOLATION AND SMOOTHING
# ============================================

def create_interpolated_heatmap(mesh, curvature, resolution=100):
    """
    Create a smooth, interpolated heatmap of curvature.
    
    This gives you a continuous view of the curvature field.
    
    Parameters:
    -----------
    mesh : trimesh object
    curvature : np.ndarray
    resolution : int
        Grid resolution for interpolation
    
    Returns:
    --------
    fig, zi : figure and interpolated data
    
    Example:
    --------
    >>> fig, zi = create_interpolated_heatmap(analyzer_2d.mesh, analyzer_2d.curvature)
    """
    face_centers = mesh.triangles_center
    bounds = mesh.bounds
    
    # Create interpolation grid
    xi = np.linspace(bounds[0][0], bounds[1][0], resolution)
    yi = np.linspace(bounds[0][1], bounds[1][1], resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate curvature
    zi = griddata((face_centers[:, 0], face_centers[:, 1]), 
                  curvature, 
                  (xi_grid, yi_grid), 
                  method='linear')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(zi, extent=[bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]], 
                   origin='lower', cmap='RdBu_r', aspect='equal',
                   vmin=np.percentile(curvature, 5),
                   vmax=np.percentile(curvature, 95))
    
    ax.set_xlabel('X position (pixels)')
    ax.set_ylabel('Y position (pixels)')
    ax.set_title(f'Interpolated Curvature Heatmap ({resolution}x{resolution})')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Curvature (1/pixels)')
    
    # Add statistics
    stats_text = f'Non-NaN points: {np.sum(~np.isnan(zi))}/{zi.size}'
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig, zi


def create_smoothed_contour_plot(mesh, curvature, smooth_sigma=2):
    """
    Create contour plot with smoothing.
    
    This helps identify curvature patterns.
    
    Parameters:
    -----------
    mesh : trimesh object
    curvature : np.ndarray
    smooth_sigma : float
        Gaussian smoothing parameter
    
    Example:
    --------
    >>> create_smoothed_contour_plot(analyzer_2d.mesh, analyzer_2d.curvature)
    """
    # First create interpolated data
    face_centers = mesh.triangles_center
    bounds = mesh.bounds
    
    xi = np.linspace(bounds[0][0], bounds[1][0], 100)
    yi = np.linspace(bounds[0][1], bounds[1][1], 100)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    zi = griddata((face_centers[:, 0], face_centers[:, 1]), 
                  curvature, (xi_grid, yi_grid), method='linear')
    
    # Apply smoothing
    zi_smooth = gaussian_filter(zi, sigma=smooth_sigma)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Original contours
    levels = np.percentile(curvature[~np.isnan(curvature)], 
                          [5, 10, 25, 50, 75, 90, 95])
    
    contour1 = ax1.contour(xi_grid, yi_grid, zi, levels=levels, colors='black')
    ax1.clabel(contour1, inline=True, fontsize=8)
    contourf1 = ax1.contourf(xi_grid, yi_grid, zi, levels=20, cmap='RdBu_r', alpha=0.7)
    ax1.set_title('Original Contours')
    ax1.set_aspect('equal')
    plt.colorbar(contourf1, ax=ax1)
    
    # Smoothed contours
    contour2 = ax2.contour(xi_grid, yi_grid, zi_smooth, levels=levels, colors='black')
    ax2.clabel(contour2, inline=True, fontsize=8)
    contourf2 = ax2.contourf(xi_grid, yi_grid, zi_smooth, levels=20, cmap='RdBu_r', alpha=0.7)
    ax2.set_title(f'Smoothed Contours (σ={smooth_sigma})')
    ax2.set_aspect('equal')
    plt.colorbar(contourf2, ax=ax2)
    
    for ax in [ax1, ax2]:
        ax.set_xlabel('X position (pixels)')
        ax.set_ylabel('Y position (pixels)')
    
    plt.tight_layout()
    return fig


# ============================================
# SECTION 5: HOTSPOT DETECTION
# ============================================

def find_curvature_extremes(mesh, curvature, percentile=95):
    """
    Find regions of extreme curvature.
    
    Parameters:
    -----------
    mesh : trimesh object
    curvature : np.ndarray
    percentile : float
        Percentile threshold for "extreme" curvature
    
    Returns:
    --------
    dict with extreme curvature information
    
    Example:
    --------
    >>> extremes = find_curvature_extremes(analyzer_2d.mesh, analyzer_2d.curvature, 90)
    >>> print(f"Found {extremes['n_extreme']} extreme curvature faces")
    """
    face_centers = mesh.triangles_center
    
    # Find thresholds
    threshold_pos = np.percentile(curvature, percentile)
    threshold_neg = np.percentile(curvature, 100 - percentile)
    
    # Find extreme faces
    extreme_pos_mask = curvature > threshold_pos
    extreme_neg_mask = curvature < threshold_neg
    extreme_mask = extreme_pos_mask | extreme_neg_mask
    
    extreme_positions = face_centers[extreme_mask]
    extreme_values = curvature[extreme_mask]
    
    results = {
        'threshold_positive': threshold_pos,
        'threshold_negative': threshold_neg,
        'n_extreme': np.sum(extreme_mask),
        'n_positive': np.sum(extreme_pos_mask),
        'n_negative': np.sum(extreme_neg_mask),
        'percent_extreme': np.sum(extreme_mask) / len(curvature) * 100,
        'positions': extreme_positions,
        'values': extreme_values,
        'mask': extreme_mask
    }
    
    return results


def visualize_curvature_extremes(mesh, curvature, extremes):
    """
    Visualize the extreme curvature regions.
    
    Parameters:
    -----------
    mesh : trimesh object
    curvature : np.ndarray
    extremes : dict
        Output from find_curvature_extremes
    
    Example:
    --------
    >>> extremes = find_curvature_extremes(analyzer_2d.mesh, analyzer_2d.curvature, 90)
    >>> visualize_curvature_extremes(analyzer_2d.mesh, analyzer_2d.curvature, extremes)
    """
    face_centers = mesh.triangles_center
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Distribution with thresholds
    ax = axes[0]
    ax.hist(curvature, bins=100, alpha=0.7, color='gray', density=True)
    ax.axvline(extremes['threshold_positive'], color='red', linestyle='--', 
              label=f"Pos threshold: {extremes['threshold_positive']:.3f}")
    ax.axvline(extremes['threshold_negative'], color='blue', linestyle='--',
              label=f"Neg threshold: {extremes['threshold_negative']:.3f}")
    ax.set_xlabel('Curvature')
    ax.set_ylabel('Density')
    ax.set_title('Curvature Distribution with Extreme Thresholds')
    ax.legend()
    ax.set_yscale('log')
    
    # 2. Spatial view
    ax = axes[1]
    # All faces in gray
    ax.scatter(face_centers[:, 0], face_centers[:, 1], 
              c='lightgray', s=0.1, alpha=0.5)
    # Extreme faces colored
    scatter = ax.scatter(extremes['positions'][:, 0], 
                        extremes['positions'][:, 1],
                        c=extremes['values'], s=5, cmap='RdBu_r',
                        vmin=extremes['threshold_negative'],
                        vmax=extremes['threshold_positive'])
    ax.set_xlabel('X position (pixels)')
    ax.set_ylabel('Y position (pixels)')
    ax.set_title(f"Extreme Curvature Regions ({extremes['percent_extreme']:.1f}% of faces)")
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax)
    
    # 3. Statistics
    ax = axes[2]
    ax.axis('off')
    
    stats_text = f"""Extreme Curvature Analysis
{'='*30}
Total faces: {len(curvature)}
Extreme faces: {extremes['n_extreme']}
Percentage: {extremes['percent_extreme']:.2f}%

Positive extremes: {extremes['n_positive']}
  Threshold: >{extremes['threshold_positive']:.3f}
  
Negative extremes: {extremes['n_negative']}
  Threshold: <{extremes['threshold_negative']:.3f}

Extreme values:
  Max: {np.max(extremes['values']):.3f}
  Min: {np.min(extremes['values']):.3f}
  Mean: {np.mean(extremes['values']):.3f}
  Std: {np.std(extremes['values']):.3f}
"""
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=10, va='top')
    
    plt.tight_layout()
    return fig


def simple_cluster_extremes(extremes, eps=20):
    """
    Simple clustering of extreme curvature regions.
    
    Parameters:
    -----------
    extremes : dict
        Output from find_curvature_extremes
    eps : float
        Maximum distance between points in a cluster
    
    Returns:
    --------
    cluster_info : dict
    
    Example:
    --------
    >>> extremes = find_curvature_extremes(analyzer_2d.mesh, analyzer_2d.curvature, 90)
    >>> clusters = simple_cluster_extremes(extremes, eps=30)
    >>> print(f"Found {clusters['n_clusters']} clusters")
    """
    if len(extremes['positions']) < 10:
        print("Too few extreme points for clustering")
        return None
    
    from sklearn.cluster import DBSCAN
    
    # Cluster based on 2D positions
    clustering = DBSCAN(eps=eps, min_samples=10).fit(extremes['positions'][:, :2])
    labels = clustering.labels_
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Get info for each cluster
    cluster_info = {
        'n_clusters': n_clusters,
        'labels': labels,
        'n_noise': np.sum(labels == -1),
        'clusters': []
    }
    
    for i in range(n_clusters):
        mask = labels == i
        cluster_positions = extremes['positions'][mask]
        cluster_values = extremes['values'][mask]
        
        cluster_info['clusters'].append({
            'id': i,
            'size': np.sum(mask),
            'center': np.mean(cluster_positions, axis=0),
            'mean_curvature': np.mean(cluster_values),
            'std_curvature': np.std(cluster_values),
            'positions': cluster_positions,
            'values': cluster_values
        })
    
    return cluster_info


# ============================================
# SECTION 6: COMPARATIVE ANALYSIS
# ============================================

def compare_two_neurons(analyzer1, analyzer2, labels=['Neuron 1', 'Neuron 2']):
    """
    Basic comparison of two neurons.
    
    Parameters:
    -----------
    analyzer1, analyzer2 : MeshAnalyzer objects
    labels : list of str
        Labels for the two neurons
    
    Example:
    --------
    >>> compare_two_neurons(analyzer_2d, analyzer_3d, ['2D', '3D'])
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distributions overlay
    ax = axes[0, 0]
    bins = np.linspace(
        min(np.min(analyzer1.curvature), np.min(analyzer2.curvature)),
        max(np.max(analyzer1.curvature), np.max(analyzer2.curvature)),
        50
    )
    ax.hist(analyzer1.curvature, bins=bins, alpha=0.5, density=True, 
           label=labels[0], color='red')
    ax.hist(analyzer2.curvature, bins=bins, alpha=0.5, density=True,
           label=labels[1], color='blue')
    ax.set_xlabel('Curvature (1/pixels)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Curvature Distributions')
    ax.legend()
    
    # 2. Box plots
    ax = axes[0, 1]
    bp = ax.boxplot([analyzer1.curvature, analyzer2.curvature], 
                    labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('red')
    bp['boxes'][1].set_facecolor('blue')
    ax.set_ylabel('Curvature (1/pixels)')
    ax.set_title('Distribution Comparison')
    ax.grid(True, alpha=0.3)
    
    # 3. Spatial views side by side
    for idx, (analyzer, label) in enumerate([(analyzer1, labels[0]), 
                                            (analyzer2, labels[1])]):
        ax = axes[1, idx]
        face_centers = analyzer.mesh.triangles_center
        scatter = ax.scatter(face_centers[:, 0], face_centers[:, 1],
                           c=analyzer.curvature, s=0.5, cmap='RdBu_r',
                           vmin=np.percentile(analyzer1.curvature, 5),
                           vmax=np.percentile(analyzer1.curvature, 95))
        ax.set_title(f'{label} - Spatial Distribution')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax)
    
    plt.suptitle('Neuron Comparison', fontsize=14)
    plt.tight_layout()
    return fig


def calculate_comparison_metrics(analyzer1, analyzer2):
    """
    Calculate statistical comparison metrics.
    
    Parameters:
    -----------
    analyzer1, analyzer2 : MeshAnalyzer objects
    
    Returns:
    --------
    dict : Various comparison metrics
    
    Example:
    --------
    >>> metrics = calculate_comparison_metrics(analyzer_2d, analyzer_3d)
    >>> print(f"Cohen's d: {metrics['cohens_d']:.3f}")
    """
    curv1 = analyzer1.curvature
    curv2 = analyzer2.curvature
    
    # Basic statistics
    metrics = {
        'mean_diff': np.mean(curv1) - np.mean(curv2),
        'median_diff': np.median(curv1) - np.median(curv2),
        'std_ratio': np.std(curv1) / np.std(curv2),
        
        # Effect size (Cohen's d)
        'cohens_d': (np.mean(curv1) - np.mean(curv2)) / 
                   np.sqrt((np.std(curv1)**2 + np.std(curv2)**2) / 2),
        
        # Distribution tests
        'ks_statistic': scipy_stats.ks_2samp(curv1, curv2)[0],
        'ks_pvalue': scipy_stats.ks_2samp(curv1, curv2)[1],
        
        # Entropy comparison
        'entropy_1': calculate_curvature_entropy(curv1),
        'entropy_2': calculate_curvature_entropy(curv2),
        
        # Spatial heterogeneity (using grid size 20)
        'heterogeneity_1': calculate_spatial_heterogeneity(analyzer1.mesh, curv1, 20),
        'heterogeneity_2': calculate_spatial_heterogeneity(analyzer2.mesh, curv2, 20)
    }
    
    return metrics


# Reuse the entropy and heterogeneity functions from before
def calculate_curvature_entropy(curvature: np.ndarray, n_bins: int = 50) -> float:
    """Calculate Shannon entropy of curvature distribution."""
    hist, _ = np.histogram(curvature, bins=n_bins)
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist))
    return entropy

def calculate_spatial_heterogeneity(mesh, curvature: np.ndarray, grid_size: int = 10) -> float:
    """Calculate spatial heterogeneity."""
    region_stats = calculate_regional_statistics(mesh, curvature, grid_size)
    if len(region_stats['means']) > 1:
        return np.std(region_stats['means'])
    return 0


# ============================================
# USAGE EXAMPLE AND WORKFLOW
# ============================================

def example_analysis_workflow(analyzer):
    """
    Example workflow showing how to use these functions.
    
    Run this to see all functions in action!
    
    Parameters:
    -----------
    analyzer : MeshAnalyzer object
    
    Example:
    --------
    >>> example_analysis_workflow(analyzer_2d)
    """
    print("Starting comprehensive spatial analysis workflow...")
    print("=" * 60)
    
    # Step 1: Basic visualization
    print("\n1. Basic distribution analysis...")
    fig1 = visualize_curvature_distribution(analyzer.curvature, 
                                          "Curvature Distribution Analysis")
    
    # Step 2: Spatial view
    print("\n2. Creating spatial visualization...")
    fig2, ax = create_basic_spatial_plot(analyzer.mesh, analyzer.curvature,
                                        "Spatial Curvature Distribution")
    
    # Step 3: Regional analysis
    print("\n3. Calculating regional statistics...")
    region_stats = calculate_regional_statistics(analyzer.mesh, 
                                               analyzer.curvature, 
                                               grid_size=15)
    print(f"   Found {len(region_stats['means'])} valid regions")
    
    fig3 = visualize_regional_statistics(analyzer.mesh, analyzer.curvature, 
                                       region_stats)
    
    # Step 4: Grid size optimization
    print("\n4. Testing different grid sizes...")
    grid_results = test_grid_sizes(analyzer.mesh, analyzer.curvature)
    fig4 = plot_grid_size_analysis(grid_results)
    
    # Step 5: Smooth visualization
    print("\n5. Creating interpolated heatmap...")
    fig5, zi = create_interpolated_heatmap(analyzer.mesh, analyzer.curvature)
    
    # Step 6: Find extremes
    print("\n6. Finding extreme curvature regions...")
    extremes = find_curvature_extremes(analyzer.mesh, analyzer.curvature, 
                                     percentile=90)
    print(f"   Found {extremes['n_extreme']} extreme faces "
          f"({extremes['percent_extreme']:.1f}% of total)")
    
    fig6 = visualize_curvature_extremes(analyzer.mesh, analyzer.curvature, 
                                      extremes)
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check the generated figures.")
    
    return {
        'region_stats': region_stats,
        'grid_results': grid_results,
        'extremes': extremes
    }


# Quick test function
def quick_test(analyzer):
    """
    Quick test to ensure everything is working.
    
    Example:
    --------
    >>> quick_test(analyzer_2d)
    """
    print(f"Mesh info:")
    print(f"  Faces: {len(analyzer.mesh.faces)}")
    print(f"  Vertices: {len(analyzer.mesh.vertices)}")
    print(f"  Curvature values: {len(analyzer.curvature)}")
    print(f"\nCurvature stats:")
    print(f"  Mean: {np.mean(analyzer.curvature):.4f}")
    print(f"  Std: {np.std(analyzer.curvature):.4f}")
    print(f"  Range: [{np.min(analyzer.curvature):.4f}, {np.max(analyzer.curvature):.4f}]")