"""
Visualization functions for mesh analysis.
"""
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt



def plot_curvature_distribution(curvature: np.ndarray, 
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot curvature distribution with linear and log scale.
    
    Parameters:
        curvature: Array of curvature values
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Linear scale
    ax1.hist(curvature, bins=100, alpha=0.7)
    ax1.axvline(0, color='red', linestyle='--', label='Zero')
    ax1.set_xlabel('Curvature (1/pixels)')
    ax1.set_ylabel('Count')
    ax1.set_title('Curvature Distribution')
    ax1.legend()

    # Log scale
    non_zero_curv = curvature[curvature != 0]
    ax2.hist(non_zero_curv, bins=100, alpha=0.7)
    ax2.set_yscale('log')
    ax2.set_xlabel('Curvature (1/pixels)')
    ax2.set_ylabel('Count (log scale)')
    ax2.set_title('Curvature Distribution (Log Scale)')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

# ========== Enter Function for 3D Visualisation ==========
# ========== Enter Function for 3D Plot ==========

def plot_vertice_distribution(vertices: np.ndarray, 
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot vertice distribution.
    
    Parameters:
        vertices: Array of vertice values
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """

    # Create visualization grid
    fig = plt.figure(figsize=(20, 15))

    # 1. 3D Overview (sampled for performance)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')

    sample_idx = np.random.choice(len(vertices), min(10000, len(vertices)), replace=False)
    ax1.scatter(vertices[sample_idx, 0],                                           # --> graps X-coordinates
            vertices[sample_idx, 1],                                               # ---> graps Y-coordinates
            vertices[sample_idx, 2],                                               # --> graps Z-coordinates
            c=vertices[sample_idx, 2],                                             # Color by Z
            cmap='viridis', s=0.5, alpha=0.6)
    ax1.set_title('3D Shape Overview (sampled)')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')


    # 2. Projections (to understand orientation)
    ax2 = fig.add_subplot(2, 3, 2)                                                 # Subplot on pos. 2 in a 2×3 grid (3D)
    ax2.scatter(vertices[sample_idx, 0], vertices[sample_idx, 1],                  # Plots only X,Y coordinates, ignoring Z
                s=0.1, alpha=0.5, c=vertices[sample_idx, 2], cmap='viridis')       # Colors by Z-coordinate (Gives depth perception)
    ax2.set_aspect('equal')                                                        # Makes 1 pixel in X = 1 pixel in Y visually
    ax2.set_title('Top View (XY)')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y')

    # XZ projection
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(vertices[sample_idx, 0], vertices[sample_idx, 2],                  # Plots X,Z coordinates, ignoring Y
                s=0.1, alpha=0.5, c=vertices[sample_idx, 1], cmap='plasma')        # Colors by Y-coordinate (Gives depth perception)
    ax3.set_aspect('equal')
    ax3.set_title('Side View (XZ)')
    ax3.set_xlabel('X'); ax3.set_ylabel('Z')

    # 4. Vertex density heatmap
    ax4 = fig.add_subplot(2, 3, 4)
    H, xedges, yedges = np.histogram2d(vertices[:, 0], vertices[:, 1], bins=50) 
    ax4.imshow(H.T, origin='lower', cmap='hot', aspect='auto',                      # Displays the counts as a heatmap
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    ax4.set_title('Vertex Density (XY plane)')
    ax4.set_xlabel('X'); ax4.set_ylabel('Y')
    cbar = plt.colorbar(ax4.images[0], ax=ax4)
    cbar.set_label('Vertex count')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig