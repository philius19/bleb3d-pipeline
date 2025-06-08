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