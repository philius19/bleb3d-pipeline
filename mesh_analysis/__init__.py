"""
Mesh analysis package for u-shape3D data processing.
"""

from .analyzer import MeshAnalyzer
from .visualization import plot_curvature_distribution, plot_vertice_distribution
from .io import load_surface_data, load_curvature_data
from .utils import convert_pixels_to_um

__version__ = "1.0.0"
__all__ = ['MeshAnalyzer', 'plot_curvature_distribution', "plot_vertice_distribution"]

