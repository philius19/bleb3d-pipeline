import os
from pathlib import Path
import warnings
from typing import Optional, Tuple, Dict, List, Union

import numpy as np
import matplotlib.pyplot as plt
import trimesh
from scipy import stats
from mat73 import loadmat


class MeshAnalyzer:
    """
    A class to analyze u-shape3D mesh data including surface geometry and curvature.
    
    This class provides methods to:
    - Load mesh and curvature data
    - Calculate mesh statistics
    - Visualize results
    - Export analysis results
    
    Attributes:
        surface_path (str): Path to surface .mat file
        curvature_path (str): Path to curvature .mat file
        mesh (trimesh.Trimesh): The loaded mesh object
        curvature (np.ndarray): Curvature values at vertices
        _processed (bool): Flag indicating if data has been processed
    
    Example:
        >>> analyzer = MeshAnalyzer('surface.mat', 'curvature.mat')
        >>> analyzer.load_data()
        >>> stats = analyzer.calculate_statistics()
    """

    # ========== CLASS VARIABLES (shared by all instances) ==========
    VERSION = "1.0.0"
    SUPPORTED_FORMATS = ['.mat', '.h5']

    DEFAULT_PIXEL_SIZE_XY = 0.1  # micrometers
    DEFAULT_PIXEL_SIZE_Z = 0.2   # micrometers

    def __init__(self, surface_path:str, curvature_path: str,
                 pixel_size_xy: float = None, pixel_size_z: float = None):
        
        """
        Initialize the MeshAnalyzer.
        
        Parameters:
            surface_path: Path to surface .mat file
            curvature_path: Path to curvature .mat file  
            pixel_size_xy: XY pixel size in micrometers (default: 0.1)
            pixel_size_z: Z pixel size in micrometers (default: 0.2)
        
        Raises:
            FileNotFoundError: If input files don't exist
            ValueError: If file format is not supported
        """

        # Store paths as Path objects for better path handling
        self.surface_path = Path(surface_path)
        self.curvature_path = Path(curvature_path)

        # Validate inputs
        self._validate_inputs()

        # Set pixel sizes
        self.pixel_size_xy = pixel_size_xy or self.DEFAULT_PIXEL_SIZE_XY
        self.pixel_size_z = pixel_size_z or self.DEFAULT_PIXEL_SIZE_Z

         # Initialize data containers (None until loaded) - Optional is a type hint (array or None)
        self.vertices: Optional[np.ndarray] = None
        self.faces: Optional[np.ndarray] = None
        self.mesh: Optional[trimesh.Trimesh] = None
        self.curvature: Optional[np.ndarray] = None

        # Private attributes (convention: prefix with _)
        self._processed = False           
        self._statistics_cache = {}

        # Analysis results storage
        self.results = {
            'statistics': {},
            'visualizations': {},
            'quality_metrics': {}
        }

        # ========== PRIVATE METHODS (internal use) ==========
    def _validate_inputs(self):
        """Validate input files exist and have correct format."""
        for path, name in [(self.surface_path, "Surface"),
                            (self.curvature_path, "Curvature")]:
            if not path.exists():
                raise FileNotFoundError(f"{name} file not found: {path}")
            
            if path.suffix not in self.SUPPORTED_FORMATS:
                raise ValueError(
                    f"{name} file format {path.suffix} not supported. "
                    f"Supported formats: {self.SUPPORTED_FORMATS}"
                    )
                
        # ========== PUBLIC METHODS ==========
    def load_data(self, verbose: bool = True) -> None:
        """
        Load mesh and curvature data from files.
        
        Parameters:
            verbose: Print loading progress
        
        Raises:
            RuntimeError: If data loading fails
        """

        if verbose:
            print(f"Loading surface from: {self.surface_path}")

        try:
            # Load surface data
            surface_data = loadmat(str(self.surface_path))
            surface = surface_data['surface']

            # Extract vertices and faces
            self.vertices = np.array(surface['vertices'], dtype=np.float32)
            self.faces = np.array(surface['faces'], dtype=np.int32) - 1

            # Create trimesh object
            self.mesh = trimesh.Trimesh(
                vertices = self.vertices,
                faces = self.faces,
                process = True
            )

            # Fix mesh orientation if needed
            if self.mesh.volume < 0:
                if verbose:
                    print("Fixing inverted mesh...")
                self.mesh.invert()

            # Load curvature
            if verbose:
                print(f"Loading curvature from: {self.curvature_path}")

            curv_data = loadmat(str(self.curvature_path))
            self.curvature = np.array(curv_data['meanCurvature']).flatten()

            if len(self.faces) != len(self.curvature):
                raise ValueError(
                    f"Curvature length ({len(self.curvature)}) doesn't match "
                    f"vertex count ({len(self.faces)})"
                )
            
            self._processed = True

            if verbose:
                print(f"✓ Loaded {len(self.vertices):,} vertices, {len(self.faces):,} faces")
                print(f"✓ Mesh volume: {self.mesh.volume:.2f} pixels³")

        except Exception as e:
            raise RuntimeError(f"Failed to load data: {str(e)}")

    def calculate_statistics(self, force_recalculate: bool = False) -> Dict:
        """
        Calculate comprehensive mesh and curvature statistics.
        
        Parameters:
            force_recalculate: Recalculate even if cached
            
        Returns:
            Dictionary containing all statistics
        """
        if not self._processed:
            raise RuntimeError("Must load data first. Call load_data()")
        
        # Check cache - avoid expensive recalculations
        if 'basic_stats' in self._statistics_cache and not force_recalculate:
            return self._statistics_cache['basic_stats']
        
        stats = {
            'mesh': {
                'n_vertices': len(self.vertices),
                'n_faces': len(self.faces),
                'n_edges': len(self.mesh.edges_unique),
                'volume_pixels3': float(self.mesh.volume),
                'volume_um3': float(self.mesh.volume * 
                                   self.pixel_size_xy**2 * self.pixel_size_z),
                'surface_area_pixels2': float(self.mesh.area),
                'surface_area_um2': float(self.mesh.area * self.pixel_size_xy**2),
                'is_watertight': self.mesh.is_watertight,
                'euler_number': self.mesh.euler_number
            },
            'curvature': {
                'mean': float(np.mean(self.curvature)),
                'std': float(np.std(self.curvature)),
                'min': float(np.min(self.curvature)),
                'max': float(np.max(self.curvature)),
                'percentiles': {
                    p: float(np.percentile(self.curvature, p))
                    for p in [1, 5, 25, 50, 75, 95, 99]
                }
            },
            'quality': self._calculate_mesh_quality()
        }

        # Cache results
        self._statistics_cache['basic_stats'] = stats
        self.results['statistics'] = stats

        return stats
    
    def _calculate_mesh_quality(self) -> Dict:
        """Calculate mesh quality metrics."""

        # Triangle areas
        areas = self.mesh.area_faces

        # Edge lengths
        edge_lengths = self.mesh.edges_unique_length

        return {
                    'triangle_area': {
                        'mean': float(np.mean(areas)),
                        'std': float(np.std(areas)),
                        'min': float(np.min(areas)),
                        'max': float(np.max(areas))
                    },
                    'edge_length': {
                        'mean': float(np.mean(edge_lengths)),
                        'std': float(np.std(edge_lengths)),
                        'min': float(np.min(edge_lengths)),
                        'max': float(np.max(edge_lengths))
                    },
                    'aspect_ratio': float(np.max(edge_lengths) / np.min(edge_lengths))
                }
    
    # ========== VISUALIZATION METHODS ==========
    def plot_curvature_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot curvature distribution.
        
        Parameters:
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        if not self._processed:
            raise RuntimeError("Must load data first")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Linear scale
        ax1.hist(self.curvature, bins=100, alpha=0.7)
        ax1.axvline(0, color='red', linestyle='--', label='Zero')
        ax1.set_xlabel('Curvature (1/pixels)')
        ax1.set_ylabel('Count')
        ax1.set_title('Curvature Distribution')
        ax1.legend()

        # Log scale

        ax2.hist(self.curvature[self.curvature != 0], bins=100, alpha=0.7)
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

    # ========== PROPERTY METHODS (computed attributes) ==========
    @property
    def is_loaded(self) -> bool:
        """Check if data has been loaded."""
        return self._processed

    @property
    def physical_dimensions(self) -> Dict[str, float]:
        """Get physical dimensions in micrometers."""
        if not self._processed:
            return {}
    
        bounds = self.mesh.bounds
        return {
            'x_um': (bounds[1][0] - bounds[0][0]) * self.pixel_size_xy,
            'y_um': (bounds[1][1] - bounds[0][1]) * self.pixel_size_xy,
            'z_um': (bounds[1][2] - bounds[0][2]) * self.pixel_size_z
        }
    
    # ========== CLASS METHODS (don't need instance) ==========
    @classmethod
    def from_config(cls, config_path: str) -> 'MeshAnalyzer':
        """
        Create analyzer from configuration file.
        
        Parameters:
            config_path: Path to configuration JSON/YAML
            
        Returns:
            MeshAnalyzer instance
        """
        # Load config and create instance
        # This is an alternative constructor
        pass

        # ========== STATIC METHODS (utility functions) ==========
    @staticmethod
    def convert_pixels_to_um(value: float, pixel_size: float) -> float:
        """Convert pixel units to micrometers."""
        return value * pixel_size
    
    # ========== MAGIC METHODS (special behavior) ==========
    def __str__(self) -> str:
        """String representation for print()."""
        if not self._processed:
            return f"MeshAnalyzer(not loaded)"
        return (f"MeshAnalyzer({len(self.vertices)} vertices, "
                f"{len(self.faces)} faces)")
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (f"MeshAnalyzer(surface='{self.surface_path.name}', "
                f"curvature='{self.curvature_path.name}')")
    
    # ========== CONTEXT MANAGER (with statement) ==========
    def __enter__(self):
        """Support 'with' statement."""
        self.load_data()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup when exiting 'with' block."""
        # Could clear memory, close files, etc.
        pass

        


from Old.Mesh_Analysis import MeshAnalyzer

DATA_ROOT = Path("~/Documents/Projects/01_Bleb3D/Datensatz/Preprocessed/Output_u_shape3D_test7/Morphology/Analysis/Mesh/ch1").expanduser()
surface_data_path = DATA_ROOT / "surface_1_1.mat"
curvature_data_path = DATA_ROOT / "meanCurvature_1_1.mat"

analyzer = MeshAnalyzer(surface_data_path, curvature_data_path)

print(analyzer.load_data())
stats = analyzer.calculate_statistics()
print({stats['mesh']['n_faces']})