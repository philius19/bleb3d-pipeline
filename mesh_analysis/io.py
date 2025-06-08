from mat73 import loadmat
from pathlib import Path
from typing import Tuple, List
import numpy as np
import json

"""
File input/output operations for mesh analysis.
"""

def validate_file_paths(surface_path: Path, curvature_path: Path, 
                       supported_formats: List[str]) -> None:
    """
    Validate input files exist and have correct format.
    
    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If format not supported
    """
    for path, name in [(surface_path, "Surface"), (curvature_path, "Curvature")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")
        
        if path.suffix not in supported_formats:
            raise ValueError(
                f"{name} file format {path.suffix} not supported. "
                f"Supported formats: {supported_formats}"
            )


def load_surface_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load surface mesh data from .mat file.
    
    Returns:
        Tuple of (vertices, faces)
    """
    surface_data = loadmat(str(filepath))
    surface = surface_data['surface']
    
    vertices = np.array(surface['vertices'], dtype=np.float32)
    faces = np.array(surface['faces'], dtype=np.int32) - 1  # Convert to 0-based
    
    return vertices, faces


def load_curvature_data(filepath: Path, expected_length: int) -> np.ndarray:
    """
    Load curvature data from .mat file.
    
    Args:
        filepath: Path to curvature file
        expected_length: Expected number of values (for validation)
        
    Returns:
        Flattened curvature array
        
    Raises:
        ValueError: If curvature length doesn't match expected
    """
    curv_data = loadmat(str(filepath))
    curvature = np.array(curv_data['meanCurvature']).flatten()
    
    if len(curvature) != expected_length:
        raise ValueError(
            f"Curvature length ({len(curvature)}) doesn't match "
            f"expected count ({expected_length})"
        )
    
    return curvature


def save_mesh_to_ply(mesh, filepath: Path) -> None:
    """Export mesh to PLY format."""
    mesh.export(str(filepath))


def save_results_to_json(results: dict, filepath: Path) -> None:
    """Save analysis results to JSON."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)