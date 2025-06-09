from mesh_analysis import MeshAnalyzer
from mesh_analysis.visualization import plot_curvature_distribution, plot_vertice_distribution
import plotly.graph_objects as go
import numpy as np


surface_path3D = "/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Datensatz/3DPreprocessed/Results/Morphology/Analysis/Mesh/ch1/surface_1_1.mat"
surface_path2D = "/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Datensatz/02_2DPreprocessed/Results/Morphology/Analysis/Mesh/ch1/surface_1_1.mat"
curvature_path3D = "/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Datensatz/3DPreprocessed/Results/Morphology/Analysis/Mesh/ch1/meanCurvature_1_1.mat"
curvature_path2D = "/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Datensatz/02_2DPreprocessed/Results/Morphology/Analysis/Mesh/ch1/meanCurvature_1_1.mat"


analyzer3D = MeshAnalyzer(surface_path3D, curvature_path3D)
analyzer2D = MeshAnalyzer(surface_path2D, curvature_path2D)


analyzer3D.load_data()
analyzer2D.load_data()

stats3D = analyzer3D.calculate_statistics()
stats2D = analyzer2D.calculate_statistics()

curv3D = analyzer3D.curvature
curv2D = analyzer2D.curvature

mesh = analyzer2D.mesh

print(analyzer3D.physical_dimensions)





#plot_curvature_distribution(curv3D, "/Users/philippkaintoch/Desktop/3D.png")
#plot_curvature_distribution(curv2D, "/Users/philippkaintoch/Desktop/2D.png")

#plot_vertice_distribution(analyzer3D.vertices, "/Users/philippkaintoch/Desktop/3D_Pl.png")
#plot_vertice_distribution(analyzer2D.vertices, "/Users/philippkaintoch/Desktop/2D_Pl.png")