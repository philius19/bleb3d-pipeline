from pathlib import Path
import tifffile as tiff
import numpy as np
import shutil

def inspect_and_export(tif_path: Path):

    tif_path = tif_path.expanduser()
    print(f"📂  File : {tif_path.name}")

    with tiff.TiffFile(tif_path) as tf:
        series = tf.series[0]
        axes = series.axes
        arr = series.asarray()
        ome = tf.ome_metadata 

        print(f"Axes: {axes}")
        print(f"Shape: {arr.shape}")

        
        imagej_metadata = tf.imagej_metadata
        z = imagej_metadata["spacing"]
        z = z * 1000
        print(f"Pixel Size Z (nm) = {z}")
        print(imagej_metadata)



path = Path("~/Documents/Projects/01_Bleb3D/Datensatz/01_B2_BAR_3D_mCherry_CAAX-CFP_decon.tif")


inspect_and_export(path)