from pathlib import Path
import tifffile as tiff
import numpy as np

class TiffReader:
    def __init__(self, tiff_path: Path): # tiff_path: Path is a static check that tiff_path is a path
        
        self.tiff_path = tiff_path.expanduser()
        print(f"📂  File : {self.tiff_path.name}")

        with tiff.TiffFile(self.tiff_path) as tf:
            series = tf.series[0]
            self.axes = series.axes
            self.arr = series.asarray()
            self.ome = tf.ome_metadata

            imagej_metadata = tf.imagej_metadata
            self.imagej_metadata = imagej_metadata
            z = imagej_metadata["spacing"]
            self.z = z * 1000
            print(f"Pixel Size Z (nm) = {z}")

img = Path("~/Documents/Projects/01_Bleb3D/Datensatz/Preprocessed/ch1/C1-01_B2_BAR_3D_mCherry_CAAX-CFP_decon-1.tif")

reader = TiffReader(img)


print(reader.imagej_metadata)
