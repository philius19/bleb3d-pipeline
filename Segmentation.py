from pathlib import Path
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

###################### Loading Tif ######################

DATA_ROOT = Path("~/Documents/Projects/01_Bleb3D/Datensatz").expanduser()
control_Image_path = DATA_ROOT / "motif3DExampleDataFinal" / "testData" / "krasMV3" / "Cell1/1_CAM01_000000.tif"
image_path = DATA_ROOT / "01_B2_BAR_3D_mCherry_CAAX-CFP_decon.tif"

print(f" Test File exists = {control_Image_path.exists()}") 
print(f" File for analysis exists = {image_path.exists()}") 

control_Stack = tiff.imread(control_Image_path)
movie = tiff.imread(image_path)

###################### TIF-Data ######################

def check_img_data(tif_stack):
    print("Shape :", tif_stack.shape)
    print("dtype :", tif_stack.dtype)
    print("min / max :", tif_stack.min(), tif_stack.max())

check_img_data(control_Stack)
check_img_data(movie)

###################### Z-Stack Extraction ######################

stack = movie[1]

print(stack.shape)

###################### TIF-Preview ######################

def slider_view(stack, ch=0):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)
    ax.set_axis_off()

    # Building Slider 
    ax_z = plt.axes([0.15, 0.05, 0.7, 0.03]) # create "slider host"
    s_z = Slider(ax_z, "Z",   0, stack.shape[0]-1, valinit=0, valstep=1) # create slider object 

    def clim(frame):
        return np.percentile(frame, (1, 99))

    vmin, vmax = clim(stack[0, ch])
    im = ax.imshow(stack[0, ch], cmap="grey", vmin=vmin, vmax=vmax)
    
    # local neasted function ("callback") --> Update-Rule for the Slider 
    def update(val):
        z = int(s_z.val) 
        frame = stack[z, ch]

        vmin, vmax = clim(frame)
        im.set_data(stack[z, ch])
        im.set_clim(vmin, vmax)
        ax.set_title(f"Z = {z}   |   C = {ch}")
        fig.canvas.draw_idle()

    s_z.on_changed(update)
    plt.show()

#slider_view(control_Stack, ch=0)
slider_view(stack, ch=0)
