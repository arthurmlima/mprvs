import cv2
from glob import glob
import os
import numpy as np
import time
import projection
import common
import pca
import hough
import tmd
import radon
import mpcall
from natsort import natsorted
import pandas as pd

#from demeter import sptools
#cd ../mpriscv/rv ; make program ; cd .. ; gcc -fPIC -shared smp.c -o mpriscv.so ; cd ../core ; sudo python3 main.py
# ****************    SET PATHS      *******************************************************

load_path = '../mpriscv/images/output_images/'
save_path = '../mpriscv/images/output_images/'
sinogram_path = '../mpriscv/images/output_images/'

# ****************    SET IMAGE NAMES      *************************************************
img_names = glob(os.path.join('../images/c*.hpp'))  # article for discipline


# *************   PUBLIC VARIABLES  ***************************************************
i = 0
list_images = []
list_angles = []
time_process_imgs = []
print_on_terminal = False

iname = []
t0 = []
t1 = []
t2 = []
t3 = []
t4 = []
t5 = []
tc = []

# *************   HEURISTIC BASIC SETUP  **********************************************
algorithm = "RADON"    # Pick up one of heuristic available [PROJECTION, MIDDLE, HOUGH, PCA, RADON]
scan_resolution = 1    # Factor for angle step image rotation in degrees [1/scan_resolution]°
image_rezise    = 1    # image downsize factor [original/image_rezise]
ig_index = 0            # image index
# *************   START PROCESSING ***********************************************


print("================  START PROCESSING  ================")
for fn in range(1, 10):
    print(fn)
    #base = os.path.basename(fn)
    #file_name = (os.path.splitext(base)[0])
    st0 = 0
    st1 = 0
    st2 = 0
    st3 = 0
    st4 = 0
    st5 = 0
    stc = 0
    start_time = time.time()
    img ,st0, st1, st2, st3, st4, st5  = mpcall.mpriscv_mean5x5(fn)
    cv2.imwrite('mean5x5/%02i.png' % fn, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    #iname.append(file_name)
    t0.append(str(st0.value))    
    t1.append(int(str(st1.value))-int(str(st0.value)))
    t2.append(int(str(st2.value))-int(str(st0.value)))
    t3.append(int(str(st3.value))-int(str(st0.value)))
    t4.append(int(str(st4.value))-int(str(st0.value)))
    t5.append(int(str(st5.value))-int(str(st0.value)))


#mean_time = sum(time_process_imgs) / len(time_process_imgs)
print("================  FINISH PROCESSING  ================")

# *************   END OF MAIN APPLICATION  ***********************************************************
# ****************************************************************************************************

data = {'t0': t0, 't1': t1, 't2': t2, 't3': t3, 't4': t4, 't5': t5}
df = pd.DataFrame(data)
df.to_excel('mean5x5.xlsx', index=False)

