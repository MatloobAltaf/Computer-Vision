# File to get coordinates of elements on click 
# Run as python coordinates.py > points.txt 

import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2

file_name = cv2.imread('rubiks_cube.jpeg') #Change the filename
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(file_name)
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print(ix, iy)
    global coords
    coords = [ix, iy]
    return coords
for i in range(0,1):
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()