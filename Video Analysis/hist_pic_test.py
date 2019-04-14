import cv2
import numpy as np
from matplotlib import pyplot as plt

frame = 1
colour = ("b","g","r")

for i, col in enumerate(color):
	globals()[str(colour[i]) + "_frame_hist"] = np.zeros((frame), dtype=[('x', 'float'), ('y', 'float'), ('z', 'float')])

img = cv2.imread('Trial.PNG')

    # this just calculates a frame# X 3 matrix of mean,std, and sum
for i,col in enumerate(color):
	histr = cv2.calcHist([img],[i],None,[256],[0,256])
	str(colour[i])_frame_hist[0] = np.mean(histr)
	str(colour[i])_frame_hist[1] = np.std(histr)
	str(colour[i])_frame_hist[2] = np.sum(histr)
	plt.plot(histr,color = col)
	plt.xlim([0,256])
	plt.show()
    
#this splits into 3 np arrays, one for each r,g,b channel
split_into_rgb_channels(img)
frame1 = equalizeHistColor(img)