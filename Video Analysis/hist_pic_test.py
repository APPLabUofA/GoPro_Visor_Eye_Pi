import cv2
import numpy as np
from matplotlib import pyplot as plt

frames = 1
colours = ['b','g','r']

def split_into_rgb_channels(image):
  red = image[:,:,2]
  green = image[:,:,1]
  blue = image[:,:,0]
  return red, green, blue

def equalizeHistColor(frame):
    # equalize the histogram of color image
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  # convert to HSV
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])  # equalize the histogram of the V channel
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)  # convert the HSV image back to RGB format

for i, col in enumerate(colours):
	globals()[str(colours[i]) + "_frame_hist"] = np.zeros((frames,3))



img = cv2.imread('Green_Flash_2.JPG')
fig, (ax1,ax2) = plt.subplots(2, sharex=True)
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
#while (cap.isOpened()):  # play the video by reading frame by frame

#equalizeHistColor(img)
    # this just calculates a frame# X 3 matrix of mean,std, and sum
for i,col in enumerate(colours):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    
    globals()[str(colours[i]) + "_frame_hist_data"] = histr
    globals()[str(colours[i]) + "_frame_hist"][(0,0)] = np.mean(histr[100:-1])
    globals()[str(colours[i]) + "_frame_hist"][(0,1)] = np.std(histr[100:-1])
    globals()[str(colours[i]) + "_frame_hist"][(0,2)] = np.sum(histr[100:-1])
    img2 = equalizeHistColor(img)
    histr_norm = cv2.calcHist([img2],[i],None,[256],[0,256]) 
    plt.plot(histr,color = col)
    ax1.plot(histr,color = col)
    ax2.plot(histr_norm, color = col)
    plt.xlim([0,256])
    plt.show()
    
    

    
#this splits into 3 np arrays, one for each r,g,b channel
#split_into_rgb_channels(img)
#frame1 = equalizeHistColor(img)