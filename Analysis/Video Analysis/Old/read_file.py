# %% Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker


# %% Initalize Variables + Lists

frames = 1
colours = ['b','g','r']

# First Pass
Trigger_Start = [] # List of [frame + start event trigger] (where the max[index] = corresponds with the last EEG event)
Trigger_Stop = [] # List of [frame + end event trigger]
Tigger_State = [] # List of [frame + trigger state] (0 B + G channels below thresholds, 1 above B channel threshold, 2 above R channel threshold
# Second Pass for extracting epochs based off first pass - figure out later
Trigger_Epoch = [] # Eventually will output an ~[-1,1] video epoch to be the raw input for deep learning

for i, col in enumerate(colours):
	globals()[str(colours[i]) + "_frame_hist"] = np.zeros((1,3))

for i, col in enumerate(colours):
	globals()[str(colours[i]) + "_frame_hist_values"] = np.zeros((1,3))

for i, col in enumerate(colours):
	globals()[str(colours[i]) + "_norm_frame_hist"] = np.zeros((1,3))

for i, col in enumerate(colours):
	globals()[str(colours[i]) + "_norm_frame_hist_values"] = np.zeros((1,3))



webcam = 0 # set to 1 if input is a webcam
exp = 1
path = 'M:\\Data\\GoPro_Visor\\GoPro_Video\\Pilot_1\\GOPRO0212' # M:\Data\GoPro_Visor\GoPro_Video\Pilot_1\GOPR0212
part = '011' # Version - example '001' or '054'
exp = '_camera_p3' # ex. '003_camera_p3'
in_format = '.MP4'
in_file = path + part + exp + in_format

if webcam == 1:
    in_file = 0

# %% # Are we saving an output file (file with overlaid filters/bounders/manipulations)?
# Version - example '001' or '054'
out_format = '.avi'
out_file = part + exp + out_format

# Output file parameter
imgSize=(640,480) # likely best to set to original  dimensions
frame_per_second=240.0
writer = cv2.VideoWriter(out_format, cv2.VideoWriter_fourcc(*"MJPG"), frame_per_second,imgSize,False)

# %% Load in participant specific info - frame number of events
if exp == 1: # original exp (2018)
    start_eeg = [0,0,652,3041,3330,567,1045,2053,616,1443,638]
    door_closed = [0,0,4800,6947,12240,5040,7440,7680,6000,8640,8400]
    start_flash = [0,0,8897,12159,17668,10446,12567,12673,11040,13343,13176]
elif exp == 2: # new data (2019)
    pass

# %% Functions

def split_into_rgb_channels(image):
  red = image[:,:,2]
  green = image[:,:,1]
  blue = image[:,:,0]
  return red, green, blue

def equalizeHistColor(frame):
    # equalize the histogram of color image
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  # convert to HSV
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])     # equalize the histogram of the V channel
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)   # convert the HSV image back to RGB format

def setup(ax):
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(which='major', width=1.00, length=5)
    ax.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.patch.set_alpha(0.0)
# %% Loop through frames
cap = cv2.VideoCapture(in_file)  # load the video
post_frame = cap.get(1) # CV_CAP_PROP_POS_FRAMES #0-based index of the frame to be decoded/captured next
#count = 0 # which frame
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, 637)
frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)

plt.ion()
fig, (ax1,ax2) = plt.subplots(2, sharex=True)
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
title = ax1.set_title("My plot", fontsize='large')


plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

#while (cap.isOpened()):  # play the video by reading frame by frame
while(True):
    ret, frame = cap.read()
    ax1.clear()
    ax2.clear()
# if ret == True
# this just calculates a frame# X 3 matrix of mean,std, and sum
    for i,col in enumerate(colours):
        histr = cv2.calcHist([frame],[i],None,[256],[0,256])  
        hist_values = [np.mean(histr[100:-1]),np.std(histr[100:-1]),np.sum(histr[100:-1])]
        globals()[str(colours[i]) + "_frame_hist_values"]= np.append(globals()[str(colours[i]) + "_frame_hist_values"],hist_values)
        globals()[str(colours[i]) + "_frame_hist"] = np.append((globals()[str(colours[i]) + "_frame_hist"]),histr)
        
        #####The same for the equalized histogram
        img = equalizeHistColor(frame)
        histr_norm = cv2.calcHist([img],[i],None,[256],[0,256])  
        hist_norm_values = [np.mean(histr_norm[100:-1]),np.std(histr_norm[100:-1]),np.sum(histr_norm[100:-1])]
        globals()[str(colours[i]) + "norm_frame_hist_values"]= np.append(globals()[str(colours[i]) + "_frame_hist_values"],hist_norm_values)
        globals()[str(colours[i]) + "norm_frame_hist"] = np.append((globals()[str(colours[i]) + "_frame_hist"]),histr_norm)
        ax1.plot(histr[100:256],color = col)
        ax2.plot(histr_norm[100:256], color = col)

    plt.draw()
        
    cv2.imshow('Original', frame) #show the new frame
    cv2.imshow('Equalized Histogram',img)
    #count += 1
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break

    # When everything done, release the capture
writer.release()
cap.release()
cv2.destroyAllWindows()







# Create a VideoCapture object and read from input file 
cap = cv2.VideoCapture('C:\\Users\\Public\\Videos\\Sample Video\\Wildlife.wmv')
cap = cv2.VideoCapture('M:\\Data\\GoPro_Visor\\Converted_Video\\003_setup.avi')
cap = cv2.VideoCapture('M:\\Data\\GoPro_Visor\\Experiment_1\\Pilot_1\\GoPro_Videos\\Converted\\GOPR0212.avi')
   
# Check if camera opened successfully 
if (cap.isOpened()== False):  
  print("Error opening video  file") 
   
# Read until video is completed 
while(cap.isOpened()): 
      
  # Capture frame-by-frame 
  ret, frame = cap.read() 
  if ret == True: 
   
    # Display the resulting frame 
    cv2.imshow('Frame', frame) 
   
    # Press Q on keyboard to  exit 
    if cv2.waitKey(25) & 0xFF == ord('q'): 
      break
   
  # Break the loop 
  else:  
    break
   
# When everything done, release  
# the video capture object 
cap.release() 
   
# Closes all the frames 
cv2.destroyAllWindows() 

