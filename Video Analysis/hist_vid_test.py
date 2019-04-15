import cv2
import numpy as np
from matplotlib import pyplot as plt

frames = 1
colours = ['b','g','r']

# First Pass
# List of [frame + start event trigger] (where the max[index] = corresponds with the last EEG event)
Trigger_Start = []
# List of [frame + end event trigger]
Trigger_Stop = []
# List of [frame + trigger state] (0 B + G channels below thresholds, 1 above B channel threshold, 2 above R channel threshold
Tigger_State = []

# Want a frame of each start trigger saved to a folder

# Second Pass for extracting epochs based off first pass - figure out later
# Eventually will output an ~[-1,1] video epoch to be the raw input for deep learning
Trigger_Epoch = []


webcam = 0 # set to 1 if input is a webcam
exp = 1
path = 'M:\\Data\\GoPro_P3_...Latency\\Videos\\'
part = '011' # Version - example '001' or '054'
exp = '_camera_p3' # ex. '003_camera_p3'
in_format = '.avi'
in_file = part + exp + in_format

if webcam == 1:
    in_file = ''

# %% # Are we saving an output file (file with overlaid filters/bounders/manipulations)?

 # # Version - example '001' or '054'
out_format = '.avi'
out_file = part + exp + out_format

# Output file parameter
imgSize=(640,480) # likely best to set to original  dimensions
frame_per_second=30.0
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
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])  # equalize the histogram of the V channel
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)  # convert the HSV image back to RGB format

# %% Loop through frames
cap = cv2.VideoCapture(in_file)  # load the video
post_frame = cap.get(1) # CV_CAP_PROP_POS_FRAMES #0-based index of the frame to be decoded/captured next
count = 0 # which frame
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for i, col in enumerate(colours):
	globals()[str(colours[i]) + "_frame_hist"] = np.zeros((frames,length))

while (cap.isOpened()):  # play the video by reading frame by frame
    ret, frame = cap.read()
    if ret == True:
# this just calculates a frame# X 3 matrix of mean,std, and sum
        for i,col in enumerate(colours):
            histr = cv2.calcHist([frame],[i],None,[256],[0,256])
            globals()[str(colours[i]) + "_frame_hist"][(0,0)] = np.mean(histr)
            globals()[str(colours[i]) + "_frame_hist"][(0,1)] = np.std(histr)
            globals()[str(colours[i]) + "_frame_hist"][(0,2)] = np.sum(histr)
            plt.plot(histr,color = col)
            plt.xlim([0,256])
            plt.show()
            
        #this splits into 3 np arrays, one for each r,g,b channel
#        split_into_rgb_channels(frame)
#        frame1 = equalizeHistColor(frame)
        
        # Add frame # to the appropriate structures - Trigger_Start Trigger_Stop are either 1 (green) or 2 (blue), Trigger state_state can also be 0 (neither green nor blue)
        # List of [frame + start event trigger] (where the max[index] = corresponds with the last EEG event)
        if count > start_flash:
            if b_frame_hist[(count)]
            Trigger_Start = []:
                
            # List of [frame + end event trigger]
            if Trigger_Stop = []
            # List of [frame + trigger state] (0 B + G channels below thresholds, 1 above B channel threshold, 2 above R channel threshold
            Tigger_State[count] = 
            else    
#            
#        writer.write(img)  # save the frame into video file
        
        if count % 500 == 0: # every 10th frame, show frame
            cv2.imshow('Original', frame)  # show the original frame
#            cv2.imshow('New', img) #show the new frame

        count += 1
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break
    else:
        print("ref != True")
        break
    # When everything done, release the capture
writer.release()
cap.release()
cv2.destroyAllWindows()