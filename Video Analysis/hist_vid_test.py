import cv2
import numpy as np
from matplotlib import pyplot as plt

color = ('b','g','r')

for i, col in enumerate(color):
	{i}
b_frame_hist
b_frame_hist = []
g_frame_hist = []
r_frame_hist = []



cap = cv2.VideoCapture(in_file)  # load the video
cap.set(2,frame_no) # first arguement of 2 indicated a zero-based index - and second indicates the frame_number that is the focus
counter = 0 # which frame
while (cap.isOpened()):  # play the video by reading frame by frame
    ret, frame = cap.read()
    if ret == True:
# this just calculates a frame# X 3 matrix of mean,std, and sum
        for i,col in enumerate(color):
        	histr = cv2.calcHist([img],[i],None,[256],[0,256])
        	str(colour[i])_frame_hist[frame,0] = np.mean(histr)
        	str(colour[i])_frame_hist[frame,1] = np.std(histr)
        	str(colour[i])_frame_hist[frame,2] = np.sum(histr)
        	plt.plot(histr,color = col)
        	plt.xlim([0,256])
        	plt.show()
            
        #this splits into 3 np arrays, one for each r,g,b channel
        split_into_rgb_channels(frame)
        frame1 = equalizeHistColor(frame)
        
#        # Add frame # to the appropriate structures - Trigger_Start Trigger_Stop are either 1 (green) or 2 (blue), Trigger state_state can also be 0 (neither green nor blue)
#        # List of [frame + start event trigger] (where the max[index] = corresponds with the last EEG event)
#        if Trigger_Start = []
#        # List of [frame + end event trigger]
#        if Trigger_Stop = []
#        # List of [frame + trigger state] (0 B + G channels below thresholds, 1 above B channel threshold, 2 above R channel threshold
#        Tigger_State[count] = 
#        else    
#            
#        writer.write(img)  # save the frame into video file
        
        if count % 10 == 0: # every 10th frame, show frame
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