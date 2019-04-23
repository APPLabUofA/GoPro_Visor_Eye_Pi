import numpy as np
import cv2
import time 

color=(255,0,0)
thickness=2
colours = ['b','g','r']
kernel = np.ones((5,5),np.uint8)
thresh = 20 # range that binary masks are derived from
thresh2 = 40
thresh3 = 80
change = 2
col_dict = {'b_hsv':np.array((0,255,255),dtype=np.uint8),'g_hsv':np.array((60,255,255),dtype=np.uint8),'r_hsv':np.array((120,255,255),dtype=np.uint8),'b_bgr':[255,0,0],'g_bgr':[0,255,0],'r_bgr':[0,0,255]} #HSV values

#for i, col in enumerate(colours): # direive the correct format from the bgr values in the col_dict
#    col_dict[str(col) + '_hsv2'] = cv2.cvtColor(np.uint8([col_dict[str(col) + '_bgr']] ),cv2.COLOR_BGR2HSV)[0][0]

for i, col in enumerate(colours): # broad values
    col_dict[str(col) + '_low_thresh_bro'] = np.array([col_dict[str(col) + '_hsv'][0] - thresh, 50, 50])
    col_dict[str(col) + '_high_thresh_bro'] = np.array([col_dict[str(col) + '_hsv'][0] + thresh, 255, 255])
#
for i, col in enumerate(colours): # specific values
    col_dict[str(col) + '_low_thresh_spe'] = np.array((col_dict[str(col) + '_hsv'][0] - thresh2,col_dict[str(col) + '_hsv'][1] - thresh2*3,col_dict[str(col) + '_hsv'][2] - thresh2*3))
    col_dict[str(col) + '_high_thresh_spe'] = np.array((col_dict[str(col) + '_hsv'][0] + thresh2, col_dict[str(col) + '_hsv'][1] + thresh2, col_dict[str(col) + '_hsv'][2] + thresh2))

#cap = cv2.VideoCapture(0)
#while(True):
#    # Capture two frames
#    ret, frame1 = cap.read()  # first image
#    time.sleep(1/25)          # slight delay
#    ret, frame2 = cap.read()  # second image 
#    img1 = cv2.absdiff(frame1,frame2)  # image difference
#    
#    # get theshold image
#    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#    blur = cv2.GaussianBlur(gray,(21,21),0)
#    ret,thresh = cv2.threshold(blur,200,255,cv2.THRESH_OTSU)
#    
#    # combine frame and the image difference
#    img2 = cv2.addWeighted(frame1,0.9,img1,0.1,0)
#    
#    # get contours and set bounding box from contours
#    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#    if len(contours) != 0:
#        for c in contours:
#            rect = cv2.boundingRect(c)
#            height, width = img2.shape[:2]            
#            if rect[2] > 0.2*height and rect[2] < 0.7*height and rect[3] > 0.2*width and rect[3] < 0.7*width: 
#                x,y,w,h = cv2.boundingRect(c)            # get bounding box of largest contour
#                img4=cv2.drawContours(img2, c, -1, color, thickness)
#                img5 = cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),2)  # draw red bounding box in img
#            else:
#                img5=img2
#    else:
#        img5=img2
frame = cv2.imread('Green_Flash_2.JPG')
cv2.imshow('',frame)
hsv_im = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
col = 'g'
#bgr = [154,230,58]
#thresh = 40
#hsv = col_dict[str(col) + '_hsv']
#hsv = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2HSV)[0][0]
#minHSV = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
#maxHSV = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])
minHSV = col_dict[str(col) + '_low_thresh_spe']
maxHSV = col_dict[str(col) + '_high_thresh_spe']
maskHSV = cv2.inRange(hsv_im, col_dict[str(col) + '_low_thresh_spe'], col_dict[str(col) + '_high_thresh_spe'])
resultHSV = cv2.bitwise_and(hsv_im, hsv_im, mask = maskHSV)
cv2.imshow("Result HSV", resultHSV)
 
imgray = cv2.cvtColor(resultHSV, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img2 = cv2.addWeighted(frame,0.9,resultHSV,0.1,0)
cv2.imshow('addweight',img2)
imgray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imshow('grey',imgray)
dilation = cv2.dilate(imgray,kernel,iterations = 7)
cv2.imshow('dilation7',dilation)
ret, thresh = cv2.threshold(dilation, 100, 255, 0)
cv2.imshow('threshold',thresh)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) != 0:
    print('true')
    for c in contours:
        rect = cv2.boundingRect(c)
        height, width = img2.shape[:2]            
#        if rect[2] > 0.2*height and rect[2] < 0.7*height and rect[3] > 0.2*width and rect[3] < 0.7*width: 
        x,y,w,h = cv2.boundingRect(c)            # get bounding box of largest contour
        img4=cv2.drawContours(img2, c, -1, color, thickness)
        img5 = cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),2)  # draw red bounding box in img
else:
    print('false')
text = 'green'
cv2.putText(img5, text, (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA) 
cv2.imshow('contours',img5)

cap = cv2.VideoCapture(0)

ret, last_frame = cap.read()

if last_frame is None:
    exit()

while(cap.isOpened()):
    ret, frame = cap.read()

    if frame is None:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Threshold the HSV image to get only 'change' color band
    mask = cv2.inRange(gray,col_dict[colours[change] + '_low_thresh'],col_dict[colours[change] + '_high_thresh'])
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
        

    a = cv2.absdiff(last_frame, frame)

    cv2.imshow('frame', frame)
    cv2.imshow('a', a)

    if cv2.waitKey(33) >= 0:
        break

    last_frame = frame

cap.release()
cv2.destroyAllWindows()




#        #second version with mask erosion and dilation
#        mask2=mask
#        res2 = cv2.bitwise_and(frame,frame, mask= mask2)
#        img_erosion = cv2.erode(mask2, kernel, iterations=1)
#        img_dilation = cv2.dilate(mask2, kernel, iterations=1)
#    
#        cv2.imshow('mask',mask)
#        cv2.imshow('mask2',mask2)
#        cv2.imshow('res',res) # plots mask and original image
#        cv2.imshow('res',res2) # plots eroded & dilated mask on the original image
#        k = cv2.waitKey(5) & 0xFF
#        if k == 27:
#            break
        
#         
#        ret, frame1 = cap.read()  # first image
#        ret, frame2 = cap.read()  # second image 
#        img1 = cv2.absdiff(frame1,frame2)  # image difference
#
#     
#        gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#        blur = cv2.GaussianBlur(gray,(21,21),0)
#        ret,thresh = cv2.threshold(blur,200,255,cv2.THRESH_OTSU)
#        
#        # combine frame and the image difference
#        img2 = cv2.addWeighted(frame1,0.9,img1,0.1,0)
#        
        # get contours and set bounding box from contours
