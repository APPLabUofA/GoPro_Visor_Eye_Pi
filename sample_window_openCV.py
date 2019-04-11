#import board
#import neopixel
#import pigpio
#import RPi.GPIO as GPIO
import time
#from random import randint, shuffle
import numpy as np
#import pygame
#from threading import Thread
import cv2
import screeninfo
#from psychopy import visual, core, event

#intialize colourspace
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
blank = (0, 0, 0)
white = (225,225,225)
brightness = 0.2

cv_red = (0, 0, 255)

screen_res = 1960, 1200
width = int(screen_res[0]/2) # half width
height = int(screen_res[1]/2) # half height

instructions1 = 'Focus on central fixation.'
instructions2 = 'Press the button when you see blue flashes, do NOT press the spacebar when you see green flashes.'
instructions3 = 'Press the button when you are ready to start.'
break_screen = 'Feel free to take a break at this time. Press the button when you are ready to start.'
end_screen = 'Congratulations, you have finished the experiment! Please contact the experimenter.'





if __name__ == '__main__':
#    img = cv2.imread('C:/Users/User/Desktop/Events.png')
#    img.shape
    img = np.zeros((screen_res[0],screen_res[1],3), np.uint8) # 1960 X 1200 X 3
    img[width-4:width+4, height-24:height+24, :] = 255
    img[width-50:width+50, height-2:height+2, :] = 255
    
    cv2.putText(img,instructions1,(int(2*width/5),height), cv2.FONT_HERSHEY_SIMPLEX , 1,(255,255,255),2,cv2.LINE_AA)
    
    window_name = 'projector'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, img)
    k = cv2.waitKey()
    if k == 27:
        cv2.destroyAllWindows()

#if __name__ == '__main__':
#    img = cv2.imread('C:/Users/User/Desktop/Events.png')
#    img.shape
##    img = np.zeros((screen_res[0],screen_res[1],3), np.uint8)
#    cv2.line(img,(screen_res[0]-10,screen_res[1]),(screen_res[0]+10,screen_res[1]),(white),3) # horizontal line
#    cv2.line(img,(screen_res[0],screen_res[1]-10),(screen_res[0],screen_res[1]+10),(white),3) # vertical line
#    window_name = 'projector'
#    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#    cv2.imshow(window_name, img)
#    k = cv2.waitKey()
#    if k == 27:
#        cv2.destroyAllWindows()


time.sleep(5)


# throw the whole thing in a loop so the experiment can be stopped by the esc or q keys

##event.waitKeys(keyList="space") # space bar to quit the window
##mywin.mouseVisible = True
##
### Cleanup
##mywin.close()
##core.quit()

