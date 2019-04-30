import board
#import neopixel
import pigpio
import RPi.GPIO as GPIO
import time
#from random import randint, shuffle
import numpy as np
#import pygame
from threading import Thread, Timer
import cv2
#from psychopy import visual, core, event



# dont know what the screen resolution of the monitor attached to the pi are?  run - "fbset -s"

#intialize colourspace
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
blank = (0, 0, 0)
white = (225,225,225)
brightness = 0.2


FPS =  24

trig_pins = [4,17,27,22,5,6,13,19]
resp_pin = 21

###initialise GPIO pins###
GPIO.setmode(GPIO.BCM)
GPIO.setup(trig_pins,GPIO.OUT)

###set triggers to 0###
GPIO.output(trig_pins,0)

###setup pin for push button###
GPIO.setup(resp_pin,GPIO.IN,pull_up_down=GPIO.PUD_UP)


instr_0 = 'Press the space bar once to continue'
instr_1 = 'During the duration of the experiment, focus on central fixation'
instr_2 = 'Press the button when you see blue flashes'
instr_3 = 'Do NOT press the spacebar when you see green flashes.'
instr_4 = 'When you are ready to start, press the space bar and THEN the button.'
instr_5 = ''
instr_6 = 'Feel free to take a break at this time. Press the button when you are ready to start.'
instr_7 = 'Congratulations, you have finished the experiment! Please contact the experimenter.'

screen_res = 1024, 1280

# Instruct lists for looping 
Start_Instruct = [instr_0, instr_1, instr_2, instr_3, instr_4, instr_5]
Break_Instruct = [instr_6]
End_Instruct = [instr_7]
# Positions of text
Start_Instruct_Width = [(int(screen_res[1]/3),int(screen_res[1]/3)),(int(screen_res[1]/5),int(screen_res[1]/3)),(int(screen_res[1]/3),int(screen_res[1]/3)),(int(screen_res[1]/4),int(screen_res[1]/3)),(int(screen_res[1]/6),int(screen_res[1]/3)),(int(screen_res[1]/4),int(screen_res[1]/3))]
Break_Instruct_Width = [(int(screen_res[1]/6),int(screen_res[1]/3))]
End_Instruct_Width =  [(int(screen_res[1]/6),int(screen_res[1]/3))]

def refresh_fixation(): 
    img[int(width-width/100):int(width+width/100), int(height-height/20):int(height+height/20), :] = 255
    img[int(width-width/20):int(width+width/20), int(height-height/100):int(height+height/100), :] = 255


width = int(screen_res[0]/2) # half width
height = int(screen_res[1]/2) # half height


#GPIO.wait_for_edge(resp_pin,GPIO.RISING)




if __name__ == '__main__':
    img = np.zeros((int(screen_res[0]),int(screen_res[1]),3), np.uint8) # 1960 X 1200 X 3 # all zeros = all black
    refresh_fixation()
    window_name = 'projector'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    for i in range(len(Start_Instruct)):
        img = np.zeros((int(screen_res[0]),int(screen_res[1]),3), np.uint8) # 1960 X 1200 X 3 # all zeros = all black
        cv2.putText(img,Start_Instruct[i],Start_Instruct_Width[i], cv2.FONT_HERSHEY_COMPLEX_SMALL , 1,(255,255,255),2,cv2.LINE_AA)
        refresh_fixation()
        cv2.imshow(window_name, img)
##        GPIO.wait_for_edge(resp_pin,GPIO.RISING)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
