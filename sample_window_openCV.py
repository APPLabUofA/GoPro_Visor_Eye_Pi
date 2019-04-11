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

instr_1 = 'Focus on central fixation.'
instr_2 = 'Press the button when you see blue flashes, do NOT press the spacebar when you see green flashes.'
instr_3 = 'Press the button when you are ready to start.'
instr_4 = 'Feel free to take a break at this time. Press the button when you are ready to start.'
instr_5 = 'Congratulations, you have finished the experiment! Please contact the experimenter.'
instruct = instr_1, instr_2, instr_3, instr_4, instr_5

if __name__ == '__main__':
    refresh_screen()
    img = np.zeros((screen_res[0],screen_res[1],3), np.uint8) # 1960 X 1200 X 3
    img[width-4:width+4, height-24:height+24, :] = 255
    img[width-50:width+50, height-2:height+2, :] = 255
    window_name = 'projector'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, img)
    for i in length(instruct)
        cv2.putText(img,instruct(i),(int(2*width/5),height), cv2.FONT_HERSHEY_SIMPLEX , 1,(255,255,255),2,cv2.LINE_AA)
        GPIO.wait_for_edge(resp_pin,GPIO.RISING)
