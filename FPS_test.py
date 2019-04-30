# -*- coding: utf-8 -*-

import cv2
from threading import Thread, Timer
import time
import pigpio
import RPi.GPIO as GPIO
import numpy as np

screen_res = 100,200

frame_state = [] # array of frame state

partnum = input("partnum: ")
filename = 'visual_p3_video'


trig_pins = [4,17,27,22,5,6,13,19]
resp_pin = 21

##amount of time needed to reset triggers##
trig_gap = 0.005

##define the ip address for the second Pi we will be controlling##
##pi = pigpio.pi('192.168.1.216')

###initialise GPIO pins###
GPIO.setmode(GPIO.BCM)
GPIO.setup(trig_pins,GPIO.OUT)

###set triggers to 0###
GPIO.output(trig_pins,0)

def pi2trig(trig_num): # trig_num = integer # sphinx - auto documentation

    pi_pins = [4,17,27,22,5,6,13,19]

    bin_num = list(reversed(bin(trig_num)[2:])) # taking all but the first 2 (removes '0b') and reverses and lists (1s and 0s)

    while len(bin_num) < len(pi_pins): # padding with 0s if shorter than 8 elements in the list
        bin_num.insert(len(bin_num)+1,str(0))

    trig_pins = []

    trig_pos = 0

    for i_trig in range(len(pi_pins)): # binary needs to be reversed (right to left)
        if bin_num[i_trig] == '1':
            trig_pins.insert(trig_pos,pi_pins[i_trig])
            trig_pos = trig_pos + 1

    return trig_pins # outputs a list of pi_pins, that are to be called

def append_frame_state(lst, index, state):
    while(index - len(lst) > 1):
        lst.append(None)
    lst.append(state)

def refresh_fixation(): 
    img[int(width-width/100):int(width+width/100), int(height-height/20):int(height+height/20), :] = 255
    img[int(width-width/20):int(width+width/20), int(height-height/100):int(height+height/100), :] = 255


class Stream (Thread): # construct - not an object - creat object and call Thread (spinning + running thread)
    def __init__(self):
        Thread.__init__(self)
        self.frame = 0 # if I want to pass this attrivute I have to pass the entire class to the other object/class
        self.time = 0
        self.file = '/home/pi/GitHub/GoPro_Visor_Eye_Pi/003_setup.avi' ### Change video input - should be in string format
        self.frame_time = 0 # time since the begining of the current frame being draw

        cap = cv2.VideoCapture(self.file)
##        self.frame_rate = cv2.get(cv2.CAP_PROP_FPS)
        self.frame_rate = 25  # can enforce the frame rate - check with 'ffprobe'
        self.frame_latency = 1/self.frame_rate # duration of a single frame
        self.frame_update_time = 0.01
        self.frame_update = 0

        self.trig = Frame_Trigger(self)
        self.trig.start()
        self.start_time = time.time()

    def run(self):
        cap = cv2.VideoCapture(self.file)
        self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret = True
        while ret == True:
#            timer = Timer(self.frame_latency, lambda: raise EndFrameFlag) # start time that will wait one frame duration - then throws error
#            timer.start()

            self.frame_start = time.time() - self.start_time 
            if self.frame_start < (self.frame+1)*self.frame_latency:
                # Capture frame-by-frame
                ret, frame = cap.read()  # ret = 1 if the video is captured; frame is the image
                if self.frame == self.length + 1:
                    break
                self.time = time.time() - self.start_time # gets the time of a given frame from the begining of the first frame
                # Display the resulting image
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                    break
                x = (time.time() - self.start_time) - (self.frame+1)*self.frame_latency # wait till the end start of the next frame
                if x > 0:
                    time.sleep(x)
            self.frame += 1

        # When everything done, release the capture
        np.savetxt(filename_part, (frame_state), delimiter=',',fmt="%s")
        cap.release()
        cv2.destroyAllWindows()
        

class Frame_Trigger (Thread):
    def __init__(self, Stream_other): # pulls both time and state into itself as per being defined in the initial state machine
        Thread.__init__(self)
        self.Stream_other = Stream_other
        self.frame_rate = self.Stream_other.frame_rate
    def run(self):
        toggle = 1 # pays attention to the first time you do something
        while True:
            self.frame = self.Stream_other.frame
            self.time = self.Stream_other.time
            if self.frame % self.frame_rate  != 0: # state_frame off
                state = 0
                toggle = 1
            else:
                state = 0
                if toggle == 1:
                    GPIO.output(pi2trig(16),1)
                    print(self.time)
                    state = 1 # state_frame on
                    time.sleep(trig_gap)
                    GPIO.output(pi2trig(255),0) # shoudn't send a trigger to turn off
##                    print(frame_state)
                toggle = 0
            append_frame_state(frame_state, self.frame, state)
            time.sleep(0.001)
        
filename_part = ("/home/pi/GitHub/GoPro_Visor_Eye_Pi/Pi3_Amp_Latencies/Video/" + partnum + "_" + filename + ".csv")
           

##########################################
# %% Threading of experiment itself to deal with systemic jitter an messy timings
            
if __name__ == '__main__':      
    stream = Stream() # initialize stream
    stream.start() # runs the run() method



