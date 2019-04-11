##import all the needed packages##
import board
import neopixel
import pigpio
import RPi.GPIO as GPIO
import time
from random import randint, shuffle
import numpy
import pygame
import cv2
from threading import Thread, Timer

##setup some constant variables##
partnum = input("partnum: ")
filename = 'visual_p3_gopro_visor'

##number of trials##
trial_num = int(input("How many trials per block?: "))

##number of blocks
block_num = int(input("How many blocks?: "))

##standard and target rate##
standard_rate = 0.8
target_rate = 0.2

##several colours for the pixels##
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
blank = (0, 0, 0)
brightness = 0.2

##number of pixels we will be controlling##
pin_num = 6

##specify which pin we will be controlling the LEDs with##
pin_out = board.D18

##pins we will be using##
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

###setup pin for push button###
GPIO.setup(resp_pin,GPIO.IN,pull_up_down=GPIO.PUD_UP)

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

def resp_trig(trig): # maps response trigger to standard (3) or target (4)
    if trig == 1:
        resp_trig = 3
    else:
        resp_trig = 4
    GPIO.output(pi2trig(resp_trig),1)
    time.sleep(trig_gap)
    GPIO.output(pi2trig(255),0)
    time.sleep(trig_gap)


def get_resp_led_off(pin, led_on_time,trig): # get response (if occured in first 1 second) + turn off the LEDs regardless
    start_resp = time.time()

    GPIO.wait_for_edge(pin,GPIO.RISING, timeout = int(led_on_time * 1000))

    button_down = time.time() - start_resp # this is response time from the start of the 1 second response window

    if button_down < led_on_time: ## right now this isn't making any sense to me
        resp_trig(trig)
        resp_time = button_down
        if button_down <= 0.990:
            time.sleep(led_on_time - (button_down + trig_gap*2)) # wait until the end of the 1 second of the light being on
    else:
        resp_time = 0

    # before_second_light = time.time() - start_exp
    pixels.fill(blank)
    # after_second_light = time.time() - start_resp
    if trig == 1: ## Maps out offset trigger to standard and target flashes
        GPIO.output(pi2trig(5),1)
    else:
        GPIO.output(pi2trig(6),1)
    time.sleep(trig_gap)
    GPIO.output(pi2trig(255),0)

    return resp_time # before_second_light, after_second_light

def get_resp(pin, wait_time, prev_delay, resp, trig): # get response (if not in the first second) + wait for wait time (delay)
    start_resp = time.time()

    GPIO.wait_for_edge(pin,GPIO.RISING, timeout = int(wait_time * 1000))

    delay_end_time = time.time() - start_resp

    if resp == 0:
        resp_time = delay_end_time + prev_delay
        if resp_time <= 2.0:
            resp_trig(trig)
    else:
        resp_time = resp

    if delay_end_time < wait_time:
        time.sleep(wait_time - delay_end_time)

    return resp_time

def wheel(pos):
    # Input a value 0 to 255 to get a color value.
    # The colours are a transition r - g - b - back to r.
    if pos < 0 or pos > 255:
        r = g = b = 0
    elif pos < 85:
        r = int(pos * 3)
        g = int(255 - pos*3)
        b = 0
    elif pos < 170:
        pos -= 85
        r = int(255 - pos*3)
        g = 0
        b = int(pos*3)
    else:
        pos -= 170
        r = 0
        g = int(pos*3)
        b = int(255 - pos*3)
    return (r, g, b)


def rainbow_cycle(wait, rainbow_time):
    start = time.time()
    while time.time() - start < rainbow_time:
        for j in range(255):
            for i in range(pin_num):
                pixel_index = (i * 256 // pin_num) + j
                pixels[i] = wheel(pixel_index & 255)
            pixels.show()
            time.sleep(wait)

##define the ip address for the second Pi we will be controlling##
##pi = pigpio.pi('192.168.1.216')

##distribution of targets and standards##
trials = numpy.zeros(int(trial_num*standard_rate)).tolist() + numpy.ones(int(trial_num*target_rate)).tolist()
shuffle(trials) # randomize order of standards and targets

##variables to save trial information##
trig_time   = []
trig_type = []
delay_length  = []
trial_resp = []
jitter_length = []
resp_latency = []
block_start_stop = []
exp_start_stop = []
frame_state = [] # array of frame state
##setup our neopixels##
pixels = neopixel.NeoPixel(pin_out, pin_num, brightness = brightness, auto_write = True)

##to significy the start of the experiment##
##let's make the LEDs all red initially##
##and then wait for a certain amount of time##
# Render screen and fixation cross

############################################################## The following is for an integrated Video display
##### Seperate Thread
##### Will have to define a class that we update up (15) every 1 second & (16) 24 frames,
# then outside of thread we will check both 15/16 each iteration (confirm this doesn't add more time)
# does this mean another public thread?

class EndFrameFlag(Exception): # creat a custom error inherited from the exception class - for use in Stream
    pass

class EndFrameUpdateFlag(Exception): # creat a custom error inherited from the exception class - for use in Stream
    pass

class Stream (Thread): # construct - not an object
    def __init__(self):
        Thread.__init__(self)
        self.frame = 0 # if I want to pass this attrivute I have to pass the entire class to the other object/class
        self.time = 0
        self.file = 0 ### Change video input - should be in string format
        self.start_time = time.time() # time since the begining of the first frame
        self.frame_time = 0 # time since the begining of the current frame being draw

        cap = cv2.VideoCapture(self.file)
        self.frame_rate = get(cv2.CAP_PROP_FPS)
        #self.frame_rate = 24  # can enforce the frame rate - using EndFrameFlag

        self.frame_latency = 1/self.frame_rate # duration of a single frame
        self.frame_update_time = 0.01
        self.frame_update = 0

        self.trig = Frame_Trigger(self)
        self.trig.start()
        self.pin = Pin_Off(self, self.trig)
        self.pin.start()

    def run(self):
        cap = cv2.VideoCapture(self.file)
        self.start_time = time.time()
        while True:
            timer = Timer(self.frame_latency, lambda: raise EndFrameFlag) # start time that will wait one frame duration - then throws error
            timer.start()

            try:
                # Capture frame-by-frame
                ret, frame = cap.read()  # ret = 1 if the video is captured; frame is the image
                self.time = time.time() - self.start_time # gets the time of a given frame from the begining of the first frame
                # Display the resulting image
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                    break
                # immediately after the frame is drawn then a second timer is started to flash the frame_update for 10 ms
                frame_timer = Timer(self.frame_update_time, lambda: raise EndFrameUpdateFlag)
                frame_timer.start()

                try:
                    self.frame += 1 # counts frame number
                    self.frame_update = 1
                    time.sleep(1)
                except EndFrameUpdateFlag: # allows EndFrameUpdateFlag to pass
                    pass

                frame_update = 0
                time.sleep(1)
            except EndFrameFlag: # allows EndFrameFlag to pass
                pass

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

class Frame_Trigger (Thread):
    def __init__(self, Stream_other): # pulls both time and state into itself as per being defined in the initial state machine
        Thread.__init__(self)
        self.Stream_other = Stream_other
        self.frame_rate = Stream_other.frame_rate
    def run(self):
        While True:
            self.frame = self.Stream_other.frame
            self.time = Stream_other.time
            if self.frame % self.frame_rate  == 0: # state_frame off
                frame_state[self.Frame_State_other.frame] = 0
            else:
                GPIO.output(pi2trig(16),1)
                frame_state[self.Frame_State_other.frame] = 1 # state_frame on
                frame_time.append(self.time)
                time.sleep(trig_gap)
            time.sleep(0.001)
        return frame_state

class Pin_Off (Thread, Frame_Trigger_other):
    def __init__(self, Stream_other):
        Thread.__init__(self)
        self.Stream_other = Stream_other
    def run(self)
        while True:
            if self.Stream_other.frame_update == 1:
                self.trig_other.change = 0
                time.sleep(trig_gap)
                GPIO.output(pi2trig(255),0) # shoudn't send a trigger to turn off
            else:
                time.sleep(0.001)
                
##########################################
# %% Threading of experiment itself to deal with systemic jitter an messy timings

class ColourTimeFlag(Exception): # creat a custom error inherited from the exception class - for use in Stream
    pass

class TrigTimeFlag(Exception): # creat a custom error inherited from the exception class - for use in Stream
    pass

def Timer_LED(colour):
    colour_timer = Timer(LED_ON, lambda: raise ColourTimeFlag) # start time that will wait one frame duration - then throws error
    colour_timer.start()
    try:
        pixels.fill(colour)
        time.sleep(5)
    except ColourTimeFlag: # allows ColourTimeFlag to pass
            pass
    pixels.fill(blank)
    return
        
def Timer_Trig(trig):
    trig_timer = Timer(trig_gap, lambda: raise TrigTimeFlag) # start time that will wait one frame duration - then throws error
    trig_timer.start()
    try:
        GPIO.output(pi2trig(trig),1)
        time.sleep(1)
    except TrigTimeFlag: # allows TrigTimeFlag to pass
            pass
    GPIO.output(pi2trig(255),0)
    return    

class Colour_Switch (Thread):
    def __init__(self): # pulls both time and state into itself as per being defined in the initial state machine
        Thread.__init__(self)
    def run(self):
        While True:
            if trig_state == 0
                time.sleep(0.0001) # 1/10 of a millisecond
            else:
                if colour_state = 1:
                Timer_LED_ON(green)
                elif colour_state == 2:
                    Timer_LED(blue)
                else:
                    Timer_LED(red)
                return tc_state

class Trig_Switch (Thread):
    def __init__(self): # pulls both time and state into itself as per being defined in the initial state machine
        Thread.__init__(self)
    def run(self):
        While True:
            if trig_state == 0
                time.sleep(0.0001) # 1/10 of a millisecond
            else:
                Timer_Trig(trig_state)
                tc_state = 0
                return tc_state   
                  
for block in range(block_num):
    GPIO.wait_for_edge(resp_pin,GPIO.RISING) ## Waits for an initial button press to turn on the LED (red)
    Stream.run # initalize the video stream - on the viewpixx
    pixels.fill(red)
    if block == 0:
        GPIO.output(pi2trig(12),1) # send unique trigger for the start of the experiment
        time.sleep(trig_gap)
        start_exp = time.time()
        exp_start_stop.append(0)
    GPIO.output(pi2trig(10),1) # send unique trigger for the start of the block
    time.sleep(trig_gap)
    trig_time.append(time.time() - start_exp)
    block_start_stop.append(time.time() - start_exp) # start of each block from start_exp
    ## structure output of CSV
    trig_type.append(3)
    delay_length.append(2)
    trial_resp.append(0)
    jitter_length.append(0)
    resp_latency.append(0)
    time.sleep(2) ## leave red on for 2 seconds
    pixels.fill(blank)
    GPIO.output(pi2trig(255),0)
    time.sleep(2)
    for i_trial in range(len(trials)):
        start_trial = time.time() + trig_gap # define start time of a given trial
        delay = ((randint(0,500)*0.001)+1.0) # define delay, to be used later
        delay_length.append(delay)
        ##determine the type of stimuli we will show on this trial##
        if trials[i_trial] == 0: #standards
            tc_state = 1 # green
    ##                pi.write(4, 1)
        elif trials[i_trial] == 1: #targets
            tc_state = 2 # blue
    ##                pi.write(17, 1)
        GPIO.output(pi2trig(trig),1) ## Specify which trigger to send Standard vs Target
        trig_type.append(trig)
        trig_time.append(time.time() - start_exp)
        time.sleep(trig_gap)

        GPIO.output(pi2trig(255),0)
        resp_time = get_resp_led_off(resp_pin, 1.0,trig) # before_second_light, after_second_light
        resp_time = get_resp(resp_pin, delay, 1.0, resp_time,trig)
        resp_latency.append(time.time() - start_exp)
        trial_resp.append(resp_time)

        GPIO.output(pi2trig(255),0) ## doesn't give us a trigger
        time.sleep(trig_gap)
        end_trial = time.time()

        actual_trial_length = end_trial - start_trial
        theoretical_trial_length = delay + 1.0
        jitter = actual_trial_length - theoretical_trial_length
        jitter_length.append(jitter)

    ##end of experiment##
    pixels.fill(red)
    GPIO.output(pi2trig(11),1) # send unique trigger for the end of a block
    trig_time.append(time.time() - start_exp)
    block_start_stop.append(time.time() - start_exp) # end of each block from start_exp
    ## structure output of CSV
    trig_type.append(4)
    delay_length.append(2)
    trial_resp.append(0)
    jitter_length.append(0)
    resp_latency.append(0)
    time.sleep(2) ## leave red on for 2 seconds
    pixels.fill(blank)
    GPIO.output(pi2trig(255),0)
    time.sleep(2)

GPIO.output(pi2trig(13),1) # send unique trigger for the start of the experiment
time.sleep(trig_gap)
exp_start_stop.append(time.time() - start_exp)
rainbow_cycle(0.001, 5) ## After all blocks flash a rainbow at a refresh of (1st arguement) ms for (2nd arguement) seconds

pixels.fill(blank)

###save trial information###
filename_part = ("/home/pi/GitHub/GoPro_Visor_Eye_Pi/Pi3_Amp_Latencies/Pi_Time_Data/" + partnum + "_" + filename + ".csv")

# What is each thing
# trig_type
# trig_time
# delay_length
# trial_resp
# jitter_length
# resp_latency
# start_stop

numpy.savetxt(filename_part, (trig_type,trig_time, delay_length, trial_resp, jitter_length, resp_latency, block_start_stop, exp_start_stop), delimiter=',',fmt="%s")
