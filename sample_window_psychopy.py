#import board
#import neopixel
#import pigpio
#import RPi.GPIO as GPIO
import time
#from random import randint, shuffle
#import numpy
#import pygame
#from threading import Thread
from psychopy import visual, core, event

#intialize colourspace
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
blank = (0, 0, 0)
white = (225,225,225)
brightness = 0.2



############################################3 setup the button
pin_num = 6

##specify which pin we will be controlling the LEDs with##
#pin_out = board.D18

##pins we will be using##
trig_pins = [4,17,27,22,5,6,13,19]
resp_pin = 21

##amount of time needed to reset triggers##
trig_gap = 0.005

##define the ip address for the second Pi we will be controlling##
##pi = pigpio.pi('192.168.1.216')

###initialise GPIO pins###
#GPIO.setmode(GPIO.BCM)
#GPIO.setup(trig_pins,GPIO.OUT)
#
####set triggers to 0###
#GPIO.output(trig_pins,0)
#
####setup pin for push button###
#GPIO.setup(resp_pin,GPIO.IN,pull_up_down=GPIO.PUD_UP)

##setup neopixel + fill blank
#pixels = neopixel.NeoPixel(pin_out, pin_num, brightness = brightness, auto_write = True)
#time1 = time.time()
#pixels.fill(blank)
#time2 = time.time()
#print(time2 -time1)
####################################################################


    # replace this with GPIO pin uprising

# Initialize Instructions    
test_text = "This is a test screen. The first trial is about to begin"
#instructions1 = 'Focus on the central fixation cross.'
#instructions2 = 'Press the button when you see blue flashes, do NOT press the spacebar when you see green flashes.'
#instructions3 = 'Press the button when you are ready to start.'
instructions1 = 'Focus on the central fixation cross. Press the button when you see blue flashes, do NOT press the spacebar when you see green flashes. Press the button when you are ready to start.'
break_screen = 'Feel free to take a break at this time. Press the button when you are ready to start.'
end_screen = 'Congratulations, you have finished the experiment! Please contact the experimenter.'
#############################
# %% ## Initialize Psychopy window
#mywin = visual.Window([1440, 900], monitor="testMonitor", units="deg", fullscr=True)

# DEBUG WINDOW
mywin = visual.Window(size=[400, 400], units="pix", fullscr=False, color=[1, 1, 1])
mywin.mouseVisible = False
fixation = visual.GratingStim(win=mywin, size=0.2, pos=[0, 0], sf=0)
def text_time(text_blurb):
    text = visual.TextStim(
    win=mywin,
    text=text_blurb,
    color=[-1, -1, -1],
    pos= [0,5])
    text.draw()
    fixation.draw()
    mywin.flip()
                            
text_time(test_text)
#text_time(test_text)  
# run experiment

#time.sleep(5) - works
#GPIO.wait_for_edge(resp_pin,GPIO.RISING)
time.sleep(1)
text_time(instructions1)
time.sleep(1)
text_time(break_screen)
#pixels.fill(red)
time.sleep(1)
text_time(end_screen)
time.sleep(5)


# throw the whole thing in a loop so the experiment can be stopped by the esc or q keys

event.waitKeys(keyList="space") # space bar to quit the window
mywin.mouseVisible = True

# Cleanup
mywin.close()
core.quit()

