##import all the needed packages##
import board
import neopixel
import pigpio
import RPi.GPIO as GPIO
import time
from random import randint, shuffle
import numpy

##setup some constant variables##
partnum = '001'
filename = 'visual_p3_gopro_visor'

##number of trials##
trial_num = 10

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

def pi2trig(trig_num):
    
    pi_pins = [4,17,27,22,5,6,13,19]      
    
    bin_num = list(reversed(bin(trig_num)[2:]))
    
    while len(bin_num) < len(pi_pins):
        bin_num.insert(len(bin_num)+1,str(0))
    
    trig_pins = []
    
    trig_pos = 0
    
    for i_trig in range(len(pi_pins)):
        if bin_num[i_trig] == '1':
            trig_pins.insert(trig_pos,pi_pins[i_trig])
            trig_pos = trig_pos + 1
    
    return trig_pins

def get_resp_led_off(pin, led_on_time):
    start_resp = time.time()
    
    GPIO.wait_for_edge(pin,GPIO.RISING, timeout = int(led_on_time * 1000))
    
    led_off_time = time.time() - start_resp
    GPIO.output(pi2trig(3),1)
    
    if led_off_time < led_on_time:
        time.sleep(trig_gap)
        GPIO.output(pi2trig(3),0)
        time.sleep(trig_gap)
        resp_time = led_off_time
        time.sleep(led_on_time - led_off_time)
    else:
        resp_time = 0
        
    pixels.fill(blank)
    GPIO.output(pi2trig(4),1)
    time.sleep(trig_gap)
    GPIO.output(pi2trig(4),0)

    return resp_time

def get_resp(pin, wait_time, prev_delay, resp):
    start_resp = time.time()

    GPIO.wait_for_edge(pin,GPIO.RISING, timeout = int(wait_time * 1000))
    
    delay_end_time = time.time() - start_resp
    GPIO.output(pi2trig(5),1)

    if resp == 0:
        resp_time = delay_end_time + prev_delay
    else:
        resp_time = resp

    if delay_end_time < wait_time:
        time.sleep(wait_time - delay_end_time)

    return resp_time

##define the ip address for the second Pi we will be controlling##
##pi = pigpio.pi('192.168.1.216')

##distribution of targets and standards##
trials = numpy.zeros(int(trial_num*standard_rate)).tolist() + numpy.ones(int(trial_num*target_rate)).tolist()
shuffle(trials)

##variables to save trial information##
trig_time   = []
trig_type = []
delay_length  = []
trial_resp = []

##setup our neopixels##
pixels = neopixel.NeoPixel(pin_out, pin_num, brightness = brightness)

##to significy the start of the experiment##
##let's make the LEDs all red initially##
##and then wait for a certain amount of time##
GPIO.wait_for_edge(resp_pin,GPIO.RISING)
pixels.fill(red)
GPIO.output(pi2trig(10),1)
start_exp = time.time()
trig_time.append(time.time() - start_exp)
trig_type.append(3)
delay_length.append(1)
time.sleep(2)
pixels.fill(blank)
GPIO.output(pi2trig(255),0)
time.sleep(2)

for i_trial in range(len(trials)):
    start_trial = time.time()
    ###wait for a random amount of time between tones###
    delay = ((randint(0,500)*0.001)+1.0)
    delay_length.append(delay)
    ##determine the type of stimuli we will show on this trial##
    if trials[i_trial] == 0: #standards
        trig = 1
        pixels.fill(green)
##                pi.write(4, 1)
    elif trials[i_trial] == 1: #targets
        trig = 2
        pixels.fill(blue)
##                pi.write(17, 1)
    GPIO.output(pi2trig(trig),1)
    trig_type.append(trig)
    trig_time.append(time.time() - start_exp)
    time.sleep(trig_gap)
    GPIO.output(pi2trig(255),0)
    resp_time = get_resp_led_off(resp_pin, 1.0)
    resp_time = get_resp(resp_pin, delay, 1.0, resp_time)
    trial_resp.append(resp_time)
    end_trial = time.time()
    print(end_trial - start_trial)
    print(delay + 1.0)
    GPIO.output(pi2trig(255),0)
    time.sleep(trig_gap)

##end of experiment##
pixels.fill(red)
time.sleep(2)
pixels.fill(blank)
time.sleep(2)

###save trial information###
filename_part = ("/home/pi/Experiments/Visual_P3_GoPro_Visor/Data/Amp/Trial_Information/" + partnum + "_" + filename + ".csv")

numpy.savetxt(filename_part, (trig_type,trig_time, delay_length, trial_resp), delimiter=',',fmt="%s")

