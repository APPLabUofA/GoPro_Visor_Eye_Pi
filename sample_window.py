import board
import neopixel
import pigpio
import RPi.GPIO as GPIO
import time
from random import randint, shuffle
import numpy
import pygame
from threading import Thread

#intialize colourspace
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
blank = (0, 0, 0)
white = (225,225,225)
brightness = 0.2

##intialize pygame
pygame.init()
pygame.display.init()


############################################3 setup the button
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

##setup neopixel + fill blank
pixels = neopixel.NeoPixel(pin_out, pin_num, brightness = brightness, auto_write = True)
time1 = time.time()
pixels.fill(blank)
time2 = time.time()
print(time2 -time1)
####################################################################

def inital_instruct():
    pygame.draw.line(screen, (255, 255, 255), (x_center-10, y_center), (x_center+10, y_center),4)
    pygame.draw.line(screen, (255, 255, 255), (x_center, y_center-10), (x_center, y_center+10),4)
    screen.blit(instructions1,(x_center-((instructions1.get_rect().width)/2),y_center+((instructions1.get_rect().height)*1)+10))
    screen.blit(instructions2,(x_center-((instructions2.get_rect().width)/2),y_center+((instructions2.get_rect().height)*2)+10))
    screen.blit(instructions3,(x_center-((instructions3.get_rect().width)/2),y_center+((instructions3.get_rect().height)*3)+10))
    pygame.display.flip()

def black_instructions():
    screen.fill(pygame.Color("black"))
    pygame.draw.line(screen, (255, 255, 255), (x_center-10, y_center), (x_center+10, y_center),4)
    pygame.draw.line(screen, (255, 255, 255), (x_center, y_center-10), (x_center, y_center+10),4)
    pygame.display.flip()
    time.sleep(1)

def refresh(): # called at the start and end of each block - add place holders for output csv. and resets triggers + visor LEDs
    trig_type.append(3)
    delay_length.append(2)
    trial_resp.append(0)
    jitter_length.append(0)
    resp_latency.append(0)
    time.sleep(2) ## leave red on for 2 seconds
    pixels.fill(blank)
    GPIO.output(pi2trig(255),0)
    time.sleep(2)

##Render the screen and fixation cross##
w = 100
y = 5*w
x = y*2
pygame.mouse.set_visible(0)
disp_info = pygame.display.Info()
screen = pygame.display.set_mode((x,y),pygame.RESIZABLE)
x_center = x/2
y_center = y/2

###setup our instruction screens###
pygame.font.init()
myfont = pygame.font.SysFont('Times New Roman', 20)
instructions1 = myfont.render('Focus on the central fixation cross.', True, white)
instructions2 = myfont.render('Press the button when you see blue flashes, do NOT press the spacebar when you see green flashes.', True, white)
instructions3 = myfont.render('Press the button when you are ready to start.', True, white)
break_screen = myfont.render('Feel free to take a break at this time. Press the button when you are ready to start.', True, white)
end_screen = myfont.render('Congratulations, you have finished the experiment! Please contact the experimenter.', True, white)



# run experiment
inital_instruct()
#time.sleep(5) - works
GPIO.wait_for_edge(resp_pin,GPIO.RISING)
black_instructions()
time.sleep(1)
pixels.fill(red)
time.sleep(1)
screen.blit(break_screen,(x_center-((break_screen.get_rect().width)/2),y_center+10))
time.sleep(5)
black_instructions()

# throw the whole thing in a loop so the experiment can be stopped by the esc or q keys



