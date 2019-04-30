# -*- coding: utf-8 -*-
import cv2
from threading import Thread, Timer
import time
import pigpio
import RPi.GPIO as GPIO


class Stream (Thread): # construct - not an object - creat object and call Thread (spinning + running thread)
    def __init__(self):
        Thread.__init__(self)
        self.frame = 0 # if I want to pass this attrivute I have to pass the entire class to the other object/class
        self.time = 0
        self.file = 0 ### Change video input - should be in string format
        self.frame_time = 0 # time since the begining of the current frame being draw

        cap = cv2.VideoCapture(self.file)
        self.frame_rate = get(cv2.CAP_PROP_FPS)
        #self.frame_rate = 24  # can enforce the frame rate - using EndFrameFlag

        self.frame_latency = 1/self.frame_rate # duration of a single frame
        self.frame_update_time = 0.01
        self.frame_update = 0

        self.trig = Frame_Trigger(self)
        self.trig.start()
        self.start_time = time.time()

    def run(self):
        cap = cv2.VideoCapture(self.file)
        while True:
#            timer = Timer(self.frame_latency, lambda: raise EndFrameFlag) # start time that will wait one frame duration - then throws error
#            timer.start()
            self.frame_start = time.time() - self.start_time 
            if self.frame_start <= (self.frame+1)*self.frame_latency
            
                # Capture frame-by-frame
                ret, frame = cap.read()  # ret = 1 if the video is captured; frame is the image
                self.time = time.time() - self.start_time # gets the time of a given frame from the begining of the first frame
                # Display the resulting image
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                    break
                
                time.sleep((time.time() - self.start_time) - (self.frame+1)*self.frame_latency ) # wait till the end start of the next frame
                
            self.frame += 1

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
            if self.frame % self.frame_rate  != 0: # state_frame off
                frame_state[self.Frame_State_other.frame] = 0
            else:
                GPIO.output(pi2trig(16),1)
                frame_state[self.Frame_State_other.frame] = 1 # state_frame on
                frame_time.append(self.time)
                time.sleep(trig_gap)
                GPIO.output(pi2trig(255),0) # shoudn't send a trigger to turn off
            time.sleep(0.001)
        return frame_state

##########################################
# %% Threading of experiment itself to deal with systemic jitter an messy timings
            
if __name__ == '__main__':
    refresh_screen()
    window_name = 'projector'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, img)
    for i in length(instruct)
        cv2.putText(img,instruct(i),(int(2*width/5),height), cv2.FONT_HERSHEY_SIMPLEX , 1,(255,255,255),2,cv2.LINE_AA)
        GPIO.wait_for_edge(resp_pin,GPIO.RISING)
        refresh_screen()
        cv2.imshow(window_name, img)
           
stream = Stream() # initialize stream
stream.start() # runs the run() method