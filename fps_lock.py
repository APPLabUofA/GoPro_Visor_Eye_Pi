import time
import sys
from threading import Thread, Timer
import queue

class FPSTimer:
    def __init__(self, fps=60):
        self.start_time = time.time()
        self.fps = fps
        self.timer = LockedFPS1(fps, self.start_time, offset=True)
        self.timer.start()

class CustomTimer(Timer):
    def __init__(self, time, catch):
        Timer.__init__(self, time, self.catchEnd)
        self.catch = catch

    def catchEnd(self):
        self.catch.put(sys.exc_info())


# Fancy polling method (pretty inefficiant)
class LockedFPS1(Thread):
    def __init__(self, fps, start_time, offset):
        Thread.__init__(self)
        self.start_time = start_time
        self.fps = fps
        self.target_time = start_time + (1./fps)
        self.catch = queue.Queue()
        self.end = False

        # Make a dynamic offset to mitigate delays
        self.offset = offset
        self.offset_amount = 0
        self.offset_count = 0
            
    def run(self):
        while(not self.end):
            time_left = max(self.target_time - time.time(), 0)
            if not self.offset:
                time_left += self.offset_amount / self.fps
            timer = CustomTimer(time_left, self.catch)
            timer.start()
            
            while(1):
                try:
                    self.catch.get(block=False)
                except queue.Empty:
                    pass
                else:
                    break


            epsilon = self.target_time - time.time()
            print("Error in time: {} ms".format(round(epsilon*1000, 3)))
            if self.offset:
                self.offset_amount += epsilon
                self.offset_count += 1
                if self.offset_count == self.fps:
                    self.offset = False
            self.target_time += 1./self.fps


# Simple polling method (a lot less overhead)
class LockedFPS2(Thread):
    def __init__(self, fps, start_time):
        Thread.__init__(self)
        self.start_time = start_time
        self.fps = fps
        self.target_time = start_time + (1./fps)
        self.end = False

    def run(self):
        while(not self.end):
            while(self.target_time - time.time() > 0):
                pass

            epsilon = self.target_time - time.time()
            print("Error in time: {} μs".format(round(epsilon*1000000, 3)))
            self.target_time += 1./self.fps




# A very hacky way to do asynchonous execptions (make sure to set MACOS to True if on MacOs)
MACOS = False
import ctypes
import threading

class FPSTimeout(Exception):
    pass

class CustomTimer2(Timer):
    def __init__(self, time, parent_thread):
        Timer.__init__(self, time, self.timerEnd)
        self.parent_thread = parent_thread

    def timerEnd(self):
        ctype_async_raise(self.parent_thread, FPSTimeout)

NULL = 0

def ctype_async_raise(thread_obj, exception):
    found = False
    target_tid = 0
    for tid, tobj in threading._active.items():
        if tobj is thread_obj:
            found = True
            target_tid = tid
            break

    if not found:
        raise ValueError("Invalid thread object")

    if MACOS:
        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(target_tid), ctypes.py_object(exception))
    else:
        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(target_tid, ctypes.py_object(exception))
    # ref: http://docs.python.org/c-api/init.html#PyThreadState_SetAsyncExc
    if ret == 0:
        raise ValueError("Invalid thread ID")
    elif ret > 1:
        # Huh? Why would we notify more than one threads?
        # Because we punch a hole into C level interpreter.
        # So it is better to clean up the mess.
        ctypes.pythonapi.PyThreadState_SetAsyncExc(target_tid, NULL)
        raise SystemError("PyThreadState_SetAsyncExc failed")
    #print("Successfully set asynchronized exception for", target_tid)



class LockedFPS3(Thread):
    def __init__(self, fps, start_time, offset):
        Thread.__init__(self)
        self.start_time = start_time
        self.fps = fps
        self.target_time = start_time + (1./fps)
        self.end = False

        # Make a dynamic offset to mitigate delays
        self.offset = offset
        self.offset_amount = 0
        self.offset_count = 0
        

    def run(self):
        while(not self.end):
            try:
                time_left = max(self.target_time - time.time(), 0)
                if not self.offset:
                    time_left += self.offset_amount / self.fps
                timer = CustomTimer2(time_left, self)
                timer.start()

                while(1):
                    pass

            except FPSTimeout:
                epsilon = self.target_time - time.time()
                print("Error in time: {} ms".format(round(epsilon*1000, 3)))
                if self.offset:
                    self.offset_amount += epsilon
                    self.offset_count += 1
                    if self.offset_count == self.fps:
                        self.offset = False
                self.target_time += 1./self.fps
                

if __name__ == '__main__':
    #timer = FPSTimer()
    fps = 30
    
    print('Timer1')
    timer1 = LockedFPS1(fps, time.time(), offset=True)
    timer1.start()
    time.sleep(2)
    timer1.end = True
    time.sleep(0.1)

    print('Timer2')
    timer2 = LockedFPS2(fps, time.time())
    timer2.start()
    time.sleep(2)
    timer2.end = True
    time.sleep(0.1)

    print('Timer3')
    timer3 = LockedFPS3(fps, time.time(), offset=True)
    timer3.start()
    time.sleep(2)
    timer3.end = True
    time.sleep(0.1)
    
