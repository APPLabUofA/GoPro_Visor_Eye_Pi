# Open world EEG

## General

Completely self-contained - 
Goals start indoors and move outside
Pi is running everything - possibly a panda or a tablet recording 
Visor = LEDs + safety glasses


## Prototype 1
Visor + button + EEG + MP4 Video File (Task 2)

### Task 1:  indoor, sitting down, fixation cross (otherwise blank screen)
Visor blink, two conditions (1 w/o response, 1 w/ resp) * should get comparable P3 w/o

### Task 2:  indoor, sitting down, fixation cross
Visor blink, two conditions (1 w/o response, 1 w/ resp) * should get comparable P3 w/o
Video in background either a busy scene or a park scene (w/o kids)

Trigger the video stream with markers of distinguishable stimuli as it enters the peripheral and focal view

Decide w/w/o audio (Task 2.5)

Relate Task 1 and Task 2 and the interference based on extraneous video stream oddballs
(break down the identification of vehicles into 10 major categories and correlate with EEG principle components)

These tasks allow us to decide whether to continue with button response and/or focuses of  attention to certain aspects of the environment, or just free wandering

Technical Challenges - taking the ~250 fps video stream, reducing it to ~ 24 fps and embedding/drawing a fixation cross in the center, marking frames with onset triggers and triggering stimuli within the video (50 hours of work) - Using OpenCV/FFMPEG/pygame

## Prototype 2
Visor + button + EEG + MP4 Video File + Eye Tracking (Home made rig?)

### Task 3: Task 2 + eye tracking (w/ w/o fixation cross)

## Prototype 3
Visor + button + EEG + MP4 Video File + Eye Tracking (Home made rig?) + Binocular external video stream (3D Mapping)

Depth Perception Test?

### Task 4: Walk/Bike down sask drive /across the river (in different contexts)
