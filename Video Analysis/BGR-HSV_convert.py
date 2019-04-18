# -*- coding: utf-8 -*-

        green = np.uint8([[[0,255,0 ]]])
        hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
        print('green hsv = {}'.format(hsv_green))
        [[[ 60 255 255]]]
        
        red = np.uint8([[[255,0,0 ]]])
        hsv_red = cv2.cvtColor(red,cv2.COLOR_BGR2HSV)
        print('red hsv = ' + hsv_red)
        
        blue = np.uint8([[[0,0,255]]])
        hsv_blue = cv2.cvtColor(blue,cv2.COLOR_BGR2HSV)
        print('blue hsv = ' + hsv_blue)
        