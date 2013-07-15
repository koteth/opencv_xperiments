#!/usr/bin/env python
""" 
optical flow interpolation example.
author: koteth
pandoricweb.tumblr.com
"""


import numpy as np
import cv2
import cv2.cv as cv

# video.py from opencv.../samples/python2
import video

help_message = '''
USAGE: optical_flow_interpolation.py [<video_source>] [<video_out>] [<out_fps>]
'''

def draw_flow(img, flow, step=22):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] /=2.0
    flow[:,:,1] /=2.0
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_CUBIC, borderMode =cv2.BORDER_REPLICATE )
    return res

if __name__ == '__main__':
    import sys
    try: 
        fn = sys.argv[1]
        out_fn = sys.argv[2]
        fps = 50 
        if len(sys.argv) > 3:
            fps = int( sys.argv[3] ) 

    except: 
        print help_message
        sys.exit(1)


    encoding = 'MP4V'
    cam = video.create_capture(fn)
    ret, prev = cam.read()
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    h, w = prevgray.shape[:2]
    out = cv2.VideoWriter( out_fn, cv.CV_FOURCC(*encoding), fps, (w, h), 1)

    while True:
        ret, img = cam.read()
        img_to_warp = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 2, 10, 5, 7, 1.2,0)
        prevgray = gray
        cv2.imshow('flow', draw_flow(gray, flow))
        warped_img = warp_flow(img_to_warp, flow)

        if out:
            out.write(img)
            out.write(warped_img)

        ch = 0xFF & cv2.waitKey(5)
        if ch == 27:
            break
    cv2.destroyAllWindows()

