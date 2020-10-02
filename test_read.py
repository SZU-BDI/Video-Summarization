# -*- coding: utf-8 -*-
import cv2
import caffe
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import time

import sys
argc=len(sys.argv)
if(argc<2):
    print("usage: python $me $input_file")
    exit()
video_path=sys.argv[1]

#########################################################################
size_pool = 1000
q_frame_raw = []
q_frame = []
fps = 25
fps_target = 3
fps_jump = 0
total_frames = 0
counter = 0
thres_distt = 20000

flg_end = False
def sigint_handler(signum, frame):
    global flg_end
    flg_end = True

import signal
signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGHUP, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)

import threading

import os
def get_frame_rate(filename):
    if not os.path.exists(filename):
        sys.stderr.write("ERROR: filename %r was not found!" % (filename,))
        return -1         
    out = os.popen("ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate "+filename).read()
    rate = out.split('/')
    if len(rate)==1:
        return float(rate[0])
    if len(rate)==2:
        return float(rate[0])/float(rate[1])
    return -1

def th_producer():
    global flg_end,fps,total_frames, fps_jump
    capture = cv2.VideoCapture(video_path)

    # NOTES: error when big mp4
    #total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    #fps = int(capture.get(cv2.CAP_PROP_FPS))
    fps_jump = int(fps / fps_target)
    while(True):
        if flg_end:
            break
        #capture.set(1,total_frames + fps_jump) # NOTES: not good, slow
        skip = 0 #+ fps_jump
        while True:
            ret2, frame = capture.read()
            skip -= 1
            if not ret2 or skip<=0:
                break
        if ret2 is True:
            #total_frames += fps_jump - skip
            total_frames += 1
        else:
            flg_end = True
            break # while
    capture.release()  

start_t = time.time()
fps = get_frame_rate(video_path)
t_prd = threading.Thread(target=th_producer) # producer
t_prd.start()
t_prd.join()
end_t = time.time()
print("SRC FPS = " , fps, "\t total frames(est)= ",total_frames)
print("TGT FPS =",fps_target,"\t output=", counter, "\t time=", end_t - start_t)

