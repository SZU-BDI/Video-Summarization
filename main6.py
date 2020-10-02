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
proto = "VS-Python/Models/mobilenet_v2_deploy.prototxt"
model = "../d/mobilenet_v2.caffemodel"
caffe.set_mode_gpu()
#caffe.set_mode_cpu()
net = caffe.Net(proto, model, caffe.TEST)

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data',(2, 0, 1))
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_raw_scale('data', 255)

net.blobs['data'].reshape(1, 3, 224, 224)

#########################################################################
'''
proto1 = "VS-Python/Models/deploy.prototxt"
model1 = "../d/memnet.caffemodel"
net1 = caffe.Net(proto1, model1, caffe.TEST)

transformer1 = caffe.io.Transformer({'data':net1.blobs['data'].data.shape})
transformer1.set_transpose('data',(2, 0, 1))
transformer1.set_channel_swap('data', (2, 1, 0))
transformer1.set_raw_scale('data', 255)
net1.blobs['data'].reshape(1, 3, 227, 227)
'''
#########################################################################
'''
def mem_calculation(resized_image):
    net1.blobs['data'].data[...] = transformer1.preprocess('data', resized_image)
    value = net1.forward()
    value = value['fc8-euclidean']
    return value[0][0]
'''

def get_features(image):
    #net.blobs['data'].reshape(1, 3, 224, 224)
    net.blobs['data'].data[...] = transformer.preprocess('data', image)
    net.forward() 
    features = net.blobs['fc7'].data[0].reshape(1,1000)
    features = np.array(features)
    return features

def shot_segment_distt(resized_image1,resized_image2):
    return euclidean_distances(get_features(resized_image1),get_features(resized_image2))
    t = []
    t.append(time.time())
#        transformer.set_mean('data',img_mean)
    net.blobs['data'].reshape(1, 3, 224, 224)
    net.blobs['data'].data[...] = transformer.preprocess('data', resized_image1)
    t.append(time.time())
    net.forward() # 0.08
    t.append(time.time())
    features1 = net.blobs['fc7'].data[0].reshape(1,1000)
    features1 = np.array(features1)

    #net.blobs['data'].reshape(1, 3, 224, 224) # wjc tmp test...
    net.blobs['data'].data[...] = transformer.preprocess('data', resized_image2)
    t.append(time.time())
    net.forward() # 0.08
    t.append(time.time())
    features2 = net.blobs['fc7'].data[0].reshape(1,1000)
    features2 = np.array(features2)

    t.append(time.time())
    rt=euclidean_distances(features1,features2)
    t.append(time.time())
    print('t=',t)
    return rt

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

def th_pre_proc(): #{
    global flg_end
    while True:
        if (flg_end):
            break # quick while True
        len_q_frame = len(q_frame_raw)
        if len_q_frame>0:
            frame_pop_a = q_frame_raw.pop()
            frame_pop_a[3] = get_features(frame_pop_a[1])
            q_frame.append(frame_pop_a)
        else:
            time.sleep(0.01) # let cpu have a rest
    #} th_pre_proc()
def th_handling(): #{
    global flg_end, counter, fps, total_frames
    hhh = 0
    frame1_a, frame2_a = (False,False)
    frame1_a_f = False
    frame2_a_f = False
    while True:
        if (flg_end):
            break # quick while True
        #len_q_frame = len(q_frame)
        len_q_frame = len(q_frame_raw)
        if len_q_frame>0:
            print('hhh=', hhh, ", len_q_frame=", len_q_frame, 'total_frames=',total_frames)
            #hhh +=1 
            hhh += fps_jump
            #frame_pop_a = q_frame.pop()
            frame_pop_a = q_frame_raw.pop()
            if not frame1_a:
                frame1_a = frame_pop_a
                #frame1_a_f = get_features(cv2.resize(frame1_a[0],(224,224)))
                #frame1_a_f = frame1_a[3]
                frame1_a_f = get_features(frame1_a[1])
            else:
                frame2_a = frame_pop_a
                #frame2_a_f = get_features(cv2.resize(frame2_a[0],(224,224)))
                #frame2_a_f = frame2_a[3]
                frame2_a_f = get_features(frame2_a[1])
                # distt = shot_segment_distt(frame1_a[1],frame2_a[1]) 
                # distt = euclidean_distances(frame1_a[3],frame2_a[3]) 
                distt = euclidean_distances(frame1_a_f,frame2_a_f) 
                distt = int(distt)
                #fc8 = mem_calculation(frame1_a[2]) 
                print ('... ', hhh, ',distt=',distt)
                if distt >= thres_distt:
                    pathh = '../d/frame' + str(counter).rjust(3,"0") + '_' + str(hhh).rjust(6,"0") + '_' + str(distt).rjust(5,'0') + '.png'
                    print (pathh)
                    counter = counter + 1
                    cv2.imwrite(pathh,frame2_a[0]) # tmp. test
                    frame1_a = frame2_a
                    frame1_a_f = frame2_a_f
        else:
            time.sleep(0.1) # let cpu have a rest
    # } th_handling()
import sys
import os
import subprocess
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
        skip = 0 + fps_jump
        while True:
            ret2, frame = capture.read()
            skip -= 1
            if not ret2 or skip==0:
                break
        if ret2 is True:
            #total_frames += 1
            #total_frames += fps_jump
            total_frames += fps_jump - skip
            while len(q_frame_raw)>size_pool:
                time.sleep(0.1)
            frame_resized = cv2.resize(frame,(224,224))
            frame_memcalc = False # not using now...
            #the_features = get_features(frame_resized)
            q_frame_raw.append((frame, frame_resized, frame_resized,
                #the_features
                ))
            #time.sleep(0.01)
        else:
            flg_end = True
            break # while
    capture.release()  

start_t = time.time()
fps = get_frame_rate(video_path)
#t_hdp = threading.Thread(target=th_pre_proc) # pre-processing
t_prd = threading.Thread(target=th_producer) # producer
t_hdl = threading.Thread(target=th_handling) # consumer
#t_hdp.start()
t_hdl.start()
t_prd.start()
#t_hdp.join()
t_hdl.join()
t_prd.join()
end_t = time.time()
print("SRC FPS = " , fps, "\t total frames(est)= ",total_frames)
print("TGT FPS =",fps_target,"\t output=", counter, "\t time=", end_t - start_t)

