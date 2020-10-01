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

#########################################################################
proto1 = "VS-Python/Models/deploy.prototxt"
model1 = "../d/memnet.caffemodel"
net1 = caffe.Net(proto1, model1, caffe.TEST)

transformer1 = caffe.io.Transformer({'data':net1.blobs['data'].data.shape})
transformer1.set_transpose('data',(2, 0, 1))
transformer1.set_channel_swap('data', (2, 1, 0))
transformer1.set_raw_scale('data', 255)
net1.blobs['data'].reshape(1, 3, 227, 227)

#########################################################################
def mem_calculation(frame1):
    resized_image = cv2.resize(frame1,(227,227))
    net1.blobs['data'].data[...] = transformer1.preprocess('data', resized_image)
    value = net1.forward()
    value = value['fc8-euclidean']
    return value[0][0]
def shot_segment_distt(frame1,frame2):
    t = []
    t.append(time.time())
    resized_image1 = cv2.resize(frame1,(224,224))
    resized_image2 = cv2.resize(frame2,(224,224))
    t.append(time.time())
#        transformer.set_mean('data',img_mean)
    net.blobs['data'].reshape(1, 3, 224, 224)
    net.blobs['data'].data[...] = transformer.preprocess('data', resized_image1)
    t.append(time.time())
    net.forward()
    t.append(time.time())
    features1 = net.blobs['fc7'].data[0].reshape(1,1000)
    features1 = np.array(features1)
    net.blobs['data'].data[...] = transformer.preprocess('data', resized_image2)
    t.append(time.time())
    net.forward()
    t.append(time.time())
    features2 = net.blobs['fc7'].data[0].reshape(1,1000)
    features2 = np.array(features2)
    t.append(time.time())
    rt=euclidean_distances(features1,features2)
    t.append(time.time())
    #print (t)
    return rt

size_pool = 1000
q_frame = []
flg_end = False

import threading

def th_handling():
    global flg_end
    hhh = 0
    while True:
        len_q_frame = len(q_frame)
        print('hhh=', hhh, ", len(q_frame)=", len_q_frame)
        if len_q_frame>0:
            hhh +=1 
            frame_pop_a = q_frame.pop()
            #time.sleep(0.1) # TODO handling...
        else:
            #print("flg_end=", flg_end);
            if (flg_end):
                break
            time.sleep(0.1)

def th_frames():
    global flg_end
    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))#getting total number of frames
    fps = int(capture.get(cv2.CAP_PROP_FPS))#getting frame rate
    '''
    print ("FPS = " , fps, "\t total frames = ",total_frames)
    m_scores = []
    m_scores.append([])
    m_scores.append([])
    '''
    #ret, frame1 = capture.read()
    counter = 0
    ttt = 0
    distt = 0

    #pathh = '../d/frame' + str(counter).rjust(3,"0") + '_' + str(ttt).rjust(6,"0") + '_' + str(distt).rjust(5,'0') + '.png'
    #print (pathh)
    #counter = counter + 1
    #cv2.imwrite(pathh,frame1) # tmp. test

    start_t = time.time()
    while(True):
        ttt = ttt + 1
        ret2, frame2 = capture.read()
        if ret2 is True:
            #print (time.time(), "ttt=", ttt);
            while len(q_frame)>size_pool:
                #print("q full, wait for handling")
                time.sleep(0.1)
            q_frame.append((frame2,cv2.resize(frame2,(224,224))))
            '''
            distt = shot_segment_distt(frame1,frame2) 
            distt = int(distt)
            fc8 = mem_calculation(frame1) 
            print ('Processing ... ', ttt, ', of ', total_frames, ',distt=',distt, ',fc8=',fc8)
            if distt >= 20000:
                pathh = '../d/frame' + str(counter).rjust(3,"0") + '_' + str(ttt).rjust(6,"0") + '_' + str(distt).rjust(5,'0') + '.png'
                print (pathh)
                counter = counter + 1
                cv2.imwrite(pathh,frame2) # tmp. test
            '''
            #
            '''
            if distt >= 40000:#{ different images , 25x4
                m_scores = np.array(m_scores)
                [rows,cols] = m_scores.shape
                
                if cols > 0:
                    max_index = m_scores[0].argmax()
                    keyframe_number = int(m_scores[1][max_index])
                    keyframe = capture.set(cv2.CAP_PROP_POS_MSEC, keyframe_number)
                    temp, keyframe = capture.read()
                    pathh = '../d/frame' + str(keyframe_number).rjust(6,"0") + '.png'
                    print ("############## \t 'Writing key frame at" , pathh, "'\t##############", "\a")
                    cv2.imwrite(pathh,keyframe)
                print ("Different images = " , distt , "\t" , counter)
                m_scores = []
                m_scores.append([])
                m_scores.append([])
                #}
            else:#{ same images
                print ("Similar images= " , distt , "\t" , counter)
                m_value = Main.mem_calculation(frame1)  # time 0.15
                m_scores[0].append(m_value)
                m_scores[1].append(counter)
                #}
            counter = counter + fps
            frame1 = frame2
            '''
        else:
            flg_end = True
            end_t = time.time()
            totall = end_t - start_t
            print ("time total=", totall, ', ttt=', ttt, ', output=', counter)
            break # while
    capture.release()  

def main4(): #{
    t_hdl = threading.Thread(target=th_handling)
    t_frm = threading.Thread(target=th_frames)
    t_hdl.start()
    t_frm.start()
    print("TODO join() and then quick");

#if __name__ == '__main__':
#    Main.main2()
main4()

