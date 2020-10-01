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
def mem_calculation(resized_image):
    net1.blobs['data'].data[...] = transformer1.preprocess('data', resized_image)
    value = net1.forward()
    value = value['fc8-euclidean']
    return value[0][0]

def shot_segment_distt(resized_image1,resized_image2):
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
    net.blobs['data'].data[...] = transformer.preprocess('data', resized_image2)
    t.append(time.time())
    net.forward() # 0.08
    t.append(time.time())
    features2 = net.blobs['fc7'].data[0].reshape(1,1000)
    features2 = np.array(features2)
    t.append(time.time())
    rt=euclidean_distances(features1,features2)
    t.append(time.time())
    #print(t)
    return rt

size_pool = 1000
q_frame = []
flg_end = False
total_frames = 0
fps = 0
counter = 0
fps_target = 4

import threading

def th_handling():
    global flg_end, total_frames, counter
    hhh = 0
    frame1_a, frame2_a = (False,False)
    while True:
        len_q_frame = len(q_frame)
        print('hhh=', hhh, ", len(q_frame)=", len_q_frame)
        if len_q_frame>0:
            hhh +=1 
            frame_pop_a = q_frame.pop()
            if not frame1_a:
                frame1_a = frame_pop_a
                #pathh = '../d/frame' + str(counter).rjust(3,"0") + '_' + str(hhh).rjust(6,"0") + '_' + str(distt).rjust(5,'0') + '.png'
                #print (pathh)
                #counter = counter + 1
                #cv2.imwrite(pathh,frame1) # tmp. test
                #break #while
            else:
                skip = 0
                while skip <= fps/fps_target:
                    skip += 1
                    hhh +=1
                    len_q_frame = len(q_frame)
                    if len_q_frame>0:
                        frame_pop_a = q_frame.pop()
                    else:
                        break #current while....
                frame2_a = frame_pop_a
                distt = shot_segment_distt(frame1_a[1],frame2_a[1]) 
                distt = int(distt)
                print ('... ', hhh, ', of ', total_frames, ',distt=',distt)
                #fc8 = mem_calculation(frame1_a[2]) 
                #print ('... ', hhh, ', of ', total_frames, ',distt=',distt, ',fc8=',fc8)
                if distt >= 20000:
                    pathh = '../d/frame' + str(counter).rjust(3,"0") + '_' + str(hhh).rjust(6,"0") + '_' + str(distt).rjust(5,'0') + '.png'
                    print (pathh)
                    counter = counter + 1
                    cv2.imwrite(pathh,frame2_a[0]) # tmp. test
                frame1_a = frame2_a
                '''
                m_scores = []
                m_scores.append([])
                m_scores.append([])
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
            #print("flg_end=", flg_end);
            if (flg_end):
                break # quick while True
            else:
                time.sleep(0.1) # let cpu have a rest

def th_producer():
    global flg_end, total_frames, fps
    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    while(True):
        ret2, frame2 = capture.read()
        if ret2 is True:
            while len(q_frame)>size_pool:
                time.sleep(0.1)
            #q_frame.append((frame2,cv2.resize(frame2,(224,224))))
            q_frame.append((frame2,
                cv2.resize(frame2,(224,224)),
                cv2.resize(frame2,(227,227)),
                ))
            time.sleep(0.01)
        else:
            flg_end = True
            break # while
    capture.release()  

def main4():
    t_hdl = threading.Thread(target=th_handling) # consumer
    t_prd = threading.Thread(target=th_producer) # producer
    t_hdl.start()
    t_prd.start()
    t_hdl.join()
    t_prd.join()

#
start_t = time.time()
main4()
end_t = time.time()
print ("SRC FPS = " , fps, "\t total frames = ",total_frames)
print("TGT FPS=",fps_target," output=", counter, " w/ time=", end_t - start_t)

#if __name__ == '__main__':
#    Main.main2()

