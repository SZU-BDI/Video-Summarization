# -*- coding: utf-8 -*-
"""
Created on Thu May 31 14:46:15 2018

@author: Tanveer
"""
# import matplotlib
# matplotlib.use('Agg')

# @ref https://www.kancloud.cn/aollo/aolloopencv
import cv2
import caffe

# from Tkinter import Tk
# from tkFileDialog import askopenfilename

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import time


# Tk().withdraw()# we don't want full GUI, so keeping the root window from appearing
# video_path = askopenfilename(initialdir="E:\Research\Datasets\YouTube Videos dataset 10 categories",
#                            filetypes =(("Video File", "*.mp4"),("Video File","*.avi"),("Video File", "*.flv"),("All Files","*.*")),
#                            title = "Choose a video.")
import sys
argc=len(sys.argv)
if(argc<2):
    print("usage: python $me $input_file")
    exit()
video_path=sys.argv[1]

'''
Loading caffe models at once
'''

proto = "VS-Python/Models/mobilenet_v2_deploy.prototxt"
model = "../d/mobilenet_v2.caffemodel"
caffe.set_mode_gpu()
net = caffe.Net(proto, model, caffe.TEST)

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data',(2, 0, 1))
transformer.set_channel_swap('data', (2, 1, 0))
transformer.set_raw_scale('data', 255)

print ("Caffe Model for Feature Extraction Loaded")
#########################################################################
proto1 = "VS-Python/Models/deploy.prototxt"
model1 = "../d/memnet.caffemodel"
net1 = caffe.Net(proto1, model1, caffe.TEST)

transformer1 = caffe.io.Transformer({'data':net1.blobs['data'].data.shape})
transformer1.set_transpose('data',(2, 0, 1))
transformer1.set_channel_swap('data', (2, 1, 0))
transformer1.set_raw_scale('data', 255)
net1.blobs['data'].reshape(1, 3, 227, 227)


print ("Caffe Model for Memorability Prediction Loaded")

class Main: 
    
    def main():
        #print ("Inside main function")
        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))#getting total number of frames
        fps = int(capture.get(cv2.CAP_PROP_FPS))#getting frame rate
        print ("FPS = " , fps, "\t total frames = ",total_frames)
        m_scores = []
        m_scores.append([])
        m_scores.append([])
        ret, frame1 = capture.read()
        counter = 1
        ttt = 0
        start_t = time.time()
        while(True):
            
#                capture.set(0,counter)
#                ret1, frame1 = capture.read()
#                capture.set(0,counter+1)

            ttt = ttt + 1
            ret2, frame2 = capture.read()
            if ret2 is True:
#            capture.release()

                #shot_segmentation=Shot_segmentation()
                #distt = shot_segmentation.segment(frame1,frame2)
                distt = Shot_segmentation.segment(frame1,frame2) # time 0.5

                print ('Processing ... ', ttt, ', of ', total_frames, 'with distt=',distt)
                if distt >= 40000:#{ different images , 25x4
                    m_scores = np.array(m_scores)
                    [rows,cols] = m_scores.shape
                    
                    if cols > 0:
                        max_index = m_scores[0].argmax()
                        keyframe_number = int(m_scores[1][max_index])
                        keyframe = capture.set(cv2.CAP_PROP_POS_MSEC, keyframe_number)
                        print ("keyframe_number=", keyframe_number, " \t########")
                        temp, keyframe = capture.read()
                        # rough pick
                        pathh = '../d/frame' + str(keyframe_number).rjust(6,"0") + '.png'
                        print ("############## \t 'Writing key frame at" , pathh, "'\t##############", "\a")
                        cv2.imwrite(pathh,keyframe)
                    print ("Different images = " , distt , "\t" , counter)
                    m_scores = []
                    m_scores.append([])
                    m_scores.append([])
                    #}
                else:#{ same images
                    #print ("Similar images= " , distt , "\t" , counter)
                    #memorability_prediction=Memorability_Prediction()
                    #m_value = memorability_prediction.mem_calculation(frame1)
                    m_value = Memorability_Prediction.mem_calculation(frame1)  # time 0.15

    #                scores[index][0] = m_value
                    m_scores[0].append(m_value)
                    m_scores[1].append(counter)
                    #}
                counter = counter + fps
                #counter = counter + 1
                frame1 = frame2
                # end_t = time.time()
                # totall = end_t - start_t
                # print ("Time consumed in main function while = ", totall)
            else:
                end_t = time.time()
                totall = end_t - start_t
                print ("Time consumed in main function while = ", totall)
                break


            #cv2.imshow(video_path,frame2)
            #cv2.waitKey(2)
        
        capture.release()  
#        cv2.destroyAllWindows()

    
#############################################################################
class Shot_segmentation:
    #def segment(self,frame1,frame2):
    def segment(frame1,frame2):
        #print ("Inside segment function")
        start_time1 = time.time()
        #resized_image1 = caffe.io.resize_image(frame1,[224,224])
        #resized_image2 = caffe.io.resize_image(frame2,[224,224])
        resized_image1 = cv2.resize(frame1,(224,224))
        resized_image2 = cv2.resize(frame2,(224,224))
#        transformer.set_mean('data',img_mean)
        net.blobs['data'].reshape(1, 3, 224, 224)
        net.blobs['data'].data[...] = transformer.preprocess('data', resized_image1)
        net.forward()
        features1 = net.blobs['fc7'].data[0].reshape(1,1000)
        features1 = np.array(features1)
        net.blobs['data'].data[...] = transformer.preprocess('data', resized_image2)
        net.forward()
        features2 = net.blobs['fc7'].data[0].reshape(1,1000)
        features2 = np.array(features2)
        
        #t = caclulate_distance(self,features1,features2)
        end_time1 = time.time()
        execution_time1 = end_time1 - start_time1
        print ("*********** \t Execution Time in shot segmentation= ", execution_time1, " secs \t***********")
        return euclidean_distances(features1,features2)
    
class Memorability_Prediction:
    #def mem_calculation(self,frame1):
    def mem_calculation(frame1):
        #print ("Inside mem_calculation function")
        start_time1 = time.time()
        #resized_image = caffe.io.resize_image(frame1,[227,227])
        resized_image = cv2.resize(frame1,(227,227))
        net1.blobs['data'].data[...] = transformer1.preprocess('data', resized_image)
        value = net1.forward()
        value = value['fc8-euclidean']
        end_time1 = time.time()
        execution_time1 = end_time1 - start_time1
        print ("*********** \t Execution Time in Memobarility = ", execution_time1, " secs \t***********")
        return value[0][0]
    

if __name__ == '__main__':
    Main.main()

#class Postprocessing:
#    def histogram_difference():
#        
#        for single_image 
#        
#        
#        hist1 = cv2.calcHist([img2],[0],None,[256],[0,256])
#        hist2 = cv2.calcHist([img2],[0],None,[256],[0,256])
#        distance = cv2.compareHist(hist1,hist2)














