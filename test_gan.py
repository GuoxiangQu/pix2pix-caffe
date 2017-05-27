import os
import time
import scipy
import sys
import cv2
import numpy

sys.path.append("/home/donny/test_data_layer/caffe/python")
import caffe

# model_file = "/home/donny/caffe_copy/caffe_dx/original_TSdata/model_cache/original_TSdata/face_train_test_iter_300000.caffemodel"
# deploy_file = "/home/donny/caffe_copy/caffe_dx/original_TSdata/face_deploy.prototxt"
model_file = "./model_cache/face_gan/face_gan_iter_280000.caffemodel"
deploy_file = "face_deploy.prototxt"
feat_layer = "g_deconv_5"


caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net(deploy_file, model_file, caffe.TEST)

file_list = os.listdir("/home/donny/tools/light_testset/dlib_aligned/donny")
for jpg_file in file_list:
    image_path = "/home/donny/tools/light_testset/dlib_aligned/donny/" + jpg_file
    face_img = cv2.imread(image_path)
    height, width = face_img.shape[:2]
    net.blobs['data'].reshape(1, 3, height, width)
    blob_data = net.blobs['data'].data
    face_img = face_img.astype(numpy.float32, copy = False)
    face_img = (face_img - 127.5)/128;
    blob_data[0,0,:,:] = face_img[:,:,0]
    blob_data[0,1,:,:] = face_img[:,:,1]
    blob_data[0,2,:,:] = face_img[:,:,2]
    out = net.forward()
    result = numpy.copy(out[feat_layer])
    
    print result.shape
    result =  result[0].transpose(1,2,0)
    result = result * 128 + 127.5
    cv2.imwrite("test_val.png", result)
    cv2.imwrite("./data_val/test" + jpg_file, result)
    image = cv2.imread("test_val.png")
    cv2.imshow("main", image)
    cv2.waitKey()
