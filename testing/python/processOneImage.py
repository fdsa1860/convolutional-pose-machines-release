def processOneImage(test_image):
    
    import cv2 as cv 
    import numpy as np
    import scipy
    import scipy.io as sio
    import math
    import caffe
    import time
    from config_reader import config_reader
    import util
    import copy
    
#     test_image = '../sample_image/7.jpg'
    
    oriImg = cv.imread(test_image) # B,G,R order
    # cv.imshow('my window',oriImg)
    
    param, model = config_reader()
    boxsize = model['boxsize']
    npart = model['np']
    scale = boxsize/(oriImg.shape[0] * 1.0)
    imageToTest = cv.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
    
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest)
    # cv.imshow("window", imageToTest_padded)
    print(imageToTest_padded.shape, pad)
    
    print(model['deployFile_person'])
    print(model['caffemodel_person'])
    
    if param['use_gpu']: 
        caffe.set_mode_gpu()
        caffe.set_device(param['GPUdeviceNumber']) # set to your device!
    else:
        caffe.set_mode_cpu()
    person_net = caffe.Net(model['deployFile_person'], model['caffemodel_person'], caffe.TEST)
    person_net.blobs['image'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
    person_net.reshape()
    person_net.forward(); # dry run to avoid GPU synchronization later in caffe
    
    person_net.blobs['image'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
    start_time = time.time()
    output_blobs = person_net.forward()
    print('Person net took %.2f ms.' % (1000 * (time.time() - start_time)))
    print(output_blobs.keys())
    print(output_blobs[output_blobs.keys()[0]].shape)
    person_map = np.squeeze(person_net.blobs[output_blobs.keys()[0]].data)
#     cv.imshow('win',person_map * 256)
    #cv.waitKey(0)
    
    person_map_resized = cv.resize(person_map, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
    data_max = scipy.ndimage.filters.maximum_filter(person_map_resized, 3)
    maxima = (person_map_resized == data_max)
    diff = (data_max > 0.5)
    maxima[diff == 0] = 0
    x = np.nonzero(maxima)[1]
    y = np.nonzero(maxima)[0]
    
    # print(x, y)
    # cv.imshow('win',person_map_resized * 256)
    #cv.waitKey(0)
    
    person_map_to_plot_1 = util.colorize(person_map_resized) * 0.5 + imageToTest_padded * 0.5
    person_map_to_plot_2 = person_map_resized * 255
    for x_c, y_c in zip(x, y):
        cv.circle(person_map_to_plot_2, (x_c, y_c), 3, (0,0,255), -1)
    # cv.imshow('win', np.concatenate((person_map_to_plot_1, np.tile(person_map_to_plot_2[:,:,np.newaxis], (1,1,3))), axis=1))
    
    num_people = x.size
    person_image = np.ones((model['boxsize'], model['boxsize'], 3, num_people)) * 128
    for p in range(num_people):
        for x_p in range(model['boxsize']):
            for y_p in range(model['boxsize']):
                x_i = x_p - model['boxsize']/2 + x[p]
                y_i = y_p - model['boxsize']/2 + y[p]
                if x_i >= 0 and x_i < imageToTest.shape[1] and y_i >= 0 and y_i < imageToTest.shape[0]:
                    person_image[y_p, x_p, :, p] = imageToTest[y_i, x_i, :]
    # show one of them for inspection
    # cv.imshow('win', person_image[:,:,:,0])
    
    gaussian_map = np.zeros((model['boxsize'], model['boxsize']))
    for x_p in range(model['boxsize']):
        for y_p in range(model['boxsize']):
            dist_sq = (x_p - model['boxsize']/2) * (x_p - model['boxsize']/2) + \
                      (y_p - model['boxsize']/2) * (y_p - model['boxsize']/2)
            exponent = dist_sq / 2.0 / model['sigma'] / model['sigma']
            gaussian_map[y_p, x_p] = math.exp(-exponent)
    # cv.imshow('win', gaussian_map * 256)
    # #cv.waitKey(0)
    
    pose_net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)
    pose_net.forward() # dry run to avoid GPU synchronization later in caffe
    output_blobs_array = [dict() for dummy in range(num_people)]
    for p in range(num_people):
        input_4ch = np.ones((model['boxsize'], model['boxsize'], 4))
        input_4ch[:,:,0:3] = person_image[:,:,:,p]/256.0 - 0.5 # normalize to [-0.5, 0.5]
        input_4ch[:,:,3] = gaussian_map
        pose_net.blobs['data'].data[...] = np.transpose(np.float32(input_4ch[:,:,:,np.newaxis]), (3,2,0,1))
        start_time = time.time()
        output_blobs_array[p] = copy.deepcopy(pose_net.forward()['Mconv7_stage6'])
        print('For person %d, pose net took %.2f ms.' % (p, 1000 * (time.time() - start_time)))
        
    prediction = np.zeros((14, 2, num_people))
    for p in range(num_people):
        for part in range(14):
            part_map = output_blobs_array[p][0, part, :, :]
            part_map_resized = cv.resize(part_map, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
            prediction[part,:,p] = np.unravel_index(part_map_resized.argmax(), part_map_resized.shape)
        # mapped back on full image
        prediction[:,0,p] = prediction[:,0,p] - (model['boxsize']/2) + y[p]
        prediction[:,1,p] = prediction[:,1,p] - (model['boxsize']/2) + x[p]
    
    prediction = prediction / scale
    prediction = np.fliplr(prediction)
    return prediction
    # print(prediction)
#     sio.savemat('pred',{'prediction':prediction})
    
