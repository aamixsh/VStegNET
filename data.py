import os
import cv2
import numpy as np
import random

def data():
    path = 'ucf/train'

    c_path = path+'/'+random.choice(os.listdir(path))

    s_path = path+'/'+random.choice(os.listdir(path))
    while s_path == c_path:
        s_path = path+'/'+random.choice(os.listdir(path))

    s = []
    c = []
 
    lc=[]
    ls=[]

    # print (c_path)

    d1 = sorted(os.listdir(c_path), key=lambda x: int(x.split('/')[-1].split('.')[0]))
    l1 = len(d1)

    d2 = sorted(os.listdir(s_path), key=lambda x: int(x.split('/')[-1].split('.')[0]))
    l2 = len(d2)
    # print(d1,d2)

    # print(l1,l2)

    l3 = min(l1,l2)
    # print(l3)

    count =0
    fcount =0

    for i in range(l3):     
        if (fcount <8):
            frame = c_path + '/' + d1[i]
            #print(frame)
            frame = cv2.imread(frame)
            frame = frame/255.0
            frame = frame.reshape((240,320,3))
            # print(frame.shape)
            c.append(frame)
        if len(c)==8:
            lc.append(c)
            c=[]
            fcount =0

        count = count+1
        
   
    count=0
    fcount =0

    for i in range(l3):        
        if (fcount<8):
            frame = s_path + '/' + d2[i]
            #print(frame)
            frame = cv2.imread(frame)

            # scale_percent = 50
            # w = int(frame.shape[0] * scale_percent /100)
            # h = int(frame.shape[1] * scale_percent /100)
            # dim = (w,h)
            
            # perform the actual resizing of the image and show it
            # resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            #print("resized shape :",resized.shape)
            
            frame = frame / 255.0
            frame = frame.reshape((240, 320, 3))
            s.append(frame)
        if len(s)==8:
            ls.append(s)
            s=[]
            fcount =0

        count = count+1
        
    
    sl = np.asarray(ls)
    cl = np.asarray(lc)
    # print(s1.shape, cl.shape)

    return cl, sl

def test_data():
    path = 'ucf/test'

    c_path = path+'/'+random.choice(os.listdir(path))
    c_name = c_path.split('/')[-1]

    s_path = path+'/'+random.choice(os.listdir(path))
    while s_path == c_path:
        s_path = path+'/'+random.choice(os.listdir(path))
    s_name = s_path.split('/')[-1]

    s = []
    c = []
 
    lc=[]
    ls=[]

    d1 = sorted(os.listdir(c_path), key=lambda x: int(x.split('/')[-1].split('.')[0]))
    l1 = len(d1)

    d2 = sorted(os.listdir(s_path), key=lambda x: int(x.split('/')[-1].split('.')[0]))
    l2 = len(d2)
    # print(d1,d2)

    # print(l1,l2)

    l3 = min(l1,l2)
    # print(l3)

    count =0
    fcount =0

    for i in range(l3):     
        if (fcount <8):
            frame = c_path + '/' + d1[i]
            #print(frame)
            frame = cv2.imread(frame)
            frame = frame/255.0
            frame = frame.reshape((240,320,3))
            # print(frame.shape)
            c.append(frame)
        if len(c)==8:
            lc.append(c)
            c=[]
            fcount =0

        count = count+1
        
   
    count=0
    fcount =0

    for i in range(l3):        
        if (fcount<8):
            frame = s_path + '/' + d2[i]
            #print(frame)
            frame = cv2.imread(frame)

            # scale_percent = 50
            # w = int(frame.shape[0] * scale_percent /100)
            # h = int(frame.shape[1] * scale_percent /100)
            # dim = (w,h)
            
            # perform the actual resizing of the image and show it
            # resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            #print("resized shape :",resized.shape)
            
            frame = frame / 255.0
            frame = frame.reshape((240, 320, 3))
            s.append(frame)
        if len(s)==8:
            ls.append(s)
            s=[]
            fcount =0

        count = count+1
        
    
    sl = np.asarray(ls)
    cl = np.asarray(lc)
    # print(sl.shape, cl.shape)

    return cl, sl, c_name, s_name

# data()
