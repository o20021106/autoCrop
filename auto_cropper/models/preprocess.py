import cv2
import scipy.io
import numpy as np
import os
from ..models.model_config import *

def preprocess_fixmaps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), 1, shape_r, shape_c))

    for i, path in enumerate(paths):
        gazes = scipy.io.loadmat(path)["gaze"]
        coords = []
        for gaze in gazes:
            coords.extend(gaze[0][2])
        for coord in coords:
            if coord[1] >= 0 and coord[1] < shape_r and coord[0] >= 0 and coord[0] < shape_c:
                ims[i, 0, coord[1], coord[0]] = 1.0
    return ims

def padding(img, shape_r=240, shape_c=320, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)
    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded

def preprocess_maps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), 1, shape_r, shape_c))

    for i, path in enumerate(paths):
        original_map = cv2.imread(path, 0)
        padded_map = padding(original_map, shape_r, shape_c, 1)
        ims[i, 0] = padded_map.astype(np.float32)
        ims[i, 0] /= 255.0
    return ims

def resize_fixation(img, rows=480, cols=640):
    out = np.zeros((rows, cols))
    factor_scale_r = rows / img.shape[0]
    factor_scale_c = cols / img.shape[1]

    coords = np.argwhere(img)
    for coord in coords:
        r = int(np.round(coord[0]*factor_scale_r))
        c = int(np.round(coord[1]*factor_scale_c))
        if r == rows:
            r -= 1
        if c == cols:
            c -= 1
        out[r, c] = 1
    return out

def padding_fixation(img, shape_r=480, shape_c=640):
    img_padded = np.zeros((shape_r, shape_c))

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = resize_fixation(img, rows=shape_r, cols=new_cols)
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = resize_fixation(img, rows=new_rows, cols=shape_c)
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded

def preprocess_fixmaps(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), 1, shape_r, shape_c))

    for i, path in enumerate(paths):
        gazes = scipy.io.loadmat(path)["gaze"]
        coords = []
        for gaze in gazes:
            coords.extend(gaze[0][2])
        for coord in coords:
            if coord[1] >= 0 and coord[1] < shape_r and coord[0] >= 0 and coord[0] < shape_c:
                ims[i, 0, coord[1], coord[0]] = 1.0

    return ims

def preprocess_images(paths, shape_r, shape_c):
    ims = np.zeros((len(paths), shape_r, shape_c, 3))
    for i, path in enumerate(paths):
        original_image = cv2.imread(path,1)
        padded_image = padding(original_image, shape_r, shape_c, 3)
        ims[i] = padded_image
    ims[:, :, :, 0] -= 103.939
    ims[:, :, :, 1] -= 116.779
    ims[:, :, :, 2] -= 123.68

    return ims
def preprocess_image_array(img, shape_r, shape_c):
    padded_image = padding(img, shape_r, shape_c, 3)
    padded_image = np.expand_dims(padded_image, axis = 0).astype(float)
    padded_image[:, :, :, 0] -= 103.939
    padded_image[:, :, :, 1] -= 116.779
    padded_image[:, :, :, 2] -= 123.68
    return padded_image  

def generator(b_s, phase_gen='train'):
    if phase_gen == 'train':   
        image_ids = [ f for f in os.listdir(imgs_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps_ids = [ f.split('.')[0] for f in os.listdir(maps_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs_ids = [ f.split('.')[0] for f in os.listdir(fixs_train_path) if f.endswith(('.mat'))]
        maps_ids.sort()
        fixs_ids.sort()
        images = [imgs_train_path + f for f in image_ids if f.split('.')[0] in fixs_ids]
        maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs = [fixs_train_path + f for f in os.listdir(fixs_train_path) if f.endswith('.mat')]
        
    elif phase_gen == 'val':
        image_ids = [ f for f in os.listdir(imgs_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps_ids = [ f.split('.')[0] for f in os.listdir(maps_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs_ids = [ f.split('.')[0] for f in os.listdir(fixs_val_path) if f.endswith(('.mat'))]
        images = [imgs_val_path + f for f in image_ids if f.split('.')[0] in fixs_ids]
        maps = [maps_val_path + f for f in os.listdir(maps_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs = [fixs_val_path + f for f in os.listdir(fixs_val_path) if f.endswith('.mat')]
    else:
        raise NotImplementedError

    images.sort()
    maps.sort()
    fixs.sort()
    counter = 0
    for i,j,k in zip(images, maps, fixs):
        i = i.split('/')[-1].split('.')[0]
        j = j.split('/')[-1].split('.')[0]
        k = k.split('/')[-1].split('.')[0] 
        if (i!=j) or (j!=k):
            print(i,j,k)
        else:
          counter+=1
    print("match count {}".format(counter))
    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    while True:
        Y = preprocess_maps(maps[counter:counter+b_s], shape_r_out, shape_c_out)
        Y_fix = preprocess_fixmaps(fixs[counter:counter + b_s], shape_r_out, shape_c_out)
        yield preprocess_images(images[counter:counter + b_s], shape_r, shape_c), [Y, Y, Y_fix]
        counter = (counter + b_s) % len(images)
