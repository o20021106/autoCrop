from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import cv2
import os 
import numpy as np
import pandas as pd
import json
import logging
import logging.config 
from math import floor 
from ..models.sam_resnet import sam_resnet
from keras.models import load_model
from ..models.loss import kl_divergence, correlation_coefficient, nss
from ..models.preprocess import preprocess_image_array
from ..models.model_config import *
import subprocess


logging.config.fileConfig(fname='base_logger.conf', disable_existing_loggers=False,
                          defaults={'logfilename': 'logs/prediction.log'})
logger = logging.getLogger('simpleExample')


class Cropper:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe('model_files/facedetection/deploy.prototxt.txt',
                                            'model_files/facedetection/res10_300x300_ssd_iter_140000.caffemodel')
        model = VGG19()
        self.vgg_model = Model(model.input, model.get_layer('fc2').output)
        self.sam_model = load_model(sam_model_path, 
                                    custom_objects={'kl_divergence': kl_divergence, 
                                                    'correlation_coefficient':correlation_coefficient, 
                                                    'nss':nss})


    def _face_detection(self, img):
        '''
        face detection
        input:
            image - 3D numpy array
        return:
            img - 3D numpy array. The same image but with faces outlined
            faces - list. top-left and bottm-right corners of face bon
        '''
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300, 300), (103.93, 116.77, 123.68))
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.9:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX, endY))
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                #cv2.rectangle(img, (startX, startY), (endX, endY),
                #              (0, 0, 255), 2)
                #cv2.putText(img, text, (startX, y),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        return img, faces


    def _translate_to_original(self, saliency, img, shape_r, shape_c):
        '''
        input:
            saliency - 
        '''
        s_h, s_w = saliency.shape
        h, w, _ = img.shape
        rows_rate = h/shape_r
        cols_rate = w/shape_c
        if rows_rate > cols_rate:
            new_cols = (w*s_h)//h
            y_start, x_start = (0, (s_w-new_cols)//2)
            y_end, x_end = (s_h, x_start+w)
        else:
            new_rows = (h*s_w)//w
            y_start, x_start = ((s_h-new_rows)//2, 0)
            y_end, x_end = (y_start+s_h, s_w)
        saliency = saliency[y_start:y_end, x_start:x_end]
        return cv2.resize(saliency, (w, h))


    def _get_saliency(self, img):
        '''
        input: 
           img - 3D numpy array
        return:
           saliency map - 2D numpy array
        '''
        preprocessed_img = preprocess_image_array(img, shape_r, shape_c)
        saliency = self.sam_model.predict(preprocessed_img)
        saliency = saliency[0][0][0]
        logger.info(f'saliency map shape: {saliency.shape}')
        saliency = self._translate_to_original(saliency, img, shape_r, shape_c)
        logger.info(f'saliency map shape(traslated into original size): {saliency.shape}')
        return saliency        


    def _get_crops(self, img, ratio = (4,3), num_grid = 5, scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1], 
                   saliency_threshold = 0.1, saliency_rate = 0.5, face_crossed_tolerance = 0.1):
        '''
        input:
           img - 3D numpy array
           scales - list. The scales of crops to use
           num_grid - int.
        return:
           crop_coords - list of tuples. top-left and right-bottom corners of candidate windows. 
        '''
        _, faces = self._face_detection(img)
        saliency = self._get_saliency(img)
        saliency = np.where(saliency > saliency_threshold, 1, 0)
        saliency_sum = saliency.sum()
        crop_coords = []
        crops_data = []
        step_base = 'col'
        r = img.shape[1]/img.shape[0]
        target_ratio = ratio[0]/ratio[1]
        if r > target_ratio:
            step_base = 'row'
        logger.info(f'The shape of the image: {img.shape}')

        for scale in scales:
            if step_base == 'row':
                logger.info(f'More steps for X axis') 
                h = floor(img.shape[0]*scale)
                w = floor(h*target_ratio)
                num_y_steps = num_grid if scale < 1 else 1
                y_step = floor((img.shape[0]-h)/num_y_steps)
                num_x_steps = 10
                x_step = floor((img.shape[1]-w)/num_x_steps)
            else:
                logger.info(f'More steps for Y axis') 
                w = floor(img.shape[1]*scale)
                h = floor(w/target_ratio)
                num_x_steps = num_grid if scale < 1 else 1
                x_step = floor((img.shape[1]-w)/num_x_steps)
                num_y_steps = 10
                y_step = floor((img.shape[0]-h)/num_y_steps)

            for i in range(num_x_steps):
                for j in range(num_y_steps):
                    x_start = i*x_step
                    x_end = i*x_step+w
                    y_start = j*y_step
                    y_end = j*y_step+h
                    not_crossed = True
                    salient = True
                    saliency_retained = saliency[y_start:y_end, x_start:x_end].sum()
                    saliency_retained_rate = saliency_retained/saliency_sum
                    if len(faces) > 0:
                        (startX, startY, endX, endY) = faces[0]
                        cross = (x_start > startX) or (x_end < endX) or (y_start > max(0, startY - (endY-startY)*0.2)) or (y_end < endY)
                        not_crossed = not cross
                    if saliency_retained_rate < saliency_rate:
                        salient = False
                    crops_data.append({'crop_coords': (x_start, y_start, x_end, y_end), 'not_crossed': not_crossed, 'salient': salient, 
                                                    'saliency_retained': saliency_retained, 'saliency_retained_rate': saliency_retained_rate})
        crops_data = pd.DataFrame(crops_data)
        use_not_crossed = crops_data['not_crossed'].mean() > face_crossed_tolerance
        crops_data['valid'] = crops_data['not_crossed'] & crops_data['salient'] 
        if crops_data['valid'].sum() > 0 & use_not_crossed:
            logger.info('return crop candidates with no face crossed and enough saliency')
            return crops_data[crops_data['valid']]['crop_coords'].tolist() 
        elif crops_data['not_crossed'].sum() > 0 & use_not_crossed:
            logger.info('return crop candidates with no faces crossed but not enough saliency')
            return crops_data[crops_data['not_crossed']]['crop_coords'].tolist()
        elif crops_data['salient'].sum() > 0:
            logger.info('return crop candidates with enough saliency but faces crossed.')
            return crops_data[crops_data['salient']]['crop_coords'].tolist()
        else:
            logger.info('return all candidates')
            return crops_data['crop_coords'].tolist()


    def _output_to_svm_rank_format(self, features):
        with open(f"data/svm_rank_formatted/svm_rannk_data.dat", 'w') as f:
            print(features.shape)
            for feature in features:
                record = '1 qid:1'
                for i in range(feature.shape[-1]):
                    value = feature[i]
                    if value == 0:
                        continue
                    else:
                        record = f"{record} {i+1}:{value}"
                f.write(f"{record}\n")


    def get_features(self, img):
        '''
        input:
           img - 3D numpy array
        return:
           features - 4D numpy array (# of crops, height, width, channel)
        '''
        crop_coords = self._get_crops(img)
        crops = [] 
        for crop_coord in crop_coords:
            x_start, y_start, x_end, y_end = crop_coord
            crop = img[y_start:y_end, x_start:x_end]   
            crop = cv2.resize(crop, (224,224))    
            crops.append(crop)
        crops = np.array(crops)
        crops = preprocess_input(crops)
        logger.info(f'The shape of crops(# of crops, height, width, channel):{ crops.shape}')
        features = self.vgg_model.predict(crops)   
        self._output_to_svm_rank_format(features)
        return crop_coords, features


    def rank_crops(self):
        completed = subprocess.run(['svm_rank_classify', 'data/svm_rank_formatted/svm_rannk_data.dat', 
                                    'model_files/trained/rank_model.dat', 'data/predictions_svm_rank/prediction'])
        logger.info(f'returncode: {completed.returncode}')
        with open('data/predictions_svm_rank/prediction', 'r') as f:
            predictions = f.read().splitlines()
        predictions = [float(prediction) for prediction in predictions]
        return predictions

