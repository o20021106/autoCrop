from .create_app_celery import create_celery_app
from .classes.cropper import Cropper, CroppingModels
import logging
import cv2
import numpy as np
from base64 import b64decode
import pandas as pd
import cv2
import os
import logging

celery = create_celery_app()

cropping_models = None


@celery.task()
def test_delay():
    import time
    time.sleep(2)
    logging.info('hey hey')
    return 'hey hey'


@celery.task()
def crop(img, cropped_filename, file_extention, cropped_dir, ratio):
    global cropping_models
    if cropping_models is None:
        logging.info('Initializing cropping models.')
        cropping_models = CroppingModels()
        logging.info('Finished initializeing cropping models.')
    
    cropper = Cropper(cropping_models, ratio)
    img = np.array(img, dtype =np.uint8)
    crop_coords, features = cropper.get_features(img)
    predictions = cropper.rank_crops()
    predictions = pd.DataFrame({'rank': predictions, 
                                'crop_coords': crop_coords}).sort_values(by = 'rank', ascending = False)
    x_start, y_start, x_end, y_end = predictions['crop_coords'].iloc[0]
    img_cropped = img[y_start:y_end, x_start:x_end]
    cv2.imwrite(os.path.join(os.getcwd(), cropped_dir, f'{cropped_filename}.{file_extention}'), img_cropped)
    s_max = cropper.saliency.max()
    cropper.saliency = cropper.saliency*(255/s_max)
    cv2.imwrite(os.path.join(os.getcwd(), cropped_dir, f'{cropped_filename}_s.{file_extention}'), cropper.saliency)
    cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(os.getcwd(), cropped_dir, f'{cropped_filename}_b.{file_extention}'), img)

