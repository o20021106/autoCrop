from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import cv2
import os 
import numpy as np
import json
import logging
import logging.config
from argparse import ArgumentParser
import subprocess

logging.config.fileConfig(fname='base_logger.conf', disable_existing_loggers=False)
logger = logging.getLogger('training')

def get_image_feature(flickr_id, crop_1, crop_0):
    img_name = [i for i in img_names if i.startswith(str(flickr_id))]
    if len(img_name) == 1:
        img_file_path = os.path.join(img_dir, img_name[0])
        img = cv2.imread(img_file_path)
        img_0 = img[crop_0[1]:crop_0[1]+crop_0[3],crop_0[0]:crop_0[0]+crop_0[2]]
        img_1 = img[crop_1[1]:crop_1[1]+crop_1[3],crop_1[0]:crop_1[0]+crop_1[2]]
        img_0 = cv2.resize(img_0, (224,224))
        img_1 = cv2.resize(img_1, (224,224))
        img_0 = preprocess_input(np.expand_dims(img_0, axis=0))
        img_1 = preprocess_input(np.expand_dims(img_1, axis=0))
        feature_0 = model.predict(img_0)
        feature_1 = model.predict(img_1)
        return feature_0, feature_1

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', '--flicker_dir', help='optional argument', dest='flickr_dir', required=True)

    args = parser.parse_args()
    flickr_dir = args.flickr_dir
    img_dir = os.path.join(flickr_dir, 'data_ranking')
    annotation_path = os.path.join(flickr_dir, 'ranking_annotation.json')
    
    with open(os.path.join(flickr_dir, 'remove_data_ranking.dat'), 'r') as f:
        remove_names = f.read().splitlines()
    remove_names = [i.split('_')[0] for i in remove_names]

    img_names = os.listdir(img_dir)
    img_names_clean = [i.split('_')[0] for i in img_names]

    qid_counter = 1
    model = VGG19()
    model = Model(model.input, model.get_layer('fc2').output)

    with open(annotation_path) as f:
        annotation = json.load(f)

    with open('data/training/train.dat', 'a') as f:
        for annot in annotation:
            if str(annot['flickr_photo_id']) in remove_names:
                continue
            if str(annot['flickr_photo_id']) not in img_names_clean:
                continue
            for pair in annot['crops']:
                feature_0, feature_1 = get_image_feature(annot['flickr_photo_id'], 
                                                         pair['crop_1'], pair['crop_0'])
                vote_1 = f'{pair["vote_for_1"]} qid:{qid_counter}'
                for i in range(feature_1.shape[-1]):
                    value = feature_1[0,i]
                    if value == 0:
                        continue
                    else:
                        vote_1 = f'{vote_1} {i+1}:{value}'
                f.write(f'{vote_1}\n')

                vote_0 = f'{pair["vote_for_0"]} qid:{qid_counter}'
                for i in range(feature_0.shape[-1]):
                    value = feature_0[0,i]
                    if value == 0:
                        continue
                    else:
                        vote_0 = f'{vote_0} {i+1}:{value}'
                f.write(f'{vote_0}\n')
                qid_counter += 1 
            break
    completed = subprocess.run(['svm_rank_learn', '-c', '20.0', 'data/training/train.dat', 'model_files/trained/rank_model.dat'])
    logger.info(f'returncode: {completed.returncode}')
    
