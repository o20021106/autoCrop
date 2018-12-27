from flask import Blueprint, request, render_template, flash, redirect, url_for, send_from_directory, jsonify
from flask import current_app as app
from werkzeug.utils import secure_filename
import logging
import os
import numpy as np
import cv2
from math import floor
from base64 import b64encode
from datetime import datetime
import logging
import os
from urllib.parse import urlparse

RESIZE_WIDTH = int(os.environ.get('RESIZE_WIDTH'))
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

cp = Blueprint('cropping', __name__)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@cp.route('/cropped/<filename>')
def cropped_file(filename):
    uploads = os.path.join(os.getcwd(), app.config['CROPPED_FOLDER'])
    return send_from_directory(uploads, filename)


@cp.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the form part
        if ('width' not in request.form) or ('height' not in request.form):
            logging.info('no width or height')
            return jsonify({'error': 'Width or height missing.'})
        # check if the post request has the file part
        if 'pic' not in request.files:
            logging.inf('no file')
            return jsonify({'error': 'No file part.'})  
      
        pic = request.files['pic']
        width = request.form['width']
        height = request.form['height']

        # check if a file is sent.
        if pic.filename == '':
            logging.info('Return error msg: No file selected')
            return jsonify({'error': 'No selected file.'})
        # convert width and height into intergers.
        try:
            width = int(width)
            height = int(height)
            logging.info(f'(width, height): {(width, height)}')
            if width*height == 0:
                logging.info(f'Return error msg becuase zero width or height')
                return jsonify({'error': 'Width and height must both be greater than zero.'})
        except:
            logging.exception('Retrun error msg: Invalid width height')
            return jsonify({'error': 'Invalid width or height.'})

        # check if file extention is allowed.
        if not allowed_file(pic.filename):
            logging.info('Invalid file type.')
            return jsonify({'error': 'Invalid file type.'})
        if pic:
            filename = secure_filename(pic.filename)
            logging.debug(f'filename: {filename}')
            from ..tasks import crop
            filestr = pic.read()
            imgbs64 = b64encode(filestr).decode("utf-8") 
            npimg = np.fromstring(filestr, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            w = img.shape[1]
            h = img.shape[0]
            
            h = RESIZE_WIDTH*(h/w)
            img = cv2.resize(img, (RESIZE_WIDTH, floor(h)))
            if filename.lower() in ALLOWED_EXTENSIONS:
                filename = '.'+filename
            cropped_filename = f'cropped_{datetime.now().timestamp()}_'+filename
            cropped_filename_split = cropped_filename.rsplit('.', 1)
            cropped_filename, file_extention = cropped_filename_split[0], cropped_filename_split[1].lower()
            logging.debug(f'cropped_filename: {cropped_filename}, file_extention: {file_extention}')
            img = crop.delay(img.tolist(), cropped_filename, file_extention,
                             app.config['CROPPED_FOLDER'], (width,height))
            host = os.environ.get('HOST_IP')
            scheme = urlparse(request.url).scheme
            port = ':'+os.environ.get('PORT') if os.environ.get('FOREMAN') == 'True' else ''         
            return jsonify({'imgURLS':f'http://{host}{port}/cropped/{cropped_filename}_s.{file_extention}',
                            'imgURLB':f'http://{host}{port}/cropped/{cropped_filename}_b.{file_extention}',
                            'imgURL':f'http://{host}{port}/cropped/{cropped_filename}.{file_extention}'})

    return render_template('upload.html')

