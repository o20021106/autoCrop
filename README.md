# autoCropper
This repo contains codes for both training the models required for cropping images
and deploying an API for prediction.

The inspiration for this repo comes from this [twitter's blog post](https://blog.twitter.com/engineering/en_us/topics/infrastructure/2018/Smart-Auto-Cropping-of-Images.html) about how Twitter crops images for more aesthetically pleasing visuals.
[Cornia et al.(2018)](https://ieeexplore.ieee.org/document/8400593) devise a deep convolution network conbined with LSTM attention mechanisms to achieve good saliency predictions. I use part of their code on this repo (Please refer to [their repo](https://github.com/marcellacornia/sam)).
It turns out, diluting Resnet in and of itself gives pretty good saliency predictions, hence I did not use any attention mechanism or trainable center bias layer mentioned in the paper.
However, saliency-based models often cropped too much out of the original picture. Though thery correctly pick out parts of the image that the human eyes focus on, they fail to include the parts of the images which are not salient but important for the crops to be aesthetically pleasing.


[Chen et al.(2017)](https://arxiv.org/abs/1701.01480) use a ranking based model for automatic image cropping, which gives good aesthetic croppings; therefore,  I also trained a [SVM-rank](https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html) model using feature extraction method proposed by this paper. Instead of AlexNet as suggested in the paper, I use VGG for feature extraction for its superior performance on image categorization task, in the hope of a better transfer learning result.

The last bits of this cropper utilizes OpenCV's face detection module. This is a deep neural network solution for face detection, and we can use it to check whether a crop crops out a face.

The overall process of prediction thus goes as follows:
1. A saliency model is used to predict saliency for each pixel.
2. A face detection model finds whether there are faces in the picture, and where they are.
3. Cropping candidates are proposed by including crops that has enough saliency sum and does not crop out certain faces.
4. Cropping candidates are passed to a ranking model, and the crop with the highest ranking score is selected as the final prediction.

# Traing

### Download training data
1. Go to [SALCON](http://salicon.net/challenge-2017/)'s website, and download the saliency data. Then copy and paste the related paths to ```auto_crop/models/model_config.py```
2. Refer to [Yi-Ling Chen](https://github.com/yiling-chen/flickr-cropping-dataset)'s github for downloading data for ranking model training.

### Download SVM-rank
Follow the instructions on [this page](https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html) on how for downloading and building SVM-rank.
Append the path to the binary to PATH.

### Train ranking model
```$ python ranking_model_train.py -f flickr_dir```

Change filckr_dir to the path to your clone of flickr-cropping-dataset.

### Train saliency model
```$ python saliency_train.py```


# Prediction API

You try a demo of this API [here](http://35.229.246.22/).

1. Install and start NGINX. Replace nginx.conf with nginx.conf in this repo.
2. Pull docker from ```$ sudo docker pull o20021106/autocrop:latest```
3. Create directories for docker volumes

```
$ sudo mkdir /var/log/autoCrop_web/

$ sudo mkdir /var/log/autoCrop_worker/

$ sudo mkdir /usr/temp/cropped
```
4. Run docker

```
$ sudo docker swarm init

$ sudo docker stack deploy -c docker/docker-compose.yml autoCrop
```
5. Go to http://your-host-name/ to test the API.

#Demo


| Ratio | Saliency Map | Bounding Box | Crop |
| ----- |------------ | ------------ | ---- |
| 1:1 | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/1_s.jpg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/1_b.jpg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/1.jpg"> |
| 1:1 | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/3_s.jpeg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/3_b.jpeg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/3.jpeg"> |
| 1:1 | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/4_s.jpeg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/4_b.jpeg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/4.jpeg"> |
| 4:3 | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/2_4to3_s.jpeg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/2_4to3_b.jpeg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/2_4to3.jpeg"> |
| 4:3 | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/7_4to3_s.jpg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/7_4to3_b.jpg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/7_4to3.jpg"> |
| 4:3 | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/10_4to3_s.jpeg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/10_4to3_b.jpeg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/10_4to3.jpeg"> |
| 9:16 | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/8_9to16_s.jpeg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/8_9to16_b.jpeg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/8_9to16.jpeg"> |
| 9:16 | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/9_9to16_s.jpeg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/9_9to16_b.jpeg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/9_9to16.jpeg"> |
| 16:9 | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/6_16to9_s.jpeg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/6_16to9_b.jpeg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/6_16to9.jpeg"> |
| 3:4 | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/5_3to4_s.jpg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/5_3to4_b.jpg"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/5_3to4.jpg"> |


