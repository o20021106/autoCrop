# autoCropper
This repo contains codes for both training the models required for cropping images 
and deploying an API for prediciton.

The inspiration of this repo comes from this [twitter's blog](https://blog.twitter.com/engineering/en_us/topics/infrastructure/2018/Smart-Auto-Cropping-of-Images.html) about how twiter crops images for more aestheically pleasing visuals.
[Predicting Human Eye Fixations via an LSTM-based Saliency Attentive Model](https://ieeexplore.ieee.org/document/8400593) provides a deep neural network solution on saliency prediction. I use part of their code on this repo (Please refer to [their repo](https://github.com/marcellacornia/sam)).
It turns out, diluting Resnet on itself gives pretty good saliency predictions, hence I did not use any attention mechanism and center bias mentioned in the paper.
However, saliency based model often cropped too much out of the original picture. Though it correctly picks out parts of the image that the human eyes focus on, it fails to include the parts of the image which are not salient but important for the crop to be aestheically pleasing.

 
[Quantitative Analysis of Automatic Image Cropping Algorithms: A Dataset and Comparative Study](https://arxiv.org/abs/1701.01480) uses a ranking based model, which gives good aesthetic croppings; therefore,  I also trained a [SVM-rank](https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html) using method proposed by this paper. Instead of AlexNet as suggested in the paper, I use VGG for feature extraction for it's superior performance on image categorization task, in the hope of a better transfer learning result.

The last bits of this cropper utilizes open-cv's face detection module. This is a deep neural network for facedetection, and we can use it to check whether a crop crops out a face.

The overall process of prediction thus goes as follows: 
1. A saliency model is used to predict saliency for each pixel.
2. A face detection model finds whether there are faces in the picture, and where they are.
3. Cropping candidates are proposed by including crops that has enough saliency sum and does not crop out certain faces.
4. Cropoping candidates are passed to a ranking model, and the crop with the highest ranking score is selected as the final prediction.

# Traing

### Download training data
1. Go to [SALCON](http://salicon.net/challenge-2017/)'s website, and download the saliency data. Then copy and paste the related paths to ```auto_crop/models/model_config.py```
2. Refer to [Yi-Ling Chen](https://github.com/yiling-chen/flickr-cropping-dataset)'s github for downaloading data for ranking model training.

### Download SVM-rank
Follow the instructions on [this page](https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html) on how for downloading and building SVM-rank.
Append the path to the binary to environment. 
 
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


| Saliency Map | Bounding Box | Crop |
| ------------ | ------------ | ---- |
| <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/1_s.jpg?token=ASVZAYap2lt0k5qIqRqGjuTZuFMT1dTsks5cLbQmwA%3D%3D"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/1_b.jpg?token=ASVZATYxAOQfvh3lGyfnzaL4JidPk_rNks5cLbRnwA%3D%3D"> | <img src="https://raw.githubusercontent.com/o20021106/autoCrop/master/data/images/1.jpg?token=ASVZAUhzlz26xvAb7E2kVWF8yyHMUJ1xks5cLbSpwA%3D%3D"> |

