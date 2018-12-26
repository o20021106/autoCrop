# autoCropper
This repo contains codes for both training the models required for cropping images 
as well as deploying an API for prediciton.

The inspiration of this repo comes from this [twitter's blog](https://blog.twitter.com/engineering/en_us/topics/infrastructure/2018/Smart-Auto-Cropping-of-Images.html) about how twiter crop images for more aestheically pleasing visuals.
[Predicting Human Eye Fixations via an LSTM-based Saliency Attentive Model](https://ieeexplore.ieee.org/document/8400593) provides a deep neural network solution on saliency prediction. I use part of their code on this repo.
It turns out diluting Resnet on itself gives pretty good saliency predictions, therefore I did not use any attention mechanism and center bias mentioned in the paper.
However, saliency based model often cropped too much out of the original picture. Though it correctly picks out parts of the image that the human eyes focus on, it fails to include the parts of the image which are not salient but important for the crop to be aestheically pleasing.

 
[Quantitative Analysis of Automatic Image Cropping Algorithms: A Dataset and Comparative Study](https://arxiv.org/abs/1701.01480) uses a ranking based model, which gives good aesthetic cropping; therefore,  I also trained a model based on this paper.




[GitHub](https://github.com/yiling-chen/flickr-cropping-dataset)

## Train ranking model
```python ranking_model_train.py -f flickr_filename```

## Train saliency model
```saliency_train.py```
# facedetection file
# binary
#rank data
#sam data github
#環境變數binary
