from keras import layers
from .resnet import dcn_resnet
from .model_config import *
from keras.models import Model

def sam_resnet():
    dcn = dcn_resnet()
    conv_feat = layers.Conv2D(1, (3, 3) , padding = 'same', activation='relu')(dcn.output)
    conv_feat = layers.Permute((3,1,2))(conv_feat)
    conv_feat = layers.UpSampling2D(size=(upsampling_factor, upsampling_factor), 
                                          data_format='channels_first', 
                                          interpolation='bilinear')(conv_feat)
    model = Model(dcn.input, [conv_feat,conv_feat,conv_feat])
    print(model.summary())
    return model
