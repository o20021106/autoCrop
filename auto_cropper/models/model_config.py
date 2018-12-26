WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
RESNET_PATH = ('model_files/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
imgs_train_path = '/root/projects/cropping_svm/training_data/images/'
maps_train_path = '/root/projects/cropping_svm/training_data/maps/train/'
fixs_train_path = '/root/projects/cropping_svm/training_data/fixations/train/'

imgs_val_path = '/root/projects/cropping_svm/training_data/images/'
maps_val_path = '/root/projects/cropping_svm/training_data/maps/val/'
fixs_val_path = '/root/projects/cropping_svm/training_data/fixations/val/'
sam_model_path = 'model_files/trained/sam_model.h5'

b_s = 1
# number of rows of input images
shape_r = 240
# number of cols of input images
shape_c = 320
# number of rows of downsampled maps
shape_r_gt = 30
# number of cols of downsampled maps
shape_c_gt = 40
# number of rows of model outputs
shape_r_out = 480
# number of cols of model outputs
shape_c_out = 640
# final upsampling factor
upsampling_factor = 16
# number of epochs
nb_epoch = 10
# number of timestep
nb_timestep = 4
# number of learned priors
nb_gaussian = 16

nb_imgs_train = 10000
nb_imgs_val = 5000
