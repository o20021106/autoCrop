from auto_cropper.models.sam_resnet import sam_resnet
from auto_cropper.models.preprocess import generator
from auto_cropper.models.model_config import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from auto_cropper.models.loss import *
from keras.optimizers import RMSprop, Adam 

model = sam_resnet()
model.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])
model.fit_generator(generator(b_s=b_s), steps_per_epoch = nb_imgs_train, epochs=nb_epoch,
                                validation_data=generator(b_s=b_s, phase_gen='val'), validation_steps = nb_imgs_val,
                                callbacks=[EarlyStopping(patience=3),
                                           ModelCheckpoint('models_files/trained/weights.sam-resnet.{epoch:02d}-{val_loss:.4f}.pkl', save_best_only=True)])
model.save(sam_model_path)
