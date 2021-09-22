from keras.applications import densenet
from keras.preprocessing import image
from keras.applications.densenet import preprocess_input, decode_predictions
import numpy as np
from keras.layers import Dense,Dropout, GlobalAveragePooling2D,Conv2D,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import os
from keras.callbacks import ModelCheckpoint
import json
from keras import regularizers
from tensorflow import keras

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

from google.colab import files

train_data_path = '/content/drive/MyDrive/Colab Notebooks/COVID_CT/brazil_hospital/training'
validation_data_path = '/content/drive/MyDrive/Colab Notebooks/COVID_CT/brazil_hospital/validation/'


img_width, img_height = 224,224
batch_size = 64
classes_num = 2
validation_steps = 100


base_model = densenet.DenseNet121(weights='/content/drive/MyDrive/Colab Notebooks/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)

# Merancang Arsitektur 
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, kernel_regularizer = regularizers.l1_l2(0.01),activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(64, kernel_regularizer = regularizers.l1_l2(0.01),activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(32, kernel_regularizer = regularizers.l1_l2(0.01),activation='relu')(x)
x = Dropout(0.3)(x)
preds = Dense(classes_num, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)


# cetak arsitektur 
print(len(model.layers))
print(model.summary())

# mengatur parameter yang tidak akan diubah (freeze) dan diubah (unfreeze)
n_freeze = 300

for layer in model.layers[:n_freeze]:
    layer.trainable=False

for layer in model.layers[n_freeze:]:
    layer.trainable=True

    


opt = keras.optimizers.Adam(learning_rate=0.01)    

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


# menyimapan model terbaik sebagai model baru 
filepath = "/content/drive/MyDrive/Colab Notebooks/Pretrain.densenet.hdf5"

metric = 'val_accuracy'
checkpoint = ModelCheckpoint(filepath, monitor=metric, verbose=2, save_best_only=True, mode='max')
callbacks_list = [checkpoint]



# Proses Augmentasi data latih 
train_datagen = ImageDataGenerator(

    rescale=1. / 255,
    #rescale=None,
    #shear_range=0.2,
    #zoom_range=0.2,
    rotation_range=.3,
    #width_shift_range=.15,
    #height_shift_range=.15,
    vertical_flip=True,
    horizontal_flip=True)

# data validasi 
test_datagen = ImageDataGenerator(rescale=1./255) 

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')
  

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


step_size_train = train_generator.n//train_generator.batch_size
#step_size_train = 100
print(step_size_train)

history = model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, epochs=5, 
                    validation_data=validation_generator,callbacks=callbacks_list,
                    validation_steps=validation_steps)


print(history.history.keys())

with open('/content/drive/MyDrive/Colab Notebooks/proses.json', 'w') as f:
    json.dump(history.history, f)

#save model akhir

model.save('/content/drive/MyDrive/Colab Notebooks/arsiktektur_densenet121.h5')

model.save_weights('/content/drive/MyDrive/Colab Notebooks/new_weights_densenet121.h5')
