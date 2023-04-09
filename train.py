import os, time, json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
#from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN

from datetime import datetime
import time


PARAM = {
    'workspace': 'my_model03',
    'dataset': {
        'dir_trainset': 'dataset/dogs-vs-cats-small/trainset',
        'dir_validset': 'dataset/dogs-vs-cats-small/validset',
        'dir_testset': 'dataset/dogs-vs-cats-small/testset'
    },
    'training': {
        'batch_size': 32,
        'img_size': (224, 224),
        'epochs': 2,
        'learning_rate':.0001
    },
    'data_augmentation': {
        'rotation_range': 20,
        'shear_range': 0.2,
        'horizontal_flip': True
    }
}

# Create workspace directories
workspace_path = PARAM['workspace']
os.makedirs(workspace_path, exist_ok=True)
os.makedirs(os.path.join(workspace_path, 'logs'), exist_ok=True)
os.makedirs(os.path.join(workspace_path, 'checkpoints'), exist_ok=True)



## Dataprep stage
# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=PARAM['data_augmentation']['rotation_range'],
    shear_range=PARAM['data_augmentation']['shear_range'],
    horizontal_flip=PARAM['data_augmentation']['horizontal_flip'],
    preprocessing_function= applications.resnet.preprocess_input 
)

valid_datagen = ImageDataGenerator(preprocessing_function=applications.resnet.preprocess_input)

# Data loading
train_generator = train_datagen.flow_from_directory(
    PARAM['dataset']['dir_trainset'],
    target_size=PARAM['training']['img_size'],
    batch_size=PARAM['training']['batch_size'],
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    PARAM['dataset']['dir_validset'],
    target_size=PARAM['training']['img_size'],
    batch_size=PARAM['training']['batch_size'],
    class_mode='categorical'
)

# Determine the number of classes
num_classes = len(train_generator.class_indices)


## prompt: Modelling stage
def create_model(base_model, num_classes):
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Load the base model (ResNet50 in this case)
base_model = applications.resnet50.ResNet50(
    include_top=False, 
    input_shape=(*PARAM['training']['img_size'], 3), 
    weights='imagenet')

# Create the model
model = create_model(base_model, num_classes)

# Make sure to set the base model layers as not trainable
for layer in base_model.layers:
    layer.trainable = False

# Print the model summary
model.summary()


## Callback definition stage

# Define the Callbacks
checkpoint = ModelCheckpoint(
    os.path.join(PARAM['workspace'], 'checkpoints', 'weights_best.h5'), 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    mode='min'
)

early_stop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5,
    verbose=1,
    mode='min',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0
)

terminate_on_nan = TerminateOnNaN()

callbacks = [checkpoint, early_stop, reduce_lr, terminate_on_nan]


## Training stage
# Compile the model
model.compile(
    optimizer=optimizers.Adam(learning_rate=PARAM['training']['learning_rate']),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // PARAM['training']['batch_size'],
    epochs=PARAM['training']['epochs'],
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // PARAM['training']['batch_size'],
    callbacks=callbacks,
    verbose=1
)

# Plot the training and validation loss and accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# todo: save the plot as image
# removed: plt.show()

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# todo: save the plot as image
# removed: plt.show()


## Saving stage
# Create the directory to save the model
save_dir = os.path.join(PARAM['workspace'], f'saved_model_{time.strftime("%Y%m%d_%H%M%S")}')
os.makedirs(save_dir, exist_ok=True)

# Save the model in various formats
model.save(os.path.join(save_dir, 'model.h5'))
model.save(os.path.join(save_dir, 'model'))

# Save the model's architecture as JSON
with open(os.path.join(save_dir, 'model_architecture.json'), 'w') as f:
    f.write(model.to_json())

print(f"Model saved in {save_dir}")

class_indices = train_generator.class_indices
file_path_class_indices=os.path.join(PARAM['workspace'],'class_indices.json')  

with open(file_path_class_indices, 'w') as f:
    json.dump(class_indices, f)
