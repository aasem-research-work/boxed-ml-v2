
import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

import heapq


PARAM = {
    'workspace': 'my_model01',
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

model_path_dir = os.listdir( os.path.join(PARAM['workspace']))
model_path_dir = [d for d in model_path_dir if d.startswith('saved_model_')]
model_path = os.path.join(PARAM['workspace'],model_path_dir[0],'model.h5')
model = load_model(model_path)


# Load class indices
with open(os.path.join(PARAM['workspace'], 'class_indices.json'), 'r') as f:
    class_indices = json.load(f)
class_indices_swapped = {v: k for k, v in class_indices.items()}

# Function to load and preprocess image
def load_and_preprocess_image(image_path, img_size):
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


test_images = []
test_image_paths = []

for subdir, dirs, files in os.walk(PARAM['dataset']['dir_testset']):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".png") or filepath.endswith(".jpg") or filepath.endswith(".jpeg"):
            test_image_paths.append(filepath)
            img_array = load_and_preprocess_image(filepath, PARAM['training']['img_size'])
            test_images.append(img_array)

test_images = np.vstack(test_images)

def predict_top_n(model, img, top_n=3):
    predictions = model.predict(img)
    top_n_indices = heapq.nlargest(top_n, range(len(predictions[0])), predictions[0].take)
    top_n_probabilities = [float(predictions[0][i]) for i in top_n_indices]  # Convert to float
    return top_n_indices, top_n_probabilities


# Predict class for each test image
predictions = {}

for i, img_path in enumerate(test_image_paths):
    img = test_images[i]
    img = np.expand_dims(img, axis=0)
    top_indices, top_probabilities = predict_top_n(model, img)
    top_indices_name=[class_indices_swapped[i] for i in top_indices]

    
    predictions[img_path] = {
        'top_probabilities': top_probabilities,
        'top_indices': top_indices_name
    }

# Save predictions to a JSON file
with open(os.path.join(PARAM['workspace'], 'class_predictions.json'), 'w') as f:
    json.dump(predictions, f)
