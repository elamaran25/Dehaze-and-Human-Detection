# app/dehazing_model.py

import tensorflow as tf
import numpy as np
import cv2
import os

from tensorflow.keras.models import model_from_json
import os

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Concatenate
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, Concatenate
from tensorflow.keras.models import Model

def aodnet_model():
    input_img = Input(shape=(None, None, 3))
    
    conv1 = Conv2D(3, (1, 1), padding='same', activation='relu')(input_img)
    conv2 = Conv2D(6, (3, 3), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(3, (5, 5), padding='same', activation='relu')(conv2)

    concat = Concatenate()([conv1, conv2, conv3])  # This makes it 3 + 6 + 3 = 12 channels
    output = Conv2D(3, (3, 3), padding='same', activation='sigmoid')(concat)

    model = Model(inputs=input_img, outputs=output)
    return model




def load_dehazing_model():
    model = aodnet_model()
    model.load_weights("models_weights/AODNet_weights.h5")  # ← This will now work perfectly
    print("[INFO] AOD-Net model loaded with weights.")
    return model



def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(image, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Post-process model output to original format
def postprocess_image(output, original_shape):
    output = output[0]  # Remove batch dimension
    output = np.clip(output, 0, 1)
    output = (output * 255).astype('uint8')
    output = cv2.resize(output, (original_shape[1], original_shape[0]))  # Resize back to original
    return output

# Full dehazing function
def dehaze_image(image, model):
    preprocessed = preprocess_image(image)
    output = model.predict(preprocessed)
    dehazed = postprocess_image(output, image.shape)
    return dehazed
