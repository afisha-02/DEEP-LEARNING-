import tensorflow as tf
from tensorflow.keras import layers, Model

IMG_SIZE = (224, 224)
NUM_CLASSES = 4

def build_model():

    # Data Augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ])

    # MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers
    base_model.trainable = False

    inputs = layers.Input(shape=(224,224,3))

    x = data_augmentation(inputs)

    # MobileNet preprocessing
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model.trainable = True

# Freeze first 100 layers
    for layer in base_model.layers[:100]:
         layer.trainable = False
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation="relu")(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs)

    return model