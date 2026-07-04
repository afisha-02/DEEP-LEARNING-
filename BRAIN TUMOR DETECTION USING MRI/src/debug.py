import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("models/brain_tumor_model.keras")

# Load ONE batch
test_dataset = tf.keras.utils.image_dataset_from_directory(
    "datasets/Testing",
    image_size=(224,224),
    batch_size=8,
    shuffle=False
)

class_names = test_dataset.class_names

for images, labels in test_dataset.take(1):

    predictions = model.predict(images)

    print("\nClass Names:", class_names)

    print("\nActual Labels:")
    print(labels.numpy())

    print("\nPredicted Labels:")
    print(np.argmax(predictions, axis=1))

    print("\nPrediction Probabilities:")
    print(np.round(predictions, 3))