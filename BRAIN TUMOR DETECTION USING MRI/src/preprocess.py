import tensorflow as tf

# Dataset paths
train_dir = "datasets/Training"
test_dir = "datasets/Testing"

# Image parameters(size of the image and batch size)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# every image becomes 224*224*3 y 3 bcz MRI images r stored as RGB images so 3
# batch size is 32, means 32 images will be processed at a time. It is a hyperparameter which can be tuned according to the system configuration and dataset size.

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Save class names BEFORE mapping
class_names = train_dataset.class_names

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

print("Classes:", class_names)