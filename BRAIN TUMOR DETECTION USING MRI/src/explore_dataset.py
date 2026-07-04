import os

# Dataset paths
train_dir = "datasets/Training"
test_dir = "datasets/Testing"

print("=" * 50)
print("TRAINING DATA")
print("=" * 50)

for category in os.listdir(train_dir):
    category_path = os.path.join(train_dir, category)

    if os.path.isdir(category_path):
        num_images = len(os.listdir(category_path))
        print(f"{category:<15} : {num_images} images")

print("\n")

print("=" * 50)
print("TESTING DATA")
print("=" * 50)

for category in os.listdir(test_dir):
    category_path = os.path.join(test_dir, category)

    if os.path.isdir(category_path):
        num_images = len(os.listdir(category_path))
        print(f"{category:<15} : {num_images} images")