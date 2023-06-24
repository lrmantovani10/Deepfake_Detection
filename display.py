from functions import *

# Codewide settings and paths
plt.style.use("ggplot")
train_metadata = "archive/train.csv"
test_metadata = "archive/test.csv"
val_metadata = "archive/valid.csv"

# Displaying metadata
show_metadata(train_metadata, test_metadata, val_metadata)

# Displaying sample images
train_images = [new_train_dir + x for x in os.listdir(new_train_dir)]
show_image(train_images[0])
show_image(train_images[-1])

# Displaying images with landmarks
Image._show(face_locations(train_images[0], display = True))
Image._show(face_locations(train_images[-1], display = True))