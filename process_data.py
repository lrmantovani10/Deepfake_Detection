from functions import *

# Codewide settings and paths
train_dir = base_path + "train/"
test_dir = base_path + "test/"
val_dir = base_path + "valid/"

# Process the corresponding training, test, and validation images if needed
old_directories = [train_dir, test_dir, val_dir]
new_directories = [new_train_dir, new_test_dir, new_val_dir]
for i in range(3):
    # Check if the new directories exist and if they have
    # the same number of images as the old directories
    if not os.path.exists(new_directories[i]) or len(
        os.listdir(new_directories[i])
    ) != len(os.listdir(old_directories[i])):
        # Check if the new directory paths exist. If not, make them
        total_path = ""
        for sub_path in new_directories[i].split("/")[:-1]:
            total_path += sub_path + "/"
            if not os.path.exists(total_path):
                os.mkdir(total_path)

        # Process the images
        process_images(old_directories[i], new_directories[i])