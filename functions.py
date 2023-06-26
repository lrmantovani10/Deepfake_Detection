# Dataset used: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
# Paper for model: https://cs230.stanford.edu/projects_spring_2020/reports/38857501.pdf

# Importing dependencies
import os, torch, cv2
import face_recognition as fr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from collections import Counter
import matplotlib.image as mpimg
import matplotlib.pylab as plt
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset

# Project settings
base_path = "archive/real_vs_fake/real-vs-fake/"
new_train_dir = "data/train/"
new_test_dir = "data/test/"
new_val_dir = "data/valid/"
weights_filename = "data/detection_model_weights.h5"


# Function to prepare the input to be inserted in the pythorch model
def prepare_input(img_path):
    # Load image with matplotlib
    image = mpimg.imread(img_path)

    # Normalizing the image with cv2
    cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)

    # Randomly mirroring the image
    if np.random.rand() > 0.5:
        image = np.fliplr(image)

    # Randomly transforming the image in brightness
    if np.random.rand() > 0.5:
        # Random contrast with openCV
        brightness = np.random.randint(-50, 50)
        image = cv2.convertScaleAbs(image, beta=brightness)

    # Randomly transforming the image in contrast
    if np.random.rand() > 0.5:
        # Random contrast with openCV
        contrast = np.random.uniform(0.5, 1.5)
        image = cv2.convertScaleAbs(image, alpha=contrast)

    # Randomly transforming the image in saturation
    if np.random.rand() > 0.5:
        # Assigning random saturation values
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image = np.array(image, dtype=np.float64)
        random_saturation = 0.5 + np.random.uniform()
        image[:, :, 1] = image[:, :, 1] * random_saturation

        # Clamping the saturation to maximum value of 255
        image[:, :, 1][image[:, :, 1] > 255] = 255
        image = np.array(image, dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    # Converting the image to a tensor. We call copy()
    # to remove potential negative strides from the image array
    tensor = torch.tensor(image.copy())

    # Reshaping the tensor. These dimensions correspond to the
    # number of channels, height, and width of the image
    tensor = tensor.reshape(3, 224, 224)

    # Converting the tensor to a float
    tensor = tensor.float()

    # Returning the tensor
    return tensor


# Defining a function to generate a dataset
def generate_dataset(dir, offset):
    # Generating array of images in the folder
    images = os.listdir(dir)
    image_length = len(images)
    images = images[int(image_length * offset) :]

    # Output set
    output = set()

    # Iterating over each image and adding it to the output set
    for image in images:
        output.add(dir + image)

    return list(output)


# Class generating the data used by the model on phase 1 (Siamese Network)
class Phase1Data(Dataset):
    # Constructor
    def __init__(self, dir, transform=None, fakes=[], reals=[], offset=0):
        # Saving the directory
        self.dir = dir

        # Saving the positives
        self.fakes = fakes

        # Saving the negatives
        self.reals = reals

        # Saving the transform
        self.transform = transform

        # Generating the dataset
        self.dataset = generate_dataset(dir, offset)

    # Function to get the length of the dataset
    def __len__(self):
        return len(self.dataset)

    # Function to get the item at a given index
    def __getitem__(self, idx):
        # Defining the anchor and its label
        anchor = self.dataset[idx]
        anchor_label = 0 if "real" in anchor else 1

        # Defining the positive and negative
        if anchor_label == 0:
            positives = self.reals
            negatives = self.fakes

            # Obtaining the positive and negative by randomly picking
            # from their corresponding arrays
            positive = np.random.choice(positives)
            negative = np.random.choice(negatives)

            # Removing the positive and negative from their arrays
            # to avoid repetition
            self.reals.remove(positive)
            self.fakes.remove(negative)

        else:
            positives = self.fakes
            negatives = self.reals

            # Obtaining the positive and negative by randomly picking
            # from their corresponding arrays
            positive = np.random.choice(positives)
            negative = np.random.choice(negatives)

            # Removing the positive and negative from their arrays
            # to avoid repetition
            self.fakes.remove(positive)
            self.reals.remove(negative)

        # Transforming the images if needed
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        else:
            anchor = torch.load(anchor)
            positive = torch.load(positive)
            negative = torch.load(negative)

        # Returning the images and the anchor label
        return anchor, positive, negative


# Class generating the data used by the model on phase 2 (CNN)
class Phase2Data(Dataset):
    # Constructor
    def __init__(self, dir, transform=None, offset=0):
        # Saving the directory
        self.dir = dir

        # Saving the transform
        self.transform = transform

        # Generating the dataset
        self.dataset = generate_dataset(dir, offset)

    # Function to get the length of the dataset
    def __len__(self):
        return len(self.dataset)

    # Function to get the item at a given index
    def __getitem__(self, idx):
        # Getting the image
        img_path = self.dataset[idx]

        # Transforming the image
        if self.transform:
            image = self.transform(img_path)
        else:
            image = torch.load(img_path)

        # Getting the label. 1 = "fake", 0 = "real"
        label = 0 if "real" in img_path else 1

        # Returning the image and its label
        return image, label


# Function to show file metadata
def show_metadata(train, test, val):
    # Reading the metadata CSV files
    metadata_files = [
        pd.read_csv(train),
        pd.read_csv(test),
        pd.read_csv(val),
    ]

    # Plot parameters
    labels = ["Train", "Test", "Validation"]

    # Capitalizing the labels and specifying whether they refer to train, test, or validation
    for index in range(0, len(metadata_files)):
        metadata_files[index]["label_str"] = (
            labels[index] + " " + metadata_files[index]["label_str"].str.capitalize()
        )

    # Concatenating dataframes
    metadata = pd.concat(metadata_files)

    # Plot colors
    colors = ["red", "green", "blue", "purple", "yellow", "orange"]

    # Making a pie chart
    _, ax = plt.subplots()
    label_counts = Counter(metadata["label_str"])

    # Plotting the pie chart
    ax.pie(
        label_counts.values(),
        labels=label_counts.keys(),
        colors=colors,
        autopct="%1.1f%%",
        textprops={"color": "white"},
        startangle=90,
    )
    ax.set_title("Distribution of image types in the dataset")

    # Plotting the data
    plt.show()

    # Printing a summary of the data
    print("Image counts per category")
    for key, value in label_counts.items():
        print(f"{key}: {value}")

    # Total counts
    print(f"Total images: {sum([i for i in label_counts.values()])}")


# Defining a function to display images
def show_image(file_path):
    # Reading sample image files
    sample_image = mpimg.imread(file_path)

    # Converting the images to RGB
    plt.imshow(sample_image)

    # Setting the titles
    split_file = file_path.split("/")
    plt.title(f"Sample {split_file[-2]} image: " + split_file[-1])

    # Displaying the image
    plt.grid(False)
    plt.show()


# Identify face locations in an image
def face_locations(file_path, display=False):
    # Reading sample image files
    image = mpimg.imread(file_path)

    if display:
        print(f"{len(locations)} face(s) were identified in this photo.")
        # Identifying face locations in an image
        locations = fr.face_locations(image)
        # Iterating over each face
        for face_location in locations:
            # Print the positions of the face
            up, right, down, left = face_location
            print(f"Face location up: {up}, left: {left}, down: {down}, right: {right}")

            # Visualizing the face itself by reading its pixel values
            face_image = image[up:down, left:right]
            _, ax = plt.subplots(1, 1, figsize=(5, 5))
            plt.grid(False)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.imshow(face_image)

    # Identifying landmarks
    landmarks = fr.face_landmarks(image)

    # Generating image from array
    image_data = Image.fromarray(image)
    drawing = ImageDraw.Draw(image_data)

    # Iterating over each face landmark
    for landmark in landmarks:
        if display:
            # Print the location of each facial feature in this image
            for feature in landmark.keys():
                print(f"The {feature} of this face is located at {landmark[feature]}")

        # Using line to sketch facial features
        for feature in landmark.keys():
            drawing.line(landmark[feature], width=2)

    return image_data


# Function to load images with face locations and resize them to 224 x 224
def process_images(dir_path, new_dir):
    # Files in the new directory
    new_dir_files = set(os.listdir(new_dir))

    # Iterate over all files in directory
    for appendix in ["real", "fake"]:
        short_path = dir_path + appendix + "/"

        # Iterate over all files in directory
        for file in os.listdir(short_path):
            # Path of processed file
            surfile = appendix + "_" + file

            # Check if final path exists and continue if so
            if surfile in new_dir_files:
                continue

            # Reading sample image files with face locations
            image = face_locations(short_path + file)

            # Resizing the image
            image = image.resize((224, 224))

            # Saving the resized image
            final_path = new_dir + surfile
            image.save(final_path)


# Generate a Common Fake Feature Network (CFFN) using PyTorch
def generate_cffn():
    # Defining a residual unit
    class ResidualUnit(nn.Module):
        def __init__(self, in_channels, out_channels, kern_size):
            super(ResidualUnit, self).__init__()

            # "Each residual unit in the dense block is a standard residual unit with 2 sets
            # of BatchNorm->Swish->Conv and a skip connection"
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.swish = nn.SiLU(inplace=True)

            # Valid padding -- input size might be different from output size
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kern_size, padding=1 if kern_size == 3 else 2
            )
            self.bn2 = nn.BatchNorm2d(out_channels)

            # Same padding -- input size = output size
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kern_size, padding="same"
            )

            # Defining the skip connection
            self.shortcut = None
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels),
                )

            # Setting the layers to a module list
            self.layers = nn.ModuleList(
                [
                    self.bn1,
                    self.swish,
                    self.conv1,
                    self.bn2,
                    self.swish,
                    self.conv2,
                    self.shortcut,
                ]
            )

        # Forward pass
        def forward(self, x):
            # Using the parameters defined above, including the
            # two layers of convolution and the Swish activation
            out = x
            for layer in self.layers[:-1]:
                out = layer(out)

            # Adding the value of the shortcut -- this makes
            # the block a residual one
            residual = x
            if self.shortcut:
                residual = self.layers[-1](x)
            out += residual
            return out

    # Defining the CFFN architecture
    class ResNet(nn.Module):
        # Function to generate residual blocks using the class defined above
        def make_residual_unit(self, in_channels, out_channels, kern_size, num_blocks):
            # Layers array
            layers = nn.ModuleList([])
            # Adding layers to the module list
            layers.append(ResidualUnit(in_channels, out_channels, kern_size))

            # Adding residual blocks
            for _ in range(1, num_blocks):
                layers.append(ResidualUnit(out_channels, out_channels, kern_size))

            return nn.Sequential(*layers)

        # Parameters specific to each model in the CFFN
        def one_model(self, kern_size):
            # First residual block
            self.block1 = self.make_residual_unit(64, 64, kern_size, 1)

            # Second residual block
            self.block2 = self.make_residual_unit(64, 96, kern_size, 1)

            # Third residual block
            self.block3 = self.make_residual_unit(96, 96, kern_size, 3)

            # Fourth residual block
            self.block4 = self.make_residual_unit(96, 128, kern_size, 1)

            # Fifth residual block
            self.block5 = self.make_residual_unit(128, 128, kern_size, 2)

            # Sixth residual block
            self.block6 = self.make_residual_unit(128, 256, kern_size, 1)

            # Seventh residual block
            self.block7 = self.make_residual_unit(256, 256, kern_size, 6)

            # Return module list of layers
            return nn.ModuleList(
                [
                    self.block1,
                    self.block2,
                    self.block3,
                    self.block4,
                    self.block5,
                    self.block6,
                    self.block7,
                ]
            )

        # Defining the ResNet architecture
        def __init__(self):
            super(ResNet, self).__init__()

            # First conv layer (7x7 with stride of 3)
            self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=3)

            # Softmax layer
            self.softmax = nn.Softmax(dim=1)

            # Set the layers to a module list
            self.layers = nn.ModuleList(
                [
                    self.conv,
                    self.softmax,
                ]
            )

            self.m1 = self.one_model(3)
            self.m2 = self.one_model(5)

        # Forward pass using our recently defined building blocks
        def forward(self, x):
            # Building forward propagation using the building blocks above
            x = self.layers[0](x)

            # Output of the two models compriisng the Siamese network
            models_output = []

            # Iterating over each model
            for model in [self.m1, self.m2]:
                # Iterating over layers
                f_x = x
                for layer in model:
                    f_x = layer(f_x)

                # Adding to model output
                models_output.append(f_x)

            return torch.cat(models_output)

        # Function to generate the cffn output
        def cffn_output(self, x):
            # Forward propagation
            x = self.forward(x)

            # Reshaping to 128 dimensions
            x = x.view(128, -1)

            # softmax function
            x = self.layers[1](x)

            return x

    # Return an instance of the ResNet model
    return ResNet()


# Generate the CNN to classify whether the face is fake or not
# using PyTorch
def generate_cnn():
    # Defining the CNN architecture
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            # Batch normalization
            self.bn = nn.BatchNorm2d(256)
            # Swish activation
            self.swish = nn.SiLU(inplace=True)
            # Convolutional layer
            self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding="same")
            # Final layer that generates the classifier output
            self.final = nn.Linear(2, 1)
            # Setting the layers to a module list
            self.layers = nn.ModuleList([self.bn, self.swish, self.conv, self.final])

        # Forward pass
        def forward(self, x):
            # Obtaining the output using the layers above
            out = x
            for layer in self.layers[:-1]:
                out = layer(out)

            # Reshape to two dimensions with reduce_mean
            # and generate the output from the final linear layer
            value_mean = out.view(2, -1).mean(dim=1)
            result = self.layers[-1](value_mean)
            return result

    # Return an instance of the CNN model
    return CNN()


# Function to generate the full model
def generate_model():
    # Generate the CFFN
    cffn = generate_cffn()

    # Generate the CNN
    cnn = generate_cnn()

    # Generate a module list
    layers = [cffn, cnn]

    # Combine both to form the model
    model = nn.Sequential(*layers)

    # Return the model
    return model


# Euclidean distance between two tensors
def euclidean_distance(t1, t2):
    return torch.pow(t1 - t2, 2).sum(dim=1)


# Function to compute the distance matrix between
# anchor, positive, and negative samples
def compute_distance_matrix(a, p, n):
    distance_matrix = torch.zeros(a.size(0), 3)
    distance_matrix[:, 0] = euclidean_distance(a, a)
    distance_matrix[:, 1] = euclidean_distance(a, p)
    distance_matrix[:, 2] = euclidean_distance(a, n)
    return distance_matrix


# Use the batch ahrd strategy to compute the triplet loss
# between the anchor, positive, and negative samples
def batch_hard_triplet_loss(samples, margin=1):
    a, p, n = samples
    distance_matrix = compute_distance_matrix(a, p, n)
    hard_negative = torch.argmax(distance_matrix[:, 2])
    loss = torch.max(
        torch.tensor(0.0), distance_matrix[:, 0] - distance_matrix[:, 1] + margin
    )
    loss += torch.max(
        torch.tensor(0.0),
        distance_matrix[:, 0][hard_negative] - distance_matrix[:, 2] + margin,
    )
    return torch.mean(loss)


# Function to fit images to device and generate the output of the model
def fit_forward_cffn(model, anchor, positive, negative, margin, device):
    # List of image forward passes
    image_fs = []

    # Iterate over each image
    for image in (anchor, positive, negative):
        # Fit the image to the device
        image = image.to(device)
        # Generate the output of the model
        image_fs.append(model[0].cffn_output(image))

    # Compute the triplet loss with negative values allowed
    loss = batch_hard_triplet_loss(image_fs, margin)

    # Return the output
    return loss, image_fs


# Function to save the model's weights
def save_model(model):
    torch.save(model.state_dict(), weights_filename)


# Function to calculate the accuracy of triplet loss for a certain batch size
def triplet_accuracy(anchor, positive, negative, margin=1):
    # Compute the distance matrix
    distance_matrix = compute_distance_matrix(anchor, positive, negative)

    # Obtain the distance of anchor-positive and anchor-negative pairs
    anchor_positive_distance = distance_matrix[:, 1]
    anchor_negative_distance = distance_matrix[:, 2]

    # Check if the distances satisfy the margin condition
    correct_triplets = torch.logical_and(
        anchor_positive_distance < anchor_negative_distance,
        anchor_positive_distance - anchor_negative_distance < margin,
    )

    # Calculate the accuracy
    accuracy = torch.sum(correct_triplets).item() / anchor.size(0)
    return accuracy


# Regularization function
def regularize(model, regularization):
    regularization_term = 0
    for param in model.parameters():
        # Adding L2 Norm to running sum
        regularization_term += torch.norm(param)

    # Multiplying by the regularization term
    return regularization * regularization_term


# Function to compute the accuracy of the binary classifier
def accuracy_ce(output, label):
    # Compute the accuracy
    accuracy = torch.sum(
        # Check if the output is greater than 0.5
        torch.round(output)
        == label
    ).item() / output.size(0)
    return accuracy


# F-score calculation
def p_metrics(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall, 2 * precision * recall / (precision + recall)


# Computing the true positives, false positives, and false negatives
# for the binary classifier
def classifier_metrics(output, label):
    # True positives
    tp = torch.sum(
        torch.round(output) == label and label == torch.ones(1).float()
    ).item()
    # False positives
    fp = torch.sum(
        torch.round(output) != label and label == torch.zeros(1).float()
    ).item()
    # False negatives
    fn = torch.sum(
        torch.round(output) != label and label == torch.ones(1).float()
    ).item()
    return tp, fp, fn


# Phase 1 validation
def phase1_val(
    model,
    val1_loader,
    margin,
    device,
):
    # Aggregate metrics
    total_accuracy = 0
    # Validation step -- fine-tuning the learning rate hyperparameter
    for anchor, positive, negative in val1_loader:
        # Forward propagation
        _, image_fs = fit_forward_cffn(
            model, anchor, positive, negative, margin, device
        )

        # Obtaining the anchor, positive, and negative outputs
        anchor, positive, negative = image_fs

        # Calculate the accuracy
        val1_acc = triplet_accuracy(anchor, positive, negative, margin)
        total_accuracy += val1_acc

        # Printing the metrics
        print("""<Validation 1> Accuracy: {} """.format(val1_acc))

    # Print the final metrics
    print("Validation of phase 1 concluded.")
    print(
        "FINAL METRICS - <Validation 1> Accuracy: {}".format(
            total_accuracy / len(val1_loader)
        )
    )


# Phase 1 training
def phase1_train(
    model, train1_loader, optimizer, margin, device, p1_epochs, scheduler, val1_loader
):
    # Separation between training and validation sets
    split = len(train1_loader) // p1_epochs
    for epoch in range(p1_epochs):
        # Aggregate metrics
        total_accuracy = 0
        # Scheduler gating
        gate_cross = False
        epoch_threshold = split * (epoch + 1)
        for index, (anchor, positive, negative) in enumerate(train1_loader):
            # Determining if validation should be done now
            if (index + 1) == epoch_threshold:
                phase1_val(
                    model,
                    val1_loader,
                    margin,
                    device,
                )
                choice = input("Continue training? (y/n): ")
                if choice != "y":
                    return

            # Calculating the loss
            loss, image_fs = fit_forward_cffn(
                model, anchor, positive, negative, margin, device
            )

            # Obtaining the anchor, positive, and negative outputs
            anchor, positive, negative = image_fs

            # Computing train accuracy for phase 1
            train1_acc = triplet_accuracy(anchor, positive, negative, margin)
            total_accuracy += train1_acc

            # Backpropagation
            loss.backward()

            # Update the weights
            optimizer.step()

            # Clear the optimizer gradients
            optimizer.zero_grad()

            # Scheduler step
            if train1_acc >= 0.9:
                # Determine if scheduler makes its long awaited step
                if not gate_cross:
                    scheduler.step()
                    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
                    gate_cross = True
                else:
                    scheduler.step()

            # Save the model weights at every 5 epochs
            if index % 5 == 0:
                save_model(model)

            # Print the metrics
            print(
                """<Training 1> Epoch: {} |  Accuracy: {}  Loss: {}""".format(
                    epoch, train1_acc, loss.item()
                )
            )

        # Print the final metrics
        print("Training of phase 1 concluded.".format(epoch))
        print(
            """FINAL METRICS - <Training 1> Loss: {} | Accuracy: {} """.format(
                loss.item(), total_accuracy / len(train1_loader)
            )
        )

        # Save the model weights
        save_model(model)


# Phase 1 testing
def phase1_test(model, test1_loader, margin, device):
    # Testing Step -- monitoring metrics
    total_accuracy = 0
    for anchor, positive, negative in test1_loader:
        # Forward propagation
        _, image_fs = fit_forward_cffn(
            model, anchor, positive, negative, margin, device
        )

        # Obtaining the anchor, positive, and negative outputs
        anchor, positive, negative = image_fs

        # Calculate the accuracy
        test1_acc = triplet_accuracy(anchor, positive, negative, margin)
        total_accuracy += test1_acc

        # Print the relevant metrics
        print("""<Testing 1> Accuracy: {} """.format(test1_acc))

    # Print the final metrics
    print("Testing of phase 1 concluded.")
    print(
        """FINAL METRICS - <Testing 1> Accuracy: {}""".format(
            total_accuracy / len(test1_loader)
        )
    )


# Phase 2 validation
def phase2_val(
    model,
    val2_loader,
    device,
):
    # Aggregate metrics
    total_accuracy = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    # Validation step -- fine-tuning the learning rate hyperparameter
    for image, label in val2_loader:
        # Fit the image to the device
        image = image.to(device)
        # Generate tensor from label
        label = (
            torch.zeros(1).float().to(device)
            if label == 0
            else torch.ones(1).float().to(device)
        )

        # Generate the output of the model
        output = model(image)

        # Compute the accuracy
        accuracy = accuracy_ce(output, label)
        total_accuracy += accuracy

        # Computing the true positives, false positives, and false negatives
        tp, fp, fn = classifier_metrics(output, label)

        # Adding to the total true positives, false positives, and false negatives
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Calculating the F-score
        precision, recall, f_val = p_metrics(tp, fp, fn)

        # Printing the relevant metrics
        print(
            """<Validation 2> Accuracy: {} | F1: {} | Precision: {} | Recall: {}""".format(
                accuracy, f_val, precision, recall
            )
        )

    # Printing the final metrics
    print("Validation of phase 2 concluded.")
    precision, recall, f_val = p_metrics(total_tp, total_fp, total_fn)
    print(
        """FINAL METRICS - <Validation 2> Accuracy: {} | Precision: {} | Recall: {} | F1: {}""".format(
            total_accuracy / len(val2_loader), precision, recall, f_val
        )
    )


# phase 2 training
def phase2_train(
    model,
    train2_loader,
    optimizer,
    cross_entropy,
    device,
    p2_epochs,
    regularization,
    scheduler,
    val2_loader,
):
    # Separation between training and validation sets
    split = len(train2_loader) // p2_epochs
    # Training step
    for epoch in range(p2_epochs):
        # Aggregate metrics
        total_accuracy = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        # Scheduler step
        gate_cross = False
        epoch_threshold = split * (epoch + 1)
        for index, (image, label) in enumerate(train2_loader):
            # Determining if validation should be done now
            if (index + 1) == epoch_threshold:
                phase2_val(model, val2_loader, device)
                choice = input("Continue training? (y/n): ")
                if choice != "y":
                    return
            # Fit the image to the device
            image = image.to(device)
            # Generate tensor from label
            label = (
                torch.zeros(1).float().to(device)
                if label == 0
                else torch.ones(1).float().to(device)
            )

            # Generate the output of the model
            output = model(image)

            # Compute the accuracy
            accuracy = accuracy_ce(output, label)
            total_accuracy += accuracy

            # Computing the true positives, false positives, and false negatives
            tp, fp, fn = classifier_metrics(output, label)

            # Adding to the total true positives, false positives, and false negatives
            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Calculating the F-score
            precision, recall, f_val = p_metrics(tp, fp, fn)

            # Calculate the loss with regularization
            loss = cross_entropy(output, label)
            loss += regularize(model, regularization)

            # Backpropagation
            loss.backward()

            # Update the weights
            optimizer.step()

            # Clear the optimizer gradients
            optimizer.zero_grad()

            # Scheduler step
            if accuracy >= 0.9:
                # Determine if scheduler makes its long awaited step
                if not gate_cross:
                    scheduler.step()
                    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
                    gate_cross = True
                else:
                    scheduler.step()

            # Print the relevant metrics
            print(
                """<Training 2> Epoch: {} | Accuracy: {} | F1: {} | Precision: {} | Recall: {}""".format(
                    epoch, accuracy, f_val, precision, recall
                )
            )

            # Save the model weights at every 5 epochs
            if index % 5 == 0:
                save_model(model)

        # Print the final metrics
        print("Training of phase 2 concluded.")
        precision, recall, f_val = p_metrics(total_tp, total_fp, total_fn)
        print(
            """FINAL METRICS - <Training 2> Epoch: {} | Accuracy: {} | Precision: {} | Recall: {} | F1: {}""".format(
                epoch, total_accuracy / len(train2_loader), precision, recall, f_val
            )
        )

        # Save the model weights
        save_model(model)


# Phase 2 testing
def phase2_test(model, test2_loader, device):
    # Relevant metrics
    total_accuracy = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for image, label in test2_loader:
        # Fit the image to the device
        image = image.to(device)
        # Generate tensor from label
        label = (
            torch.zeros(1).float().to(device)
            if label == 0
            else torch.ones(1).float().to(device)
        )

        # Generate the output of the model
        output = model(image)

        # Compute the accuracy
        accuracy = accuracy_ce(output, label)
        total_accuracy += accuracy

        # Computing the true positives, false positives, and false negatives
        tp, fp, fn = classifier_metrics(output, label)

        # Adding to the total true positives, false positives, and false negatives
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Calculating the F-score
        precision, recall, f_val = p_metrics(tp, fp, fn)

        # Printing the relevant metrics
        print(
            """<Testing 2> Accuracy: {} | F1: {} | Precision: {} | Recall: {}""".format(
                accuracy, f_val, precision, recall
            )
        )

    # Print the final metrics
    print("Testing of phase 2 concluded.")
    precision, recall, f_val = p_metrics(total_tp, total_fp, total_fn)
    print(
        """FINAL METRICS - <Testing 2> Accuracy: {} | Precision: {} | Recall: {} | F1: {}""".format(
            total_accuracy / len(test2_loader), precision, recall, f_val
        )
    )
