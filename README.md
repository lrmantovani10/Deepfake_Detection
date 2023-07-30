# Deepfake Detection Project
I recently embarked on a personal side project of attempting to detect whether an image posted online is a deepfake or not. To do so, I created a two-step model inspired by [this paper](https://www.semanticscholar.org/paper/Deep-Fake-Image-Detection-Based-on-Pairwise-Hsu-Zhuang/5598872d32fd15e367584eb99c07ae79f794243e), with an implementation adapted from [this paper](https://cs230.stanford.edu/projects_spring_2020/reports/38857501.pdf). 
Model architecture diagram:

![image description](Diagram.png)

The first part of the model is a Siamese network to detect similarities between real and fake images, and the second is a convolutional neural network with a binary classifier. Despite the accomplishments of the model described by the latter paper, its architecture is excessively large, and its hyperparameters are not supported by my hardware limitations, so I decided to decrease the batch size and shift from a "batch all" to a "batch hard" strategy when calculating the triplet loss to make better use of the data provided. Furthermore, believing that I could optimize the model even harder, I used an image preprocessing strategy that involved using OpenCV to draw facial features that help the model discern significant differences between real and deepfake components of the face, such as nose and eyes.

The file distribution of the project is as follows:
* display.py &rarr; used to visualize certain images in the dataset during preprocessing. This allows us to visualize, for example, markings on the nose and eyes among the training images.
* process_data.py &rarr; file used to do the preprocessing of the data, which is initially in a folder called "archive" (whose contents are available here: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces). This file will add facial landmarks to these images and store them in a folder called "data/", following preprocessing steps that resemble those proposed by participants of the Facebook Deepfake Detection Challenge (https://www.kaggle.com/code/robikscube/kaggle-deepfake-detection-introduction)
* functions.py &rarr; file containing the functions necessary for main.py to execute properly. These include functions to train, validate, and test the model.
* main.py &rarr; main file of the program, which determines when certain steps of the program's execution (such as training and testing) will be run.
* paper.pdf &rarr; reference paper for the model's implementation. Describes the model's architecture, most of which was adopted by this implementation. However, hyperparameters such as batch size and learning rate were modified in my implementation. Furthermore, my preprocessing steps differed entirely from the paper at hand and instead involved using OpenCV to draw facial landmarks. 
* unzip.py &rarr; script used to unzip the contents of the "data/" folder, used for training the model in a cloud computing environment where uploading a large number of files is not recommended.

Note: Although I was unable to run the model for all the given epochs due to
hardware limitations, the combination of the facial landmarks preprocessing step 
and the high accuracy of the architecture proposed by the referenced paper lead
me to believe that this is an innovative solution to the problem at hand, explaining
why initial iterations of training already demonstrated an accuracy of ~80%. I 
believe that with sufficient training and validation time with the parameters specified by
the code, this number would rapidly increase to 90%. 
