# Objective
Comprehension of emotion has broad outcomes for both society and commerce. In the
government field, organizations could make immeasurable use of this knowledge to identify
emotions like fear, hate, guilt and even lie. Different corporations such as health-care,
video-game testing and etc. have also been using emotion recognition to accelerate their outcome.
Facial emotion recognition, which is being used at subway stations and airports to recognise
wicked suspects, is the most advanced rise in predicting crime systems globally.
The human brain analyzes emotions automatically. Facial emotion recognition is the method,
which computers detecting human emotions from facial expressions.
In this project I aim to create a deep neural network to classify a person's emotion to fit into 
7 universal facial expressions which include: anger, disgust, fear, happiness, sadness, surprise and
neutral using convolutional neural networks.



# Resource requirement
Language : 
- Python

Dependencies: 
- Anaconda Jupyter notebook, Numpy, Pandas
- TensorFlow, Keras, Open-cv, matplotlib, pyplot, pillow



# Dataset
The fer2013 dataset can be found from the Kaggle machine learning repository via the link below:
www.kaggle.com. 
The dataset has three columns. The first column is labelled as emotion, the second column is labelled
as pixels and the last column is usage. Values of the first column are integers between [o - 6],
and each integer represents one of the emotions (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad,
5=Surprise, 6=Neutral). 
The second column is a vector and set of integer numbers between 0 to 255 and represents 48px * 48px 
grayscale images of a face.
The dataset contains overall 35887 images which labelled as bellow:
0: 4593 images -> Angry,
1: 547 images -> Disgust,
2: 5121 images -> Fear,
3: 8989 images -> Happy,
4: 6077 images -> Sad,
5: 4002 images -> Surprise,
6: 6198 images -> Neutral.



# Code Information
To predict a person's emotion, a convolutional network is constructed.
It is a sequential model. The following layers can
automatically do shape inference so there is no need for a sequential layer.
The model consists of multiple layers such as Conv2D, MaxPooling, Flatten, Dense. Conv2D
layer is a 2D convolution layer, which is a spatial convolution over input images.
The Max Pooling layer subsamples the image, and flatten layer reduces the dimension of
input from n-dimension to 1-dimension. Using dropout layer leads to neurons be chosen
randomly which help the model reduce overfitting.

To test the model on the static inputs, a folder with the name 'custom' must be created and 
images be located in the mentioned folder, then the file 'facial_emotion_static' can
be executed

To build the model the dataset named 'fer2013' must be downloaded from www.kaggle.com and located 
into the same folder as the file 'conda_emotion_model.ipynb' located. Then 'conda_emotion_model.ipynb'
must be exectuted. 



# Evaluation
Models are scored on classification accuracy. Because this is a multi-class problem, the score is the number of correct classifications divided by the total number of elements in the test set.
