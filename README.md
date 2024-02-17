# Ad-Creative-Recognition
Advertisement Detection Model
This repository contains code for an advertisement detection model built using TensorFlow and deployed using Flask and HTML. The model predicts whether an image contains an advertisement or not.
----------------------------------------------------------------
Dataset
----------------------------------------------------------------
The dataset used to train the model consists of images categorized into two classes: advertisements and non-advertisements. The dataset was collected from various sources and manually labeled for classification.
Images collected from pixels, pinterest and other websites.

Model Architecture
----------------------------------------------------
The model architecture used is based on a pre-trained ResNet-50 convolutional neural network (CNN) architecture. The ResNet-50 model was fine-tuned on the advertisement detection dataset to learn relevant features.

Data Preprocessing
-----------------------------------------------------
Before training the model, the images were preprocessed by resizing them to the input shape required by the model (224x224 pixels) and normalizing the pixel values to the range [0, 1].

Training
-----------------------------------------------------
The model was trained using the preprocessed images and their corresponding labels. The training process involved minimizing the binary cross-entropy loss function using the Adam optimizer with a decaying learning rate schedule.

Model Evaluation
---------------------------------------------------
The trained model was evaluated on a separate test dataset to assess its performance in terms of accuracy and other relevant metrics.

Deployment
---------------------------------------------------
The model was deployed using Flask, a Python web framework, along with HTML for creating a simple user interface. The deployed application allows users to upload an image and get predictions from the trained model in real-time.

Instructions for Running the Application
--------------------------------------------------
Clone this repository to your local machine.
Install the required dependencies by running pip install -r requirements.txt.
Run the Flask application by executing python app.py in the terminal.
Open a web browser and navigate to http://localhost:5000 to access the application.
Upload an image and click the "Upload" button to get predictions from the model.

Author
Rohit Kumar Verma 
