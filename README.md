# Handwritten Digit Recognition with Keras

**Handwritten Digit Recognition with Keras** is a practical deep learning project that recognizes handwritten numerical digits drawn by a user through an interactive desktop application. The project combines a Keras neural network, the MNIST handwritten digit dataset, NumPy-based image processing, Pillow for image manipulation, and Tkinter for the graphical user interface.

The core objective is to demonstrate how a machine learning model can move beyond training and become an interactive real-world application. Instead of simply evaluating pre-existing MNIST images, the application provides a 300×300 drawing canvas where users can draw a digit with the mouse and receive an immediate prediction together with the model's confidence score.

The deep learning component is implemented using **Keras Sequential API**. The model accepts 28×28 grayscale images, flattens the image into a one-dimensional representation, and passes it through a fully connected Dense layer containing 128 neurons with ReLU activation. A final Dense layer with 10 neurons and Softmax activation produces probabilities for digits from 0 to 9. The model is compiled using the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric. It is trained on the MNIST dataset after normalizing pixel values from 0–255 to the 0–1 range. The current implementation trains for three epochs with a batch size of 64.

One of the most important parts of the project is the **custom preprocessing pipeline**. Since a user's drawing is created on a 300×300 canvas rather than directly in MNIST's 28×28 format, the application first detects the bounding box of the handwritten strokes. It crops the relevant region, resizes the digit while maintaining its aspect ratio, and places it onto a 28×28 black canvas. The processed image is then normalized and centered according to its center of mass. This preprocessing makes the user's handwritten input more compatible with the images used during MNIST training and helps improve recognition reliability.

The `predictDigit()` function sends the processed image to the trained Keras model, obtains the predicted probability distribution, identifies the digit with the highest probability using `argmax`, and returns both the predicted digit and confidence value.

The **Tkinter GUI** provides a simple and accessible user experience. Users can draw directly on the canvas, click **Recognise** to classify the handwriting, and use **Clear** to remove the previous drawing and start again. The predicted digit and confidence percentage are displayed within the application.

### Key Features

* Keras-based neural network for handwritten digit classification
* MNIST dataset training
* 28×28 grayscale image processing
* Pixel normalization
* Bounding-box detection and cropping
* Aspect-ratio-preserving image resizing
* Center-of-mass alignment
* Real-time handwritten digit prediction
* Confidence percentage display
* Interactive Tkinter drawing canvas
* Clear and redraw functionality
* NumPy and Pillow-based preprocessing

### Technology Stack

**Python | Keras | NumPy | Pillow (PIL) | Tkinter | MNIST | Deep Learning**

This project demonstrates the complete workflow of a small computer vision application: **dataset → preprocessing → neural-network training → prediction → graphical user interface**. It provides a strong foundation for extending the application toward more sophisticated convolutional neural networks (CNNs), model persistence, improved accuracy, prediction history, image export, and deployment as a web or desktop AI application.

**Repository:** https://github.com/godfreypurification7/digit_recognization_keras
