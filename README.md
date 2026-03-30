# Image KNN Classifier

## Overview
This project implements a K-Nearest Neighbors (KNN) classifier for image classification.

The model predicts the label of an input image based on the labels of the most similar images in the training dataset.

---

## Features
- Custom KNN implementation (no built-in libraries)
- Image distance calculation
- Voting mechanism for classification
- Basic image processing

---

## How It Works
1. Each image is represented as pixel data
2. Distance is computed between images
3. The K nearest neighbors are selected
4. The most common label among neighbors is returned

