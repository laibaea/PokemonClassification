from google.colab import drive
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

drive.mount('/content/drive')

trainDF = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Pokemon/Train/train.csv')

X = trainDF.iloc[:, 0].values
y = trainDF.iloc[:, 1].values

def dist(image1, image2):
    return np.sum((image1 - image2) ** 2)

def KNN(X, y, test_point, k=5):
    vals = []
    for i in range(len(X)):
        trainImage = cv2.imread('/content/drive/MyDrive/Colab Notebooks/Pokemon/Train/TrainImages/Images/' + X[i])
        coloredTrainImage = cv2.cvtColor(trainImage, cv2.COLOR_BGR2RGB)
        resizedTrainImage = cv2.resize(coloredTrainImage, (300, 300))
        distance = dist(resizedTrainImage, test_point)
        vals.append((distance, y[i]))
    vals = sorted(vals, key=lambda x: x[0])
    vals = vals[:k]
    
    return vals

testDF = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Pokemon/Test/test.csv')
testData = testDF.values
z = testData[:,0]


plt.figure()
for i in range(100, 109):
    testImage = cv2.imread('/content/drive/MyDrive/Colab Notebooks/Pokemon/Test/TestImages/Images/' + z[i])
    coloredTestImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2RGB)
    resizedTestImage = cv2.resize(coloredTestImage, (300, 300))
    pred = KNN(X, y, resizedTestImage, k=7)
    labels = [label[1] for label in pred]
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_label = unique_labels[np.argmax(counts)]
    plt.imshow(resizedTestImage)
    plt.title('Hello, I am ' + max_label)
    plt.axis('off')
    plt.show()
