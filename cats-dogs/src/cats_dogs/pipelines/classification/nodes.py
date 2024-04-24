from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input

def preprocess_images(images: dict[str, Image.Image]) -> dict[str, Image.Image]:
    ''' Convert images to grayscale and resize them to 64x64 pixels. '''
    for name, image in images.items():
        images[name] = image.convert('L').resize((64, 64))
    return images

def data_split(images: dict[str, Image.Image]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ''' Split data into training and testing sets. '''
    classes = ['cat', 'dog']
    features = [np.array(value) for value in images.values()]
    targets = [classes.index(name.split(".")[0]) for name in images.keys()]
    targets = [[lbl, 1-lbl] for lbl in targets]

    X = np.array(features)
    y = np.array(targets)
    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    return X_train, X_test, y_train, y_test

def build_model() -> Sequential:
    model = Sequential()
    model.add(Input(shape=(64,64,1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> list:
    model = build_model()
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=0)
    weights = model.get_weights()
    return weights

def evaluate_model(weights: list, X_test: np.ndarray, y_test: np.ndarray) -> Image.Image:
    model = build_model()
    model.set_weights(weights)
    fig, axs = plt.subplots(2, 5)
    accuracy = round(model.evaluate(X_test, y_test)[1] * 100,2)
    fig.suptitle(f"Model accuracy: {accuracy}%")

    classes = ['cat', 'dog']
    for i in range(10):
        ax = axs[i//5, i%5]
        ax.imshow(X_test[i].reshape(64, 64), cmap='gray')
        prediction = model.predict(X_test[i].reshape(1, 64, 64, 1))
        prediction = classes[1-np.argmax(prediction[0])]
        actual = classes[(y_test[i][0])]
        print(f"Prediction: {prediction}")
        print(f"Actual: {actual}")
        ax.set_title(prediction, color='green' if prediction == actual else 'red')
        ax.axis('off')

    fig.canvas.draw()
    image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    return image