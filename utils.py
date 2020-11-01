import numpy as np
from torch.utils.data.dataset import Dataset


class MyCustomDataset(Dataset):
    def __init__(self, X, y, transforms=None):
        self.X = X
        self.y = y
        self.transforms = transforms

    def __getitem__(self, index):
        img = self.X[index]
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.y[index]
        return img, label

    def __len__(self):
        return len(self.X)


def to_categorical(y):
    num_classes = len(np.unique(y))
    return np.eye(num_classes, dtype='uint8')[y]


def get_data():
    from tensorflow.keras.datasets.fashion_mnist import load_data

    (X_train, y_train), (X_test, y_test) = load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                   'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']

    img_rows = X_train[0].shape[0]
    img_cols = X_test[0].shape[1]

    input_shape = (img_rows, img_cols, 1)

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (X_train, y_train), (X_test, y_test), class_names, input_shape
