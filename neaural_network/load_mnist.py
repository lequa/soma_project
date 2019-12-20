import pickle
import numpy as np
import gzip

key_file = {
        "x_train":"train-images-idx3-ubyte.gz",
        "t_train":"train-labels-idx1-ubyte.gz",
        "x_test":"t10k-images-idx3-ubyte.gz",
        "t_test":"t10k-labels-idx1-ubyte.gz",
        }


def load_label(file_name):
    file_path = file_name
    with gzip.open(file_path, "rb") as f: #最初の8バイト分はデータ本体でないので飛ばす
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    one_hot_labels = np.zeros((labels.shape[0], 10))
    for i in range(labels.shape[0]):
        one_hot_labels[i, labels[i]] = 1
    return one_hot_labels


def load_image(file_name):
    file_path = file_name
    with gzip.open(file_path, "rb") as f: #画像本体のほうは16バイト分飛ばす必要がある
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    return images


def convert_into_numpy(key_file):
    dataset = {}
    
    dataset["x_train"] = load_image(key_file["x_train"])
    dataset["t_train"] = load_label(key_file["t_train"])
    dataset["x_test"] = load_image(key_file["x_test"])
    dataset["t_test"] = load_label(key_file["t_test"])
    
    return dataset


def load_mnist():
    #mnistを読み込みnumpy配列として出力する
    dataset = convert_into_numpy(key_file)
    #データ型をfloat32型に指定しておく
    dataset["x_train"] = dataset["x_train"].astype(np.float32)
    dataset["x_test"] = dataset["x_test"].astype(np.float32)
    dataset["x_train"] /= 255.0
    dataset["x_test"] /= 255.0
    dataset["x_train"] = dataset["x_train"].reshape(-1, 28*28)
    dataset["x_test"] = dataset["x_test"].reshape(-1, 28*28)
    return dataset
    
