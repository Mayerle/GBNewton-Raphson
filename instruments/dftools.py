import pandas as pd
import numpy as np
import math

def normalize_features(objects: np.ndarray)->np.ndarray:
    table = objects.T
    means = np.mean(table,axis=1).reshape(table.shape[0],1)
    stds = np.std(table,axis=1).reshape(table.shape[0],1)
    normalized_table = (table-means)/stds
    return normalized_table.T

def convert_to_onevectors(objects: np.ndarray) -> np.ndarray:
    table = objects.T
    ones = np.ones((1,table.shape[1]))
    table_ones = np.concatenate([table,ones])
    return table_ones.T

def one_hot_encode(classes: np.ndarray) -> np.ndarray:
    unique_classes = np.unique(classes)
    index_of = lambda class_: np.argwhere(class_ == unique_classes)[0][0]
    return np.array(list(map(index_of,classes)))

def label_encode(classes: np.ndarray, target_class) -> np.ndarray:
    index_of = lambda class_: 1 if class_ == target_class else -1
    return np.array(list(map(index_of,classes))) 

def split_data(features: np.ndarray, targets:np.ndarray, test_size: float = 0.7, validate_size: float = 0.3) -> list[np.ndarray]:
    test_n   = math.trunc(test_size*features.shape[0])
    validate_n   = math.trunc(test_size*validate_size*features.shape[0])
    if(len(features.shape)):
        x_test     = features[:test_n]
        x_validate = x_test[validate_n:]
        x_test     = x_test[:validate_n]
        x_train    = features[test_n:]
        
        y_test     = targets[:test_n]
        y_validate = y_test[validate_n:]
        y_test     = y_test[:validate_n]
        y_train    = targets[test_n:]
    else:
        x_test     = features[:test_n,:]
        x_validate = x_test[validate_n:,:]
        x_test     = x_test[:validate_n,:]
        x_train    = features[test_n:,:]
        
        y_test     = targets[:test_n,:]
        y_validate = y_test[validate_n:,:]
        y_test     = y_test[:validate_n,:]
        y_train    = targets[test_n:,:]
    
    return [x_train, x_validate, x_test, y_train, y_validate, y_test]

def train_test_split(data: np.ndarray, test_size: float = 0.3) -> list[np.ndarray]:
    test_n   = math.trunc(test_size*data.shape[0])
    train = data[test_n:]
    test  = data[:test_n]
    return [train, test]

def train_validate_test_split(df: type[pd.DataFrame | pd.Series], test_size: float = 0.3, validate_size: float = 0.3) -> type[pd.DataFrame | pd.Series]:
    test_n   = math.trunc(test_size*df.shape[0])
    validate_n   = math.trunc(test_size*validate_size*df.shape[0])
    train = df[test_n:]
    test  = df[:test_n]
    validate = test[:validate_n]
    test  = test[validate_n:]
    return [train, validate, test]

def convert_targets_to_vector(targets: np.ndarray) -> np.ndarray:
    uniques = np.unique(targets).tolist()
    count = len(uniques)

    vector_targets = np.zeros((targets.shape[0], count))
    
    for i in range(targets.shape[0]):
        vector_targets[i,uniques.index(targets[i])] = 1
    return vector_targets


def sample_class(targets: np.ndarray, clss: int):
    class_indexes = np.where(targets == clss)
    classes = np.zeros(targets.shape[0])
    classes[class_indexes] = 1
    return classes            