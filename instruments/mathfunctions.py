import numpy as np

def softmax(arr: np.ndarray) -> np.ndarray:
    arr_max = max(arr)
    
    exp_arr = np.exp(arr-arr_max)
    statistic_sum = np.sum(exp_arr)
    if(statistic_sum == 0):
        return softmax(arr + 0.1)
    return exp_arr/statistic_sum

def is_float(x: any) -> bool:
    if(x is None):
        return False
    try:
        float(x)
        return True
    except ValueError:
        return False

def find_uniques(arr: np.ndarray) -> np.ndarray:
    uniques = []
    for value in arr:
        if(value in uniques):
            continue
        else:
            uniques.append(value)
    return np.array(uniques)

def softmax_array(array: np.ndarray) -> np.ndarray:
    result = np.zeros(array.shape)
    for i in range(array.shape[0]):
        result[i] = softmax(array[i])
    return result

def kronecker_delta(i: int, j: int):
        if(i == j):
            return 1
        else:
            return 0

def cross_entropy(targets: np.ndarray, predictions: np.ndarray, weights: np.ndarray = np.ones(4)) -> np.ndarray:
    loss = 0
    
    for i in range(targets.shape[0]):
        for c in range(targets.shape[1]):
            t = -targets[i,c]*weights[c]
            if(t != 0):
                p = predictions[i,c]
                if(np.abs(p) < 0.00001):
                    p = 0.00001
                loss += t*np.log(p)
            
    loss = loss/targets.shape[0]
    return loss
