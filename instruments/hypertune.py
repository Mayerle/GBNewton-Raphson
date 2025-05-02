import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time
from model.NDModel import *
from instruments.datasample import *
from instruments.dftools import *
from instruments.statistic import *
from instruments.view import *

def cross_validate(data, folds, depth, leaf_size, count,lr, l1, l2):
    train_objects, test_objects, train_targets, test_targets = data
    data_weights = np.ones(4)
    train_cross_entropies = []
    validate_cross_entropies = []

    for fold in folds:
        train_objects, train_targets, validate_objects, validate_targets = fold

        model = NDModel(depth=depth, 
                        leaf_size=leaf_size,
                        l1 = l1,
                        count=count,
                        lr=lr,
                        l2 = l2,
                        weights = data_weights
                        )
        model.fit(train_objects, train_targets)

        train_predictions  = model.predict(train_objects)
        validate_predictions = model.predict(validate_objects)


        train_cross_entropy = cross_entropy(train_targets, train_predictions,data_weights)
        train_cross_entropies.append(train_cross_entropy)
        validate_cross_entropy = cross_entropy(validate_targets, validate_predictions,data_weights)
        validate_cross_entropies.append(validate_cross_entropy)


    macro_train_ce     = np.mean(train_cross_entropies)
    macro_validate_ce  = np.mean(validate_cross_entropies)

    print(f"              | Train Validate")
    print(f"Cross Entropy | {macro_train_ce:.2f}  {macro_validate_ce:.2f}")

def random_search(iter:int, random_state:int, depths: np.ndarray, leaf_sizes: np.ndarray, counts: np.ndarray, lrs: np.ndarray, l1s: np.ndarray, l2s: np.ndarray):
    data = get_data(rank=1)
    train_objects, test_objects, train_targets, test_targets = data
    folds = split1R_folds(train_objects, train_targets)
    generator = np.random.default_rng(seed=random_state)
    results = np.zeros((iter,4))

    params = []
    for depth in depths:
        for leaf_size in leaf_sizes:
            for count in counts:
                for lr in lrs:
                    for l1 in l1s:
                        for l2 in l2s:
                            params.append([depth, leaf_size, count, lr, l1, l2])
    generator.shuffle(params,axis=0)

    for i in range(iter):
        t0 = time.time()
        print(f"Iter {i+1}/{iter}")
        depth, leaf_size, count, lr, l1, l2 = params[i]

        ce, acc, f1 = cross_validate(data, folds, depth, leaf_size, count, lr, np.ones(4)*l1, np.ones(4)*l2)
        
        t1 = time.time()
        dt = t1-t0
        print(f"Iter time: {dt:.0f}s")
        results[i] = np.array([dt, ce, acc, f1])
    return results, params

def cv(all_params):
        data,folds, params, i = all_params
        t0 = time.time()
        print(f"Iter {i+1}")
        depth, leaf_size, count, lr, l1, l2 = params[i]
        ce, acc, f1 = cross_validate(data, folds, depth, leaf_size, count, lr, np.ones(4)*l1, np.ones(4)*l2)
        t1 = time.time()
        dt = t1-t0
        return np.array([dt, ce, acc, f1])


def random_search_mt(iter:int, random_state:int, depths: np.ndarray, leaf_sizes: np.ndarray, counts: np.ndarray, lrs: np.ndarray, l1s: np.ndarray, l2s: np.ndarray):
    data = get_data(rank=1)
    train_objects, test_objects, train_targets, test_targets = data
    folds = split1R_folds(train_objects, train_targets)
    generator = np.random.default_rng(seed=random_state)

    params = []
    for depth in depths:
        for leaf_size in leaf_sizes:
            for count in counts:
                for lr in lrs:
                    for l1 in l1s:
                        for l2 in l2s:
                            params.append([depth, leaf_size, count, lr, l1, l2])
    generator.shuffle(params,axis=0)

    mt_params = []
    for i in range(iter):
        mt_params.append([data,folds,params,i])
        
    with Pool(processes=cpu_count()) as pool:
        results = np.array(pool.map(cv, mt_params))
    return results, params


def random_search_mt_l1l2(iter:int, 
                          random_state:int,
                          l10: np.ndarray,
                          l11: np.ndarray,
                          l12: np.ndarray,
                          l13: np.ndarray,

                          l20: np.ndarray,
                          l21: np.ndarray,
                          l22: np.ndarray,
                          l23: np.ndarray
                          ):
    data = get_data(rank=1)
    train_objects, test_objects, train_targets, test_targets = data
    folds = split1R_folds(train_objects, train_targets)
    generator = np.random.default_rng(seed=random_state)

    params = []
    for a in l10:
        for b in l11:
            for c in l12:
                for d in l13:
                    for x in l20:
                        for y in l21:
                            for z in l22:
                                for w in l23:
                                    l1 = np.array([a,b,c,d])
                                    l2 = np.array([x,y,z,w])
                                    params.append([3, 3, 200, 0.05, l1, l2])
    print(f"All combinations: {len(params)}")
    generator.shuffle(params,axis=0)

    mt_params = []
    
    for i in range(iter):
        mt_params.append([data,folds,params,i])
    t0 = time.time()    
    with Pool(processes=cpu_count()) as pool:
        results = np.array(pool.map(cv, mt_params))
    t1 = time.time()
    print(f"Total Time: {t1-t0:.0f}")
    return results, params
