
import matplotlib.pyplot as plt
import time
from model.NDModel import *
from instruments.datasample import *
from instruments.dftools import *
from instruments.statistic import *
from instruments.view import *
from model.Baseline import BaseLineModel
from instruments.random_search import random_search, random_search_mt
np.set_printoptions(precision=2,formatter={'float': '{: 0.2f}'.format})
if __name__ == "__main__":
    results, params = random_search_mt(
                                iter = 12*3,
                                random_state=42,
                                depths= [3, 4],
                                leaf_sizes=[3,5,7],
                                counts=[50,100,200],
                                l1s=[0.1, 0.5, 0.7],
                                l2s=[1, 5, 10, 15],
                                lrs=[0.01, 0.05, 0.1]
                                )
    ce = results[:,1]
    best_ce_index = np.argmin(ce)
    depth, leaf_size, count, lr, l1, l2 = params[best_ce_index]

    print(f"Best params\ndepth: {depth}\nleaf_size: {leaf_size}\ncount: {count}\nlr: {lr}\nl1: {l1*np.ones(4)}\nl2: {l2*np.ones(4)}")


    fig, axe = plt.subplots()
    axe.plot(ce)
    axe.set_ylabel("Cross Entropy")
    axe.set_xlabel("Iter")
    axe.set_xticks(ticks=range(0,ce.shape[0]))
    plt.show()
