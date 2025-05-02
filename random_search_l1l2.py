
import matplotlib.pyplot as plt
from model.NDModel import *
from instruments.datasample import *
from instruments.dftools import *
from instruments.statistic import *
from instruments.view import *
from instruments.hypertune import random_search_mt_l1l2
np.set_printoptions(precision=2,formatter={'float': '{: 0.2f}'.format})
if __name__ == "__main__":
    results, params = random_search_mt_l1l2(
                                iter = 12*3,
                                random_state=42,
                                l10=[0 ],
                                l11=[0.7],
                                l12=[1],
                                l13=[0 ],

                                l20=[6, 8, 10, 12, 14],
                                l21=[5],
                                l22=[10, 12,14,16,18],
                                l23=[15, 17,19,21,23]
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


# Best params
# depth: 3
# leaf_size: 3
# count: 200
# lr: 0.05
# l1: [ 0.00  0.70  1.00  0.00]
# l2: [ 8.00  5.00  12.00  17.00]

# depth: 3
# leaf_size: 3
# count: 200
# lr: 0.05
# l1: [ 0.00  0.70  1.00  0.00]
# l2: [ 10.00  5.00  18.00  19.00]