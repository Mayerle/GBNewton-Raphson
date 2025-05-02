import matplotlib.pyplot as plt
import time
from model.NDModel import *
from instruments.datasample import *
from instruments.dftools import *
from instruments.statistic import *
from instruments.view import *
class BaseLineModel:
    def fit(self, objects, targets):
        counts = np.sum(targets, axis=0)
        stat_sum = np.sum(counts)
        self._propabilities = counts/stat_sum


    def predict(self, objects):
        id = np.argmax(self._propabilities)
        answers = np.zeros((objects.shape[0],4))
        for i in range(answers.shape[0]):
            answers[i,id] = 1
        return answers