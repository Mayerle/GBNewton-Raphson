import numpy as np

class ClassificationStatistics:
    def __init__(self, observations: np.ndarray, predictions: np.ndarray):
        self.observations = observations
        self.predictions = predictions
        self.confusion_matrix = None
        
    def __check_matrix(self) -> None:
        if self.confusion_matrix is None:
            raise TypeError("Confusion matrix have not calculated!")
        
    def calculate_confusion_matrix(self) -> np.ndarray:
        uniques =  np.unique(self.observations)
        class_count = len(uniques)
        confusion_matrix = np.zeros((class_count, class_count))
        for observation, prediction in zip(self.observations,self.predictions):
            confusion_matrix[int(observation), int(prediction)] += 1
        self.confusion_matrix = confusion_matrix
        return confusion_matrix

    def calculate_accuracy(self) -> float:
        self.__check_matrix()
        diagonal = self.confusion_matrix.diagonal()
        all_count = self.confusion_matrix.sum()
        correct = diagonal.sum()
        return float(correct/all_count)
    
    def calculate_precisions(self) -> np.ndarray:
        self.__check_matrix()
        diagonal = self.confusion_matrix.diagonal()    
        all_count = self.confusion_matrix.sum(0)
        result = np.zeros(diagonal.shape[0])
        for i in range(diagonal.shape[0]):
            if(diagonal[i] != 0):
                result[i] = diagonal[i]/all_count[i]
        return result
    
    def calculate_recalls(self) -> np.ndarray:
        self.__check_matrix()
        diagonal = self.confusion_matrix.diagonal()    
        all_count = self.confusion_matrix.T.sum(0)
        result = np.zeros(diagonal.shape[0])
        for i in range(diagonal.shape[0]):
            if(diagonal[i] != 0):
                result[i] = diagonal[i]/all_count[i]
        return result
    
    def calculate_all(self) -> list:
        accuracy   = self.calculate_accuracy()
        precisions = self.calculate_precisions()
        recalls    = self.calculate_recalls()
        precision = float(np.mean(precisions))
        recall = float(np.mean(recalls))
        return [accuracy,precision, recall]
    
    def calculate_f_score(self, b: float = 1, blend = "macro") -> float:
        precisions = self.calculate_precisions()
        recalls    = self.calculate_recalls()

        if(blend == "macro"):
            scores = []
            for i in range(precisions.shape[0]):
                if(precisions[i]*recalls[i] == 0):
                    score = 0
                else:
                    score = (1+b**2)*precisions[i]*recalls[i]/((b**2)*precisions[i]+recalls[i])
                scores.append(score)
            return float(np.mean(scores))
        else:
            precision = float(np.mean(precisions))
            recall = float(np.mean(recalls))
            score = (1+b**2)*precision*recall/((b**2)*precision+recall)
            return score
    
def calculate_all_per_class(self) -> list:
    precisions = self.calculate_precisions()
    recalls    = self.calculate_recalls()
    return list(zip(precisions, recalls)) 
def calculate_acc(targets, predictions) -> list[float]:
    acc = np.zeros(4)
    all = np.zeros(4)
    for i in range(targets.shape[0]):
        target = np.argmax(targets[i])
        pred = np.argmax(predictions[i])
        if(pred == target):
            acc[np.argmax(targets[i])] +=1
        all[np.argmax(targets[i])] +=1
    micro = np.sum(acc)/np.sum(all)
    macro = np.mean(acc/all)
    return micro, macro
def matrix2vector(matrix: np.ndarray)->np.ndarray:
    vector = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        vector[i] = np.argmax(matrix[i])
    return vector