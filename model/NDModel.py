from model.basetree import *
from model.NDTree import *


class NDModel: 
    def __init__(self, **kwargs):
        self._leaf_size = None
        self._weights = np.ones(4)
        self._depth = None
        self._count = None
        self._lr = None
        self._l2 = None
        self._l1 = None
        self._history = []
        self._trees = []

        if("leaf_size" in kwargs.keys()):
            self._leaf_size = kwargs["leaf_size"]
        if("weights" in kwargs.keys()):
            self._weights = kwargs["weights"]
        if("depth" in kwargs.keys()):
            self._depth = kwargs["depth"]
        if("count" in kwargs.keys()):
            self._count = kwargs["count"]
        if("lr" in kwargs.keys()):
            self._lr = kwargs["lr"]
        if("l2" in kwargs.keys()):
            self._l2 = kwargs["l2"]
        if("l1" in kwargs.keys()):
            self._l1 = kwargs["l1"]

        

    def predict(self, objects: np.ndarray) -> np.ndarray:
        logits = np.zeros((objects.shape[0], self._start_logits.shape[0]))
        logits +=  self._start_logits
        for i in range(objects.shape[0]):
            logit = np.zeros(self._start_logits.shape[0])
            for tree in self._trees:
                logit += self._lr*tree.predict(objects[i])
            logits[i] = logit
        
        return softmax_array(logits)
    
    def get_history(self):
        return self._history
    
    def fit(self, objects: np.ndarray, targets: np.ndarray):
        self._history = []
        self._trees = []

        self._find_start_logits(targets)
        for i in range(self._count):
            predictions = self.predict(objects)
            ce = cross_entropy(targets, predictions, self._weights)
            self._history.append(ce)

            antigradients = self._calculate_antigradient(targets, predictions)
            hessians = self._calculate_hessians(predictions)
            
            tree = NDTree(objects, antigradients, hessians, leaf_size=self._leaf_size, l2 =self._l2, l1 = self._l1)
            for _ in range(self._depth):
                tree.split()
            self._trees.append(tree)

        predictions = self.predict(objects)
        ce = cross_entropy(targets, predictions, self._weights)
        self._history.append(ce)      
 
    def save(self, path="weights/NDModel.csv"):
        __save_columns= ["Depth", "IsTerminal", "Position", "FeatureIndex", "FeatureValue", "Value0", "Value1", "Value2", "Value3","Number"]

        all_dat = []
        for j in range(len(self._trees)):
            tree = self._trees[j]
            tree.save(j, all_dat)
            
        data = pd.DataFrame(all_dat, columns = __save_columns)

        model_data = np.array([self._start_logits[0],
                               self._start_logits[1],
                               self._start_logits[2],
                               self._start_logits[3],
                               self._weights[0],
                               self._weights[1],
                               self._weights[2],
                               self._weights[3],
                               self._lr,
                               np.nan
                               ])
        data.loc[data.shape[0]] = np.array(model_data)
        data.to_csv(path, index=False)

    def load(self, path="weights/NDModel.csv"):
        self._trees = []
        data = pd.read_csv(path)
        model_data = data.iloc[-1,:]
        tree_data = data.iloc[:-1,:]
        
        self._start_logits = model_data.iloc[:4].to_numpy().astype(dtype=np.float32)
        self._weights = model_data.iloc[4:8].to_numpy().astype(dtype=np.float32)
        self._lr = float(model_data.iloc[8])
        max_number = tree_data["Number"].max()

        for i in range(int(max_number)+1):
            dat = tree_data[tree_data["Number"] == i].iloc[:,:-1]
            tree = NDTree(None, None, None, leaf_size=None, l2=None, l1=None)
            tree.load(dat) 
            self._trees.append(tree)

    def _calculate_antigradient(self, targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        return -(predictions-targets*self._weights)
    
    def _find_start_logits(self, targets: np.ndarray) -> np.ndarray:
        odds = np.sum(targets, axis=0)/np.sum(targets)
        self._start_logits = np.log(odds)
    
    def _find_predictions(self, logits: np.ndarray)-> np.ndarray:
        return softmax_array(logits)

    def _calculate_hessians(self, predictions: np.ndarray):
        hessians = np.zeros( (predictions.shape[0],predictions.shape[1],predictions.shape[1]) )
        for i in range(predictions.shape[0]):
            for j in range(predictions.shape[1]):
                for k in range(predictions.shape[1]):
                    h = 0
                    if(j == k):
                        h = predictions[i,j]*(1-predictions[i,j])
                    else:
                        #h = -predictions[k,j]*predictions[j,k] 
                        h = -predictions[i,j]*predictions[i,k] 
                    hessians[i,j,k] = h 
        return hessians

    def _get_logits(self, tree: BaseTree, objects: np.ndarray) -> np.ndarray:
        logits = np.zeros((objects.shape[0], self._start_logits.shape[0]))
        for i in range(objects.shape[0]):
            logits[i] = tree.predict(objects[i])
        return logits