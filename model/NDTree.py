from model.basetree import *



class NDTree(BaseTree):
    def __init__(self, objects, antigradients, hessians, **kwargs):
        super().__init__(objects, antigradients, hessians, **kwargs)
        self.__l2 = kwargs["l2"]
        self.__l1 = kwargs["l1"]

    def find_leaf_value(self) -> float:
        anti_g = np.sum(self._antigradients,axis = 0 )
        l2 = np.zeros((self._antigradients.shape[1],self._antigradients.shape[1]))
        for i in range(self._antigradients.shape[1]):
            l2[i,i] = self.__l2[i]

        h = np.sum(self._hessians,axis = 0 ) + l2
        inv_h = np.linalg.inv(h)

        raw_logit = np.matmul(inv_h,anti_g)
        delta = np.abs(raw_logit) - np.matmul(inv_h, self.__l1)
        logit = np.sign(raw_logit)*np.max([np.zeros(raw_logit.shape[0]), delta], axis=0)
        return logit


    def _create_leaf(self, objects: np.ndarray, antigradients: np.ndarray, hessians: np.ndarray, data: list):
        return NDTree(objects, antigradients, hessians, **data)
    
    def _calculate_loss(self, antigradients: np.ndarray, hessians: np.ndarray):
        if(antigradients.shape[0] == 0):
            return 0
        anti_g = np.sum(antigradients,axis = 0 )

        l2 = np.zeros((antigradients.shape[1],antigradients.shape[1]))
        for i in range(antigradients.shape[1]):
            l2[i,i] = self.__l2[i]
        
        h = np.sum(hessians,axis = 0 ) + l2
        inv_h = np.linalg.inv(h)

        loss = np.matmul(np.matmul(inv_h,anti_g),anti_g)
        return loss

    def _is_sufficient_split(self, sample_loss: float, best_gain: float) -> bool:
        return best_gain > 0
    
    def get_data(self, data) -> list:
        return data
   
