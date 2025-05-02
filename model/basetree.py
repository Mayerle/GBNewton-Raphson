import numpy as np
import pandas as pd
from typing import Self
from abc import ABC, abstractmethod
from collections.abc import Callable
from instruments.mathfunctions import *
type LeafIndexer = Callable[[np.ndarray],int]


class BaseTree(ABC):
    
    def __init__(self, objects: np.ndarray, antigradients: np.ndarray, hessians: np.ndarray, **kwargs):
        self._objects         = objects
        self._antigradients    = antigradients
        self._hessians = hessians
        self.__leafs           = []
        
        self.__is_splitable    = True
        self.__leaf_size   = kwargs["leaf_size"]

        self.__feature_index   = None
        self.__feature_value   = None
        self.__leaf_value      = None
        
        self.__data = kwargs
        self.__is_computed = False
    def split(self) -> None:
        
        if(self.__is_splitable == False):
            return
        
        if(self.__is_terminal()):
            
            if(self._objects.shape[0] < self.__leaf_size):
                
                self.__is_splitable = False
                return 
            best_gain         = 0
            best_feature_value = None
            best_feature_index = None
            best_indexes_list = None

            sample_loss = self._calculate_loss(self._antigradients, self._hessians)
            
            for feature_index in range(self._objects.shape[1]):
                values = self._objects[:,feature_index]
                unique_values = find_uniques(values)

                for feature_value in unique_values:
                    if(np.isnan(feature_value)):
                        continue


                    indexes_list = self.__find_split_indexes(feature_value, values)
                    leafs_loss = 0
                    for indexes in indexes_list:
                        leafs_loss += self._calculate_loss(self._antigradients[indexes], self._hessians[indexes])

                    gain = leafs_loss-sample_loss
                    #print(gain)
                    if(best_gain < gain):
                        best_gain = gain
                        best_feature_value = feature_value
                        best_feature_index = feature_index
                        best_indexes_list = indexes_list
            
            if(self._is_sufficient_split(sample_loss, best_gain) == False):
                self.__is_splitable = False
                return
        

            left_indexes, right_indexes = best_indexes_list

            if( (len(left_indexes[0]) < self.__leaf_size) or 
                (len(right_indexes[0]) < self.__leaf_size)
              ):
                self.__is_splitable = False
                return
            
            left_leaf = self._create_leaf(self._objects[left_indexes],
                                          self._antigradients[left_indexes],
                                          self._hessians[left_indexes],
                                          self.get_data(self.__data)
                                          ) 
            right_leaf = self._create_leaf(self._objects[right_indexes],
                                          self._antigradients[right_indexes],
                                          self._hessians[right_indexes],
                                          self.get_data(self.__data)
                                          ) 
            self.__leafs = [left_leaf, right_leaf]
            self._objects = None
            self._antigradients = None
            self._hessians = None
            self.__feature_index = best_feature_index
            self.__feature_value = best_feature_value


        else:
            
            for leaf in self.__leafs:
                leaf.split()
    
    def predict(self, obj: np.ndarray) -> np.ndarray:     
        
        if(self.__is_terminal()):
            return self.__get_leaf_value()
        else:
            value = obj[self.__feature_index]
            if(np.isnan(value)):
                return self.__leafs[1].predict(obj)
            else:
                if(value <= self.__feature_value):
                    return self.__leafs[0].predict(obj)
                else:
                    return self.__leafs[1].predict(obj)
    def __get_leaf_value(self):
        if(self.__is_computed == False):
            self.__leaf_value = self.find_leaf_value() 
            self.__is_computed = True
        return self.__leaf_value               
    def __find_split_indexes(self, feature_value: float, objects: np.ndarray) -> list:      
        if(np.isnan(feature_value)):
            raise ValueError("Feature value is nan!")
        
        left_indexes  = np.where(objects <= feature_value)
        right_indexes = np.where( (objects > feature_value) | (objects != objects) )
        return left_indexes, right_indexes
    
    def __is_terminal(self):
        return len(self.__leafs) == 0
    
    @abstractmethod 
    def get_data(self, data) -> list:
        pass
    @abstractmethod
    def find_leaf_value(self) -> None:
        pass
    @abstractmethod
    def _create_leaf(self, objects: np.ndarray, antigradients: np.ndarray, hessians: np.ndarray, data: list):
        pass
    @abstractmethod
    def _calculate_loss(self, antigradients: np.ndarray, hessians: np.ndarray):
        pass
    @abstractmethod
    def _is_sufficient_split(self, sample_loss: float, best_gain: float) -> bool:
        pass
    
    def save(self, number, data = [], i = 0, position = None):
        is_terminal = self.__is_terminal()
        feature_value = self.__feature_value
        feature_index = self.__feature_index
        
        __save_columns= ["Depth", "IsTerminal", "Position", "FeatureIndex", "FeatureValue", "Value0", "Value1", "Value2", "Value3","Number"]


        if(is_terminal):
            value = self.__get_leaf_value()
            data.append([i, is_terminal, position,  np.nan,  np.nan, value[0],value[1],value[2],value[3],number])
        else:
            data.append([i, is_terminal, position, feature_index, feature_value, np.nan, np.nan, np.nan, np.nan,number])
            self.__leafs[0].save(number, data, i+1, "left")
            self.__leafs[1].save(number, data, i+1, "right")
        
    def load(self, data, i = 0, position = np.nan):
        dat = None
        if(i == 0):
            dat = data[data["Depth"] == 0].iloc[0,:]
        else:
            dat = data[ (data["Depth"] == i) & (data["Position"] == position) ].iloc[0,:]
        
        if(dat["IsTerminal"]):
            self.__leaf_value = dat[["Value0", "Value1", "Value2", "Value3"]].to_numpy().astype(dtype=np.float32)
            self.__is_computed = True
        else:
            self.__feature_index = int(dat["FeatureIndex"])
            self.__feature_value = float(dat["FeatureValue"])
            left_leaf = self._create_leaf(None, None,None,self.__data)
            right_leaf = self._create_leaf(None, None,None,self.__data)
            self.__leafs = [left_leaf, right_leaf]
            left_leaf.load(data, i+1,"left")
            right_leaf.load(data, i+1,"right")
        
        
