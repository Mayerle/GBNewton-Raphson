import pandas as pd
import numpy as np
from instruments.dftools import train_test_split, convert_targets_to_vector

def get_data(rank: int = 0) -> list[np.ndarray]:
    random_state = 42
    df = pd.read_csv("cirrhosis.csv").sample(frac=1,random_state=random_state)
    df = df[df["Stage"].notna()]
    replaces = {
                "D-penicillamine": 0, 
                "Placebo": 1,
                "N": 0, 
                "Y": 1,
                "C": 0, 
                "D": 1, 
                "CL": 2,
                "F": 0, 
                "M": 1,
                "S": 2
                }
    with pd.option_context("future.no_silent_downcasting", True):
        df = df.replace(replaces).infer_objects(copy=False)

    features_columns = [
                        "N_Days", "Status", "Age",  "Sex","Ascites",
                        "Hepatomegaly","Spiders","Edema","Bilirubin",
                        "Cholesterol","Albumin","Copper","Alk_Phos",
                        "SGOT","Tryglicerides","Platelets","Prothrombin"
                       ]
    target_column = "Stage"
    objects = df[features_columns].to_numpy()
    targets = df[target_column].to_numpy()
    classes = np.unique(targets)

    train_objects_list = []
    test_objects_list = []
    train_targets_list = []
    test_targets_list = []
    for class_ in classes:
        targets_indexes = np.where(targets == class_)
        train_objects, test_objects = train_test_split(objects[targets_indexes])
        train_targets, test_targets = train_test_split(targets[targets_indexes])
        train_objects_list.append(train_objects)
        test_objects_list.append(test_objects)
        train_targets_list.append(train_targets)
        test_targets_list.append(test_targets)

    train_objects = np.concatenate(train_objects_list, axis = 0) 
    test_objects =  np.concatenate(test_objects_list,  axis = 0) 
    train_targets = np.concatenate(train_targets_list, axis = 0) 
    test_targets =  np.concatenate(test_targets_list,  axis = 0) 



    if(rank == 0):
        return train_objects, test_objects, train_targets, test_targets
    elif(rank == 1):
        rank1_train_targets = convert_targets_to_vector(train_targets)
        rank1_test_targets  = convert_targets_to_vector(test_targets)
        return train_objects, test_objects, rank1_train_targets, rank1_test_targets  
    else:
        raise ValueError("Unsupported rank!")



def split1R_folds(objects: np.ndarray, targets: np.ndarray, fold_numbers = 3) -> list:
    targets_1D = np.argmax(targets, axis=1)
    classes = np.unique(targets_1D)
    classed_splits = []
    for class_ in classes:
        indexes = np.where(targets_1D == class_)[0]
        splits = np.array_split(indexes,fold_numbers)
        classed_splits.append(splits)
    folds = []

    for i in range(fold_numbers):
            fold_splits = []
            validates = []
            for class_ in classes:
                for j in range(fold_numbers):
                    if(i != j):
                        fold_splits.append(classed_splits[class_][j])
                    else:
                        validates.append(classed_splits[class_][j])
            train_indexes    = np.concatenate(fold_splits,  axis = 0)
            validate_indexes = np.concatenate(validates,  axis = 0)
            train_objects    = objects[train_indexes]
            train_targets    = targets[train_indexes]
            validate_objects = objects[validate_indexes]
            validate_targets = targets[validate_indexes]
            fold = [train_objects, train_targets, validate_objects, validate_targets]
            folds.append(fold)
    return folds