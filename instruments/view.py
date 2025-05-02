
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText      
      
def show_confusion_matrix(matrix: np.ndarray, labels: list, show_values = True, cmap="cividis") -> None:
    fig, axe = plt.subplots()
    graph = axe.matshow(matrix,cmap = cmap)
    axe.set_xlabel("Predicted class")
    axe.set_ylabel("Observed class")
    axe.set_xticklabels(labels)
    axe.set_yticklabels(labels)
    if(show_values):
        for (i, j), z in np.ndenumerate(matrix):
            axe.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    fig.colorbar(graph, ax = axe) 
    
def print_all_per_class(data: list, labels: list) -> None:
    line0 = "         :"
    line1 = "Precision:"
    line2 = "Recall   :"
    line3 = "F1       :"
    for dat, label in zip(data,labels):
        line0 += f" {label}   "
        line1 += f" {dat[0]:.2f}"
        line2 += f" {dat[1]:.2f}"
        f1 = 2*dat[0]*dat[1]/(dat[0]+dat[1])
        line3 += f" {f1:.2f}"
    print(line0)
    print(line1)
    print(line2)
    print(line3)
    
def print_all(data: list, label: str) -> None:
        print(label)
        print(f"Accuracy : {data[0]:.2f}")
        print(f"Precision: {data[1]:.2f}")
        print(f"Recall   : {data[2]:.2f}")
        print("\n")

def plot_all(statistics: dict, digits: int) -> list:
    metrics = ("Accuracy", "Precision", "Recall")
    statistics_rounded = {}
    for m, y in statistics.items():
        statistics_rounded.update({m:[round(k,digits) for k in y]}) 
        
    x = np.arange(len(metrics))  # the label locations
    width = 0.15 
    multiplier = 0

    fig, axe = plt.subplots(layout='constrained')

    for model, value in statistics_rounded.items():
        offset = width * multiplier
        rects = axe.bar(x + offset, value, width, label=model)
        axe.bar_label(rects, padding=3)
        multiplier += 1

    axe.set_ylabel('Metric')
    axe.set_title('Model comparison')
    axe.set_xticks(x + width, metrics)
    axe.legend(loc='upper left', ncols=len(metrics))
    axe.set_ylim(0.8, 1)   
    return (fig,axe)

def plot_all_T(statistics: list, bars: list,title: str = "", digits: int= 2, ylim: list = [0.8,1],margin = 5,width =1) -> list:   
    fig, axe = plt.subplots(layout='constrained')
    statistics = [[round(x,digits) for x in y]  for y in statistics]
    
    padding = 0.2
    labels = ["Accuracy","Precision","Recall"]
    colors = ["#D6C6AD","#D20222","#45666B"]

    count = len(statistics)
    base_space = np.linspace(-1,1,count)
    for i in range(count):
        base = base_space[i]
        apr = statistics[i]
        position = (width+padding)*np.linspace(-1,1,3) + margin*base*np.ones(3)
        rects = axe.bar(position, apr, width, label=labels,color=colors)
        axe.bar_label(rects, padding=3)


    axe.set_xticks(margin*base_space, bars)
    axe.set_ylim(*ylim) 
    axe.legend(loc='upper left', ncols=3,labels=labels)
    factor = 0.8
    fig.set_size_inches(16*factor, 9*factor, forward=True)
    axe.set_title(title)


def plot_roc(targets, predictions, title, macro_params = []):
    fig, axe = plt.subplots()


    for i in range(4):
        tars = targets[:,i]
        preds = predictions[:,i]

        sorted_ids = np.argsort(preds)
        preds = preds[sorted_ids]
        tars = tars[sorted_ids]
        xs = []
        ys = []
        point = np.zeros(2)
        for j in range(preds.shape[0]):
            if(tars[j] == 1):
                point[1] += 1
            else:
                point[0] += 1
            xs.append(point[0])
            ys.append(point[1])
        xs = np.array(xs)
        ys = np.array(ys)
        xs = xs/(1+max(xs))
        ys = ys/(1+max(ys))
        
        auc = 0
        for i in range(xs.shape[0]-1):
            auc += ys[i]*(xs[i+1]-xs[i])

        axe.plot(xs,ys, label = f"class {i} | AUC: {auc:.2f}")
    
    axe.legend()
    txt = f"Marco CE:   {macro_params[0]:.2f}\nMarco F1:   {macro_params[1]:.2f}\nMarco ACC: {macro_params[2]:.2f}"

    text_box = AnchoredText(txt, frameon=True, loc=4, pad=0.5)
    plt.setp(text_box.patch, facecolor='white', alpha=0.5)
    fig.gca().add_artist(text_box)
    
    axe.set_xlabel("FPR")
    axe.set_ylabel("TPR")
    axe.set_title(title)

def plot_history(histroy, title):
    fig, axe = plt.subplots()

    axe.plot(histroy)
    axe.legend()
    axe.set_xlabel("Iter")
    axe.set_ylabel("Cross Entropy")
    axe.set_title(title)

