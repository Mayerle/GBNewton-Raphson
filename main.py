import matplotlib.pyplot as plt
from model.NDModel import *
from instruments.datasample import *
from instruments.dftools import *
from instruments.statistic import *
from instruments.view import *
from model.Baseline import BaseLineModel
np.set_printoptions(precision=2,formatter={'float': '{: 0.2f}'.format})

train_objects, test_objects, train_targets, test_targets = get_data(rank=1)

baseline = BaseLineModel()
baseline.fit(train_objects,train_targets)
base_predictions = baseline.predict(test_objects)
base_cross_entropy = cross_entropy(test_targets, base_predictions)
base_stats = ClassificationStatistics(matrix2vector(train_targets),matrix2vector(base_predictions))
base_confusion = base_stats.calculate_confusion_matrix()
base_f1 = base_stats.calculate_f_score(b=1, blend="macro")
base_acc = base_stats.calculate_accuracy()
print("[BaseLine]")
print(f"Cross-entropy: {base_cross_entropy:.2f}")
print(f"F1 Score:      {base_f1:.2f}")
print(f"Accuracy:      {base_acc:.2f}")
plot_roc(test_targets, base_predictions, "Baseline",[base_cross_entropy,base_f1,base_acc])

model = NDModel()
model.load()
model_predictions = model.predict(test_objects)
model_cross_entropy = cross_entropy(test_targets, model_predictions)
model_stats = ClassificationStatistics(matrix2vector(train_targets),matrix2vector(model_predictions))
model_confusion = model_stats.calculate_confusion_matrix()
model_f1 = model_stats.calculate_f_score(b=1, blend="macro")
model_acc = model_stats.calculate_accuracy()
print("[NDModel]")
print(f"Cross-entropy: {model_cross_entropy:.2f}")
print(f"F1 Score:      {model_f1:.2f}")
print(f"Accuracy:      {model_acc:.2f}")

plot_roc(test_targets, model_predictions, "Model",[model_cross_entropy,model_f1,model_acc])
plt.show()
