import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pandas as pd

# cm_student = pd.read_excel(
#     'output/distillation.xlsx', 'pretrain-googlenet-23')
# header = list(cm_student.columns)[:25]
# cm_student = np.array(cm_student)[:25, :25]


# distance_array = []
# for i in range(25):
#     for j in range(i, 25):
#         if i != j:
#             dis = 1 - max(cm_student[i, j], cm_student[j, i]) / 50
#             distance_array.append(dis)

# X = np.array(distance_array)
# labels = header
# dendrogram = sch.dendrogram(sch.linkage(X, method="single"), labels=labels)
# plt.title('Dendrogram')
# plt.xlabel('Classes')
# plt.ylabel('1 - Error rate')
# plt.savefig('test.png')

N = 102
cm = pd.read_csv(
    '../../../data/attr_net/outputs/eval_model/confusion_matrix.csv')
cm = cm.iloc[27:, 28:]

header = list(cm.columns)
cm = np.array(cm)
print(cm)


# distance_array = []
# for i in range(25):
#     for j in range(i, 25):
#         if i != j:
#             dis = 1 - max(cm_student[i, j], cm_student[j, i]) / 50
#             distance_array.append(dis)

print("get matrix")
X = 1 - np.maximum(cm, cm.T ) / 50
X = X[np.triu_indices(N, 1)]
# X = np.array(distance_array)
labels = header
print("start model")
dendrogram = sch.dendrogram(sch.linkage(X, method="single"), labels=labels)

class_id = np.array(header)[dendrogram['leaves']]
with open("class_id.txt", "w") as f:
    class_id_str = np.array_repr(class_id)
    f.write(class_id_str)
plt.title('Dendrogram')
plt.xlabel('Classes')
plt.ylabel('1 - Error rate')
plt.savefig('test.png')






cm = pd.read_csv(
    '../../../data/attr_net/outputs/eval_model/confusion_matrix.csv')
cm = cm.iloc[27:, 28:]

name2id = {n: i for i, n in enumerate(cm.columns[:])}

new_order = class_id
new_index = [name2id[s] for s in class_id]

cm = cm[list(new_order)]

print(cm)
print(new_index)
cm = cm.iloc[new_index, :]

# cm.columns = list(new_order)
cm.to_csv('test.csv')