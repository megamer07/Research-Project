# import the libraries
# --------------------
import pandas as pd
# import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
# from sklearn import preprocessing as pp
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,roc_curve, roc_auc_score
from sklearn import tree
import pydotplus
import collections
# read the input file
# --------------------
path="C:/Users/Arman Chand/Desktop/Project works/Random_Forest/Churn_Modelling.csv"
bc = pd.read_csv(path)


# drop unwanted columns
# --------------------
bc = bc.drop(['RowNumber','CustomerId','Surname'],axis=1)
# split the dataset into X and Y variables. These are numpy array
# -----------------------------------------------------------------
X = bc.iloc[:, :10].values
y = bc.iloc[:, 10:].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# split the dataset into train and test
# -------------------------------------
x_train, x_test, y_train, y_test = \
    train_test_split( X, y, test_size = 0.3, random_state = 100)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# build the model
# --------------------------
clf1 = RandomForestClassifier(n_estimators=30,criterion='gini')
fit1 = clf1.fit(x_train,y_train)
print(fit1)

# predict
# --------
pred1 = fit1.predict(x_test)

# print some predictions
# -----------------------
for i in range(50):
    print("Actual value = {}, Predicted value = {}".format(y_test[i], pred1[i]))

# print the accuracy
# ------------------
print("Test Accuracy  :: ", accuracy_score(y_test, pred1))

# confusion matrix
# ----------------
#print(confusion_matrix(y_test, pred1))
import pylab as plt
labels=[2,4]
cm = confusion_matrix(pred1, y_test)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn import metrics

##Computing false and true positive rates
fpr, tpr,_=roc_curve(pred1,y_test,drop_intermediate=False)
auc = metrics.roc_auc_score(y_test, pred1)

import matplotlib.pyplot as plt
plt.figure()
##Adding the ROC
plt.plot(fpr, tpr, color='red',
 lw=2, label="ROC curve auc="+str(auc))
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()

'''
data_feature_names = [ 'CreditScore', 'Geography', 'Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMembe','EstimatedSalary']

# Visualize data
dot_data = tree.export_graphviz(classifier,
                                feature_names=data_feature_names,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree1.png')

'''
