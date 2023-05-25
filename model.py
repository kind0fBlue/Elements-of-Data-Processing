from pandas.core.indexes import interval
import wrangled
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import csv

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#Ziming, display output fully with no collapse
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None) 
pd.set_option('display.expand_frame_repr', False) 
pd.set_option('display.max_colwidth', 1000)

#build trained data frame
data = pd.DataFrame()
data['population density'] = list(wrangled.populationDensity2020)
data['first dose'] = list(wrangled.first_dose_rate)
data['internal arrival'] = list(wrangled.internal_arrival) #notice:the wrangled data has index, result in disorder, so list() is safer, 
data.index = (list(range(len(wrangled.first_dose_rate))))

#our class label(covid), binned
covid = wrangled.covid_rate
covid = np.array(covid).reshape(-1, 1)
binner = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='uniform')
covid_bin = binner.fit_transform(covid)
covid_bin = covid_bin.flatten()
covid_bin = pd.Series(covid_bin).astype(int)

#knn and decision tree
knn_file = open('model_output/KNN.csv', 'w', newline='')
knnWriter = csv.writer(knn_file)
knnWriter.writerow(["KNN output"])
tree_file = open('model_output/tree.csv', 'w', newline = '')
treeWriter = csv.writer(tree_file)
treeWriter.writerow(["Tree output"])
accuracyKnn = 0
accuracyTree = 0
iterTimes = 43
for i in range(0, iterTimes):
    #split
    x_train, x_test, y_train, y_test = train_test_split(data,covid_bin, train_size=0.8, test_size=0.2, random_state=i)
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train=scaler.transform(x_train)
    x_test=scaler.transform(x_test)
    #knn
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)
    y_pred_KNN=knn.predict(x_test)
    knnWriter.writerow(['accuracy score:',accuracy_score(y_test, y_pred_KNN)])
    knnWriter.writerow(['Y_test: '] + list(y_test))
    knnWriter.writerow(['Y_pred'] + list(y_pred_KNN))
    knnWriter.writerow([' '])
    accuracyKnn += accuracy_score(y_test, y_pred_KNN)
    #tree
    dt = DecisionTreeClassifier(criterion="entropy",random_state=i)
    dt.fit(x_train, y_train)
    y_pred_tree=dt.predict(x_test)
    treeWriter.writerow(['accuracy score:',accuracy_score(y_test, y_pred_tree)])
    treeWriter.writerow(['Y_test: '] + list(y_test))
    treeWriter.writerow(['Y_pred'] + list(y_pred_tree))
    treeWriter.writerow([' '])
    accuracyTree += accuracy_score(y_test, y_pred_tree)
print('average accuracy for knn: ', accuracyKnn/iterTimes)
print('average accuracy for Tree: ', accuracyTree/iterTimes)
knn_file.close()
tree_file.close()

#visualize k-neighbour
#split again, use 0 as seed
x_train, x_test, y_train, y_test = train_test_split(data,covid_bin, train_size=0.8, test_size=0.2, random_state=0)
scaler = preprocessing.StandardScaler().fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
ks = range(1,8,1)
accu_list=[]
for k in ks:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train) 
    y_pred=knn.predict(x_test)
    accu_list.append(accuracy_score(y_test, y_pred))
plt.plot(ks,accu_list)
plt.title('K-neighbour, varying k versus the prediction accuracy')
plt.xlabel('k')
plt.ylabel('prediction accuracy')
plt.savefig('graph/Sample k-NN accuracy visualization', dpi=300)
plt.clf()

#visualize decision tree
export_graphviz(dt, out_file="mytree.dot",feature_names = data.columns,filled=True,rounded=True)
with open("./mytree.dot") as f:
   dot_graph = f.read()
graphviz.Source(dot_graph)

########################################################################
#linear regression, notice: use covid, dont use covid_bin as input!!!
lm_file = open('model_output/lm.csv', 'w', newline='')
lmWriter = csv.writer(lm_file)
lmWriter.writerow(["linear regression output"])
r2Test = 0
r2Train = 0
coef1 = 0
coef2 = 0
coef3 = 0
intercept = 0
accuracySum = 0
for i in range(0, iterTimes):
    x_train_lm, x_test_lm, y_train_lm, y_test_lm = train_test_split(data, covid,train_size = 0.8, test_size=0.2, random_state=i)
    lm = linear_model.LinearRegression()
    model = lm.fit(x_train_lm, y_train_lm)
    y_pred_lm = lm.predict(x_test_lm)
    lmWriter.writerow(["test"] + list(y_test_lm.flatten()))
    lmWriter.writerow(["pred"] + list(y_pred_lm.flatten()))
    lmWriter.writerow(["coefficient", lm.coef_[0][0], lm.coef_[0][1], lm.coef_[0][2]])
    lmWriter.writerow(['Intercept: ', lm.intercept_])
    lmWriter.writerow(['r2 train: ', lm.score(x_train_lm, y_train_lm)])
    lmWriter.writerow(['r2 test: ', lm.score(x_test_lm, y_test_lm)])
    
    r2Train += lm.score(x_train_lm, y_train_lm)
    r2Test += lm.score(x_test_lm, y_test_lm)
    coef1 += lm.coef_[0][0]
    coef2 += lm.coef_[0][1]
    coef3 += lm.coef_[0][2]
    intercept += lm.intercept_
    accuracy = 0
    for j in range(0, len(list(y_test_lm.flatten()))):
        if abs(list(y_test_lm.flatten())[j] - list(y_pred_lm.flatten())[j]) <= 100:
            accuracy += 1
    accuracy = accuracy/len(list(y_test_lm.flatten()))
    accuracySum += accuracy
    lmWriter.writerow(['accuracy: ', accuracy])
    lmWriter.writerow([' '])

print('mean Coefficient of determination (training): ', r2Train/iterTimes)
print('mean Coefficient of determination (testing): ', r2Test/iterTimes)
print('mean coefficient of LM is: ', coef1/iterTimes,' ', coef2/iterTimes,' ', coef3/iterTimes)
print('mean intercept is: ', intercept/iterTimes)
print('mean accuracy: ', accuracySum/iterTimes)
lm_file.close()
