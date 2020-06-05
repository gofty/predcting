# load dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import tree
import graphviz
from sklearn.model_selection import cross_val_score
import zipfile

!wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
with zipfile.ZipFile("student.zip","r") as zip_ref:
    zip_ref.extractall("")

d = pd.read_csv('student-por.csv', sep=';')
len(d)
--2019-07-31 09:04:13--  https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip
Resolving archive.ics.uci.edu (archive.ics.uci.edu)... 128.195.10.252
Connecting to archive.ics.uci.edu (archive.ics.uci.edu)|128.195.10.252|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 20478 (20K) [application/x-httpd-php]
Saving to: ‘student.zip.2’

student.zip.2       100%[===================>]  20.00K  --.-KB/s    in 0.06s   

2019-07-31 09:04:14 (319 KB/s) - ‘student.zip.2’ saved [20478/20478]

649

d['pass']=d.apply(lambda row:1 if(row['G1']+row['G2']+row['G3'])>=35 else 0,axis=1)
d.drop(['G2','G1','G3'],axis=1,inplace=True)

d=pd.get_dummies(d,columns=['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
 'nursery', 'higher', 'internet', 'romantic'])
d.head()

#shuffle rows
d=d.sample(frac=1)
#split training amd testing data
d_train = d[:500]
d_test = d[500:]

d_train_att = d_train.drop(['pass'], axis=1)
d_train_pass = d_train['pass']

d_test_att = d_test.drop(['pass'], axis=1)
d_test_pass = d_test['pass']

d_att = d.drop(['pass'], axis=1)
d_pass = d['pass']

#number of passing students in whole dataset
print("Passing: %d out of %d (%.2f%%)" % (np.sum(d_pass), len(d_pass), 100*float(np.sum(d_pass)) / len(d_pass)))

#fit a decision tree
t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
t = t.fit(d_train_att, d_train_pass)

# save tree
dot_data = tree.export_graphviz(t, out_file="student_performance.dot", label="all", impurity=False, proportion=True,
                     feature_names=list(d_train_att), class_names=["fail", "pass"],
                     filled=True, rounded=True)

import pydot
(graph,) = pydot.graph_from_dot_file('student_performance.dot')
graph.write_png('student_performance.png')
# due to the image size it's better to download it and view it on your machine.

scores = cross_val_score(t, d_att, d_pass, cv=5)
#show avarage score and +/- two standered deviations away (covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

for max_depth in range(1, 20):
    t=tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores = cross_val_score(t, d_att, d_pass, cv=5)
    print("Max depth: %d, Accurecy: %0.2f (+/- %0.2f)" % (max_depth, scores.mean(), scores.std() * 2))

    depth_acc = np.empty((19,3), float)
i=0
for max_depth in range(1, 20):
    t=tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores = cross_val_score(t, d_att, d_pass, cv=5)
    depth_acc[i,0] = max_depth
    depth_acc[i,1] = scores.mean()
    depth_acc[i,2] = scores.std() * 2
    i +=1

depth_acc

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.errorbar(depth_acc[:,0], depth_acc[:,1], yerr=depth_acc[:,2])
plt.show()