import numpy as np 
import matplotlib.pyplot as pt 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#as_matrix converts it into numpy array
data = pd.read_csv("train.csv").as_matrix()

#command to print size of pandas dataframe/ numpy array
print(data.shape)

#higher number of estimators higher the accuracy
clf = DecisionTreeClassifier()
ran = RandomForestClassifier(n_estimators=200)

#dataset
xtrain = data[0:21000,1:]
train_lable = data[0:21000,0]

#clf.fit(data to fit on , result of that data)
clf.fit(xtrain,train_lable)
ran.fit(xtrain,train_lable)

#test_data
xtest = data[21000:,1:]
actual_lable = data[21000:,0]

#uncomment to check particular output
# d = xtest[5]
# d.shape=(28,28)
# pt.imshow(255-d,cmap="gray")
# print(clf.predict([xtest[5]]))
# pt.show()

p = clf.predict(xtest)

#uncomment to directly get accuracy as ratio
#print(clf.score(xtest,actual_lable,sample_weight = None))

pr = ran.predict(xtest)

#uncomment to directly get accuracy as ratio
#print(ran.score(xtest,actual_lable,sample_weight = None))

#alternate way to get accuracy
count1 = 0
count2 = 0
for i in range(0,21000):
	count1 += 1 if p[i] == actual_lable[i] else 0
	count2 += 1 if pr[i] == actual_lable[i] else 0

print("DecisionTreeClassifier Accuracy = ", (count1/21000)*100)
print("RandomForestClassifier Accuracy = ", (count2/21000)*100)