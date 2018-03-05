import matplotlib.pyplot as pt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import datasets
from sklearn import svm 
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets,svm,metrics

data=pd.read_csv("~/Desktop/projects/ML/digit-recogniser/train.csv").as_matrix()
#digits=datasets.load_digits()
#print digits.data

clf=DecisionTreeClassifier()



#training data set

xtrain=data[0:21000,1:]
train_label=data[0:21000,0]
clf.fit(xtrain,train_label)

#testing data
xtest=data[21000:,1:]
actual_label=data[21000:,0]

d=xtest[8]
d.shape=(28,28)
pt.imshow(d,cmap='gray')

#predit 
print(clf.predict([xtest[8]]))

#show on graph
pt.show()

#Finding accuracy
#print len(xtest[0]) - 28*28=784
p=clf.predict(xtest)
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(actual_label, p)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(actual_label, p))

count=0;
for i in range(0,21000):
	count+=1 if p[i]==actual_label[i] else 0
print "Accuracy=",100*(float(count)/float(21000))


#Accuracy jaada aachi nhi hai 84% only, but thik hai sabse kaam time laga tujko implement karne mein