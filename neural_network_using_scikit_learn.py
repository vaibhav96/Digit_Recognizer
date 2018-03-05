import matplotlib.pyplot as pt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import datasets
from sklearn import svm 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import datasets,metrics

#data read kiya
data=pd.read_csv("~/Desktop/projects/ML/digit-recogniser/train.csv").as_matrix()
#print data.shape

#multiperceptron classifier ka object banaya
clf = MLPClassifier(solver='adam',hidden_layer_sizes=(700,500), alpha=1e-5, random_state=1)

#training data set

xtrain=data[0:5000,1:]
train_label=data[0:5000,0]
clf.fit(xtrain,train_label)

#testing data
xtest=data[5000:10001,1:]
#print len(xtest)
actual_label=data[5000:10001,0]

#xtest[5] jo actual hai vo plot karke dekhte hai phele
d=xtest[5]
d.shape=(28,28)
pt.imshow(d,cmap='gray')

#predited xtest[5] 
print(clf.predict([xtest[5]]))

# Now predict the value of the digit on the second half:
expected = data[5000:10001,0]
predicted = clf.predict(data[5000:10001,1:])

#Confusion matrix bana ke dekho mast lagega 
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

#show on graph
pt.show()


#Finding accuracy

p=clf.predict(xtest)
count=0;
for i in range(0,len(p)):
	count+=1 if p[i]==actual_label[i] else 0 
print "Accuracy=",100*(float(count)/float(len(p)))


#Good classifier Acuracy obtained 94.9761904762

#Advantage -Capability to learn non-linear models.

#Disadvantages-1) MLP requires tuning a number of hyperparameters such as the number of hidden neurons, layers, and iterations.

#Meine experience kiya hidden layer nodes ka count bahot important role play karta 
#phele 5 set karke dekha accuracy kewal 11% aa rhi thi fir jab 700 kiya toh 95% tak aayi per time bahot lag rha tha chale mein takes makes me little sad :)