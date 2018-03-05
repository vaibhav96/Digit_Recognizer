import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
# from scipy.spatial import distance
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


class KNNClassifier(object):
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def euc_distance(self, a, b):
        return np.linalg.norm(a-b)
        # return distance.euclidean(a, b)

    def closest(self, row):
        """
        Returns the label corresponding to the single closest training example.
        This is a k=1 nearest neighbor(s) implementation.
        :param row:
        :return:
        """
        dist = [self.euc_distance(row, trainer) for trainer in self.X_train]
        best_index = dist.index(min(dist))

        return self.y_train[best_index]

    def fit(self, training_data, training_labels):
        self.X_train = training_data
        self.y_train = training_labels

    def predict(self, to_classify):
        predictions = []
        for row in to_classify:
            label = self.closest(row)
            predictions.append(label)

        return predictions


data=pd.read_csv("~/Desktop/projects/ML/digit-recogniser/train.csv").as_matrix()


#classifier = KNeighborsClassifier()  # k=5 by default

clf = KNNClassifier()
#training data set

xtrain=data[0:5000,1:]
train_label=data[0:5000,0]
clf.fit(xtrain,train_label)

#testing data
xtest=data[5001:10000,1:]
actual_label=data[5001:10000,0]

d=xtest[5]
d.shape=(28,28)
pt.imshow(d,cmap='gray')

#predited xtest[5] 
print(clf.predict([xtest[5]]))

#show on graph
pt.show()

# Now predict the value of the digit on the second half:
expected = data[5001:10000,0]
predicted = clf.predict(data[5001:10000,1:])

#Confusion matrix bana ke dekho mast lagega 
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))



#Finding accuracy

p=clf.predict(xtest)
count=0;
for i in range(0,len(p)):
    count+=1 if p[i]==actual_label[i] else 0 
print "Accuracy=",100*(float(count)/float(len(p)))

#Advantage - Accuracy aachi de rha 97% tak
#Disadvantage - But bahot slow hai large data ho toh bhul ke bhi mat chalana isse
#K ki value kitni leni usse kafi farq padta. In general k ki value jaada lete hai toh the effect 
#of noise suppress toh hoga but classification boundaries bhi less distinct hogi
#So choose K wisely 
 