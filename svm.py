#digit plot karna hai toh matplotlib ko import karo
import matplotlib.pyplot as plt

#import datasets,classifiers and performance metrics

from sklearn import datasets,svm,metrics

#dataset ko load karo
digits=datasets.load_digits()
#print digits.images.shape
# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))
l=len(images_and_labels)
l=l-4;

# plot kar diya last ke 4 ko
for index, (image, label) in enumerate(images_and_labels[l:]):
    plt.subplot(2, 4, index+1)
    plt.axis('on')
    plt.imshow(255-image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
    plt.show()

#turn the data in (samples,feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
#print data.shape  - 1764*8*8 ----> 1764*64
#ab classifier banate hai 
#naam dete hai : a support vector classifier
classifier = svm.SVC(gamma=0.001)

#Now learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])


print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

l=len(predicted)
l=l-4

#plot kiya last predicted 4 ko 
images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[l:]):
    plt.subplot(2, 4, index + 1)
    plt.axis('on')
    plt.imshow(255-image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()

#OLLA WHAT AN CLASSIFIER IT IS 
#EXPECTED PLOT - 0 8 9 8
#PREDICTED PLOT- 0 8 9 8

#Finding accuracy
count=0;
l2=len(images_and_labels)
l2=l2//2;

for i in range(0,l2):
   count+=1 if predicted[i]==expected[i] else 0
print "Accuracy=",100*(float(count)/float(l))   

# %ACCURACY = 97.2067039106
#ADVANTAGE - Accuracy % is so awesome maza aa jaata dekh ke upto 97% :)
#DISADVANTAGE - For higher sample dataset of order 100000 it will be slow as it takes quadratic amount of time
   