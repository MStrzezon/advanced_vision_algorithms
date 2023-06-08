import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from joblib import dump
from lab12.ex1 import hog

HOG_data = np.zeros([2 * 100, 3781], np.float32)
for i in range(0, 100):
    IP = cv2.imread('images/train/pos/per%05d.ppm' % (i + 1))
    IN = cv2.imread('images/train/neg/neg%05d.png' % (i + 1))
    F = hog(IP)
    HOG_data[i, 0] = 1
    HOG_data[i, 1:] = F
    F = hog(IN)
    HOG_data[i + 100, 0] = 0
    HOG_data[i + 100, 1:] = F

labels = HOG_data[:, 0]
data = HOG_data[:, 1:]

clf = svm.SVC(kernel='linear', C=1.0)

X_train, X_val_test, y_train, y_val_test = train_test_split(data, labels, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

clf.fit(X_train, y_train)

# Validation
y_val_pred = clf.predict(X_val)
cm = confusion_matrix(y_val, y_val_pred)
TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]
acc_val = (TP + TN) / (TP + TN + FP + FN)
print('acc val:', acc_val)

# TEST
y_test_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)
TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]
acc_test = (TP + TN) / (TP + TN + FP + FN)
print('acc test:', acc_test)

dump(clf, 'svm.joblib')
