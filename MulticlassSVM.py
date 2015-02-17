import IAlgorithm
from sklearn.svm import LinearSVC
from numpy import array

__author__ = 'simon'


class MulticlassSVM(IAlgorithm.IAlgorithm):
    def __init__(self,penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True,
                 intercept_scaling=1, class_weight=None, verbose=0, random_state=None):
        self.svm_model = LinearSVC(penalty=penalty, loss=loss, dual=dual, tol=tol, C=C, multi_class=multi_class,
                                   fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                   class_weight=class_weight, verbose=verbose, random_state=random_state)

    def compute(self, in_array):
        return self.svm_model.predict(in_array.flatten())

    # Optional, only required if your algorithm needs training, otherwise remove this function
    def train(self, data_generator, label_generator):
        # First, collect all elements of the input
        data = [d.flatten() for d in data_generator]
        labels = array(list(label_generator))
        out = self.svm_model.fit_transform(data,labels)
        return (d for d in out)