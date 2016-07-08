from AbstractPredictor import *
import numpy as np

class EuclideanDistancePredictor(AbstractPredictor):
    """
    Author: Payam Azad 07-JUL-2016
    It is a class of simplest predictor.
    It reads extracted features from each folder and by using closest Euclidean Distance
    tries to guess the class.
    """
    def __init__(self, dataset_path=None, folder_list=None, feature_file_postfix=None):
        """
        :param feature_file_postfix: This the postfix for feature file e.g: features_caffenet.txt ('_caffenet')
        """
        AbstractPredictor.__init__(self, dataset_path, folder_list)
        self.feature_file_name = 'features' + str(feature_file_postfix) + '.txt'

    def predict(self):
        accuracy = {}
        for f, sub in self.folder_list.iteritems():
            ref = np.loadtxt(f + 'Reference/' + self.feature_file_name)
            res = {}
            for ff in sub:
                    test = np.loadtxt( ff + self.feature_file_name)
                    res[ff.split('/')[-2]] = self.calculate_accuracy(ref, test)
            accuracy[f.split('/')[-2]] = res
        return accuracy

    def calculate_accuracy(self, ref, test):
        pred = self.prediction(ref, test)
        hit_count = 0.0
        for i in range(len(test)):
            if i == pred[i]:
                hit_count += 1
        return hit_count / len(test) * 100

    def prediction(self, ref, test):
        return self.predict_using_euclidean_distance( ref, test)
    def predict_using_euclidean_distance(self, ref, test):
        pred = []
        for i in range(len(test)):
            min_dist = 999999.0
            min_index = -1
            for j in range(len(ref)):
                dist = np.linalg.norm(ref[i] - test[j])
                if dist < min_dist:
                    min_index = j
                    min_dist = dist
            pred.append(min_index)
        return pred
