import os
import numpy as np
from scipy.stats.stats import pearsonr

def calculate_accuracy(ref, test):
    count_pear = 0
    for i in range(0,len(test)):
        max_pear = 0.0
        max_pear_index = -1
        for j in range(0,len(ref)):
            pear = abs(pearsonr(ref[j], test[i])[0])
            if pear > max_pear:
                max_pear_index = j
                max_pear = pear
        if i == max_pear_index:
            count_pear += 1
    return count_pear

if __name__ == "__main__":
    directory = '/home/payam/dataset/'
    for f in os.listdir(directory):
        if os.path.isdir(directory + f):
            in_dir = directory + f + '/'
            ref_caffe = np.loadtxt(in_dir + 'Reference/caffenet_features.txt')
            ref_google = np.loadtxt(in_dir + 'Reference/googlenet_features.txt')
            test_caffe = {}
            test_google = {}
            for ff in os.listdir(in_dir):
                if os.path.isdir(in_dir + ff) and ff != 'Reference':
                    test_caffe = np.loadtxt(in_dir + ff + '/caffenet_features.txt')
                    test_google = np.loadtxt(in_dir + ff + '/googlenet_features.txt')