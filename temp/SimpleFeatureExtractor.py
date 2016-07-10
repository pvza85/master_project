from AbstractFeatureExtractor import *
import numpy as np
import sys

class SimpleFeatureExtractor(AbstractFeatureExtractor):
    """
    Simple Feature Extractor using one layer of a pretrained Deep Learning network
    """
    version = '0'
    def __init__(self, dataset_path = None, folder_list = None, model_name = 'bvlc_reference_caffenet',
                 layer_name = 'fc8', caffe_root = '/home/ubuntu/caffe/', mode='gpu'):
        AbstractFeatureExtractor.__init__(self, dataset_path, folder_list)
        self.model_name = model_name
        self.layer_name = layer_name
        caffe_root = caffe_root
        sys.path.insert(0, caffe_root + 'python')
        global caffe
        import caffe

        model_prototxt = "{0}models/{1}/deploy.prototxt".format(caffe_root, model_name)
        model_trained = "{0}models/{1}/{1}.caffemodel".format(caffe_root, model_name)
        self.postfix = "_" + self.model_name.replace('/','') + '_' + self.layer_name.replace('/','') + '_' + self.version
        if mode == 'gpu':
            caffe.set_device(0)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.net = caffe.Classifier(model_prototxt, model_trained,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

    def save_features(self):
        for f, sub in self.folder_list.iteritems():
<<<<<<< HEAD
            print f.split('/')[-2]
=======
            print '{0}:\n'.format(f.split('/')[-2])
            print '\t{0}\n'.format('Reference')
>>>>>>> 982f5d8a37f186fe422f58e5540840969aabe541
            self.folder_feature_extractor(f + 'Reference/')
            print '\t{0}\n'.format('Reference');
            for ff in sub:
<<<<<<< HEAD
                print '\t{0}\n'.format(ff.split('/')[-2]);
=======
                print '\t{0}\n'.format(ff.split('/')[-2])
>>>>>>> 982f5d8a37f186fe422f58e5540840969aabe541
                self.folder_feature_extractor(ff)
        return self.postfix

    def folder_feature_extractor(self, folder_name):
        output_file = "{0}features{1}.txt".format(folder_name, self.postfix)
        with open(output_file, 'w') as writer:
            writer.truncate()
            for file_name in os.listdir(folder_name):
                if file_name.endswith('.jpg'):
                    file_name = folder_name + file_name
                    if os.path.isfile(file_name):
                        image = caffe.io.load_image(file_name)
                        prediction = self.net.predict([image], oversample=False)
                        res = []
                        res = np.append(res, self.net.blobs[self.layer_name].data[0])
                        np.savetxt(writer, res.reshape(1,-1), fmt='%.8g')
