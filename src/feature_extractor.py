import numpy as np
import getopt
import os, sys

# Main path to your caffe installation
caffe_root = '/home/payam/workspace/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

# Model prototxt file
caffe_model_prototxt = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
google_model_prototxt = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'

# Model caffemodel file
caffe_model_trained = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
google_model_trained = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'

# Name of the layer we want to extract
google_layer_name = 'pool5/7x7_s1'
caffe_layer_name = 'fc8'

dateset_path = '/home/payam/dataset/'


def folder_feature_extractor(input_folder, model_prototxt, model_trained, layer_name, output_file):
    caffe.set_mode_cpu()
    net = caffe.Classifier(model_prototxt, model_trained,
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))

    output_file = input_folder + output_file + '_features.txt'
    with open(output_file, 'w') as writer:
        writer.truncate()
        for i in range(1,101):
            file_name =  str(i).zfill(3) + '.jpg'
            file_name = input_folder + file_name
            if os.path.isfile(file_name):
                image = caffe.io.load_image(file_name)
                prediction = net.predict([image], oversample=False)
                res = []
                for i in range(10):
                    res = np.append(res, net.blobs[layer_name].data[i])
                np.savetxt(writer, res.reshape(1,-1), fmt='%.8g')



if __name__ == "__main__":
    for f in os.listdir(dateset_path):
        if os.path.isdir(dateset_path + f):
            in_dir = dateset_path + f + '/'
            for ff in os.listdir(in_dir):
                if os.path.isdir(in_dir + ff):
                    print in_dir + ff + ':'
                    model = 'caffenet'
                    print '\t' + model
                    folder_feature_extractor(in_dir + ff + '/', caffe_model_prototxt, caffe_model_trained,
                                             caffe_layer_name,model)
                    model = 'googlenet'
                    print '\t' + model
                    folder_feature_extractor(in_dir + ff + '/', google_model_prototxt, google_model_trained,
                                             google_layer_name,model)