import os

class AbstractFeatureExtractor:
    def __init__(self, dataset_path = None, folder_list = None):
        if dataset_path == None:
            self.dataset_path = '/home/ubuntu/dataset/stanford_mobile_images/'
        else:
            self.dataset_path = dataset_path
        if folder_list == None:
            self.folder_list = self.list_subfolders(self.dataset_path)
        else:
            self.folder_list = folder_list
        self.folder_list = self.list_folders(self.folder_list)



    def list_folders(self, folder_list):
        """
        :return: a list of folders to predict
        """
        # in the case folder_list is complete or already parsed
        if type(folder_list) == dict:
            return self.folder_list

        res = {}
        for f in folder_list:
            res[f] = self.list_subfolders(f)
        return res



    def list_subfolders(self, folder_name):
        """
        :param folder_name: folder to navigate
        :return: list of all folders in given folder_name
        """
        subfolder_list = []
        for f in os.listdir(folder_name):
            if os.path.isdir(folder_name + f) and f != 'Reference':
                subfolder_list.append(folder_name + f + '/')
        return subfolder_list


    def save_features(self):
        for f, sub in self.folder_list.iteritems():
            self.folder_feature_extractor(f + 'Reference/')
            for ff in sub:
                self.folder_feature_extractor(ff)
        return ''

    def folder_feature_extractor(self, folder_name):
        pass
