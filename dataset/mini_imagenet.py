from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import numpy as np
from PIL import Image as pil_image
import pickle
import torch
import random

# from : https://stackoverflow.com/questions/12886768/how-to-unzip-file-in-python-on-all-oses
import zipfile,os.path
def unzip(source_filename, dest_dir):
    with zipfile.ZipFile(source_filename) as zf:
        for member in zf.infolist():
            # Path traversal defense copied from
            # http://hg.python.org/cpython/file/tip/Lib/http/server.py#l789
            words = member.filename.split('/')
            path = dest_dir
            for word in words[:-1]:
                while True:
                    drive, word = os.path.splitdrive(word)
                    head, word = os.path.split(word)
                    if not drive:
                        break
                if word in (os.curdir, os.pardir, ''):
                    continue
                #path = os.path.join(path, word)
            zf.extract(member, path)

class MiniImagenetBase(data.Dataset):

    def __init__(self, root = './mini_imagenet', train='train',
                    datasetCompactSize = None, # typically 84
                    size = None, # normally size = 84
                    transform=None, target_transform=None):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training, validation or test set
        self.size = size
        self.datasetCompactSize = datasetCompactSize

        if not self._check_exists_():
            self._init_folders_()
            if self.check_decompress():
                self._decompress_()
        if self._check_preprocess():
            self._preprocess_()

    def _init_folders_(self):
        decompress = False
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        if not os.path.exists(os.path.join(self.root, 'images')):
            os.makedirs(os.path.join(self.root, 'images'))
            decompress = True
        if not os.path.exists(os.path.join(self.root, 'compacted')):
            os.makedirs(os.path.join(self.root, 'compacted'))
            decompress = True
        return decompress

    def check_decompress(self):
        return os.listdir('%s/images' % self.root) == []

    def _decompress_(self):
        print("\nDecompressing Images...")
        compressed_file = '%s/images.zip' % self.root
        if os.path.isfile(compressed_file):
            if os.name == 'nt': # Windows
                unzip(compressed_file, self.root)
            else:
                os.system('unzip %s -d %s/' % (compressed_file, self.root))
        else:
            raise Exception('Missing %s' % compressed_file)
        print("Decompressed")

    def _check_exists_(self):
        if not os.path.exists(os.path.join(self.root, 'images')) or self.check_decompress():
            return False
        else:
            return True

    def _check_preprocess(self):
        str_size = 'None' if self.datasetCompactSize is None else str(self.datasetCompactSize)
        return os.listdir('%s/compacted' % self.root) == [] or \
                not os.path.exists(os.path.join(self.root, 'compacted', 'mini_imagenet_size_%s_train.pickle' % (str_size)))

    def get_image_paths(self, file):
        images_path, class_names = [], []
        with open(file, 'r') as f:
            f.readline()
            for line in f:
                name, class_ = line.split(',')
                class_ = class_[0:(len(class_)-1)]
                path = self.root + '/images/'+name
                images_path.append(path)
                class_names.append(class_)
        return class_names, images_path

    def load_img(self, path, size = None):
        img = pil_image.open(path)
        img = img.convert('RGB')
        if not(size is None):
            img = img.resize((size,size), pil_image.ANTIALIAS) # 2-tuple resize: (width, height)
        img = np.array(img, dtype='float32')
        return img

    def _preprocess_(self):

        print('\nPreprocessing Mini-Imagenet images...')
        (class_names_train, images_path_train) = self.get_image_paths('%s/train.csv' % self.root)
        (class_names_test, images_path_test) = self.get_image_paths('%s/test.csv' % self.root)
        (class_names_val, images_path_val) = self.get_image_paths('%s/val.csv' % self.root)

        keys_train = list(set(class_names_train))
        keys_test = list(set(class_names_test))
        keys_val = list(set(class_names_val))
        label_encoder = {}
        label_decoder = {}
        for i in range(len(keys_train)):
            label_encoder[keys_train[i]] = i
            label_decoder[i] = keys_train[i]
        for i in range(len(keys_train), len(keys_train)+len(keys_test)):
            label_encoder[keys_test[i-len(keys_train)]] = i
            label_decoder[i] = keys_test[i-len(keys_train)]
        for i in range(len(keys_train)+len(keys_test), len(keys_train)+len(keys_test)+len(keys_val)):
            label_encoder[keys_val[i-len(keys_train) - len(keys_test)]] = i
            label_decoder[i] = keys_val[i-len(keys_train)-len(keys_test)]

        counter = 0
        train_set = {}
        for class_, path in zip(class_names_train, images_path_train):
            data_to_store = None
            if self.datasetCompactSize == None:
                data_to_store = path
            else:
                data_to_store = self.load_img(path, self.datasetCompactSize)
            if label_encoder[class_] not in train_set:
                train_set[label_encoder[class_]] = []
            train_set[label_encoder[class_]].append(data_to_store)
            counter += 1
            if counter % 1000 == 0:
                print("Counter "+str(counter) + " from " + str(len(images_path_train) + len(class_names_test) +
                                                               len(class_names_val)))

        test_set = {}
        for class_, path in zip(class_names_test, images_path_test):
            data_to_store = None
            if self.datasetCompactSize == None:
                data_to_store = path
            else:
                data_to_store = self.load_img(path, self.datasetCompactSize)
            if label_encoder[class_] not in test_set:
                test_set[label_encoder[class_]] = []
            test_set[label_encoder[class_]].append(data_to_store)
            counter += 1
            if counter % 1000 == 0:
                print("Counter " + str(counter) + " from "+str(len(images_path_train) + len(class_names_test) +
                                                               len(class_names_val)))

        val_set = {}
        for class_, path in zip(class_names_val, images_path_val):
            data_to_store = None
            if self.datasetCompactSize == None:
                data_to_store = path
            else:
                data_to_store = self.load_img(path, self.datasetCompactSize)
            if label_encoder[class_] not in val_set:
                val_set[label_encoder[class_]] = []
            val_set[label_encoder[class_]].append(data_to_store)
            counter += 1
            if counter % 1000 == 0:
                print("Counter "+str(counter) + " from " + str(len(images_path_train) + len(class_names_test) +
                                                               len(class_names_val)))

        str_size = 'None' if self.datasetCompactSize is None else str(self.datasetCompactSize)
        with open(os.path.join(self.root, 'compacted', 'mini_imagenet_size_%s_train.pickle' % (str_size)), 'wb') as handle:
            pickle.dump(train_set, handle, protocol=2)
        with open(os.path.join(self.root, 'compacted', 'mini_imagenet_size_%s_test.pickle' % (str_size)), 'wb') as handle:
            pickle.dump(test_set, handle, protocol=2)
        with open(os.path.join(self.root, 'compacted', 'mini_imagenet_size_%s_val.pickle' % (str_size)), 'wb') as handle:
            pickle.dump(val_set, handle, protocol=2)

        label_encoder = {}
        keys = list(train_set.keys()) + list(test_set.keys())
        for id_key, key in enumerate(keys):
            label_encoder[key] = id_key
        with open(os.path.join(self.root, 'compacted', 'mini_imagenet_label_encoder.pickle'), 'wb') as handle:
            pickle.dump(label_encoder, handle, protocol=2)

        label_decoder = {}
        keys = list(train_set.keys()) + list(test_set.keys())
        for id_key, key in enumerate(keys):
            label_decoder[key] = id_key
        with open(os.path.join(self.root, 'compacted', 'mini_imagenet_label_decoder.pickle'), 'wb') as handle:
            pickle.dump(label_decoder, handle, protocol=2)

        print('Images preprocessed')

    def load_dataset(self, partition):
        
        print("Loading dataset")

        str_size = 'None' if self.datasetCompactSize is None else str(self.datasetCompactSize)
        with open(os.path.join(self.root, 'compacted', 'mini_imagenet_size_%s_%s.pickle' % (str_size, partition)),
                    'rb') as handle:
            data = pickle.load(handle)

        with open(os.path.join(self.root, 'compacted', 'mini_imagenet_label_encoder.pickle'),
                  'rb') as handle:
            label_encoder = pickle.load(handle)
        
        with open(os.path.join(self.root, 'compacted', 'mini_imagenet_label_decoder.pickle'),
                  'rb') as handle:
            label_decoder = pickle.load(handle)

        # If we have an option size then resize if needed.
        if not (self.size is None) and not (self.datasetCompactSize is None):
            # Resize images and normalize
            for class_ in data:
                for i in range(len(data[class_])):
                    image2resize = pil_image.fromarray(np.uint8(data[class_][i]))
                    if not (self.size is None) and not self.size == image2resize.width and not self.size == image2resize.height:
                        image_resized = image2resize.resize((self.size, self.size))
                    else:
                        image_resized = image2resize
                    data[class_][i] = image_resized

        classes = data.keys()
        print("Num classes " + str(len(data)))
        num_images = 0
        for class_ in data:
            num_images += len(data[class_])
        print("Num images " + str(num_images))

        return data, label_encoder, label_decoder


class MiniImagenet(MiniImagenetBase):

    def __init__(self, root = './mini_imagenet', train='train',
                 datasetCompactSize = None,
                 size=None, # normally size = 84
                 transform=None, target_transform=None):

        MiniImagenetBase.__init__(self, root = root, train = train, datasetCompactSize = datasetCompactSize, 
                                    size = size, transform = transform, target_transform = target_transform)
        self.data, self.label_encoder, self.label_decoder = self.load_dataset(self.train)
        self.idx_to_class = np.array([np.repeat(class_,len(self.data[class_])) for class_ in self.data]).flatten()
        self.idx_to_img = np.array([range(len(self.data[class_])) for class_ in self.data]).flatten()

    def __getitem__(self, index):

        class_choice = self.idx_to_class[index]
        sample_choice = self.idx_to_img[index]

        img1 = self.data[class_choice][sample_choice]
        target = class_choice

        # If we have paths in self.data then load the image
        if type(img1).__name__ == 'str':
            img1 = self.load_img(path=img1,size=self.size)
            img1 = pil_image.fromarray(np.uint8(img1))

        if self.transform is not None:
            img1 = self.transform(img1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        '''
        import cv2
        cv2.imwrite('/home/aberenguel/tmp/arc_img/index_' + str(index) + 'img1_target_' + str(target) +  '.png',
                            img1.transpose(0, 1).transpose(1, 2).cpu().numpy() * 255)
        cv2.imwrite('/home/aberenguel/tmp/arc_img/index_' + str(index) + 'img2_target_' + str(target) +  '.png',
                            img2.transpose(0, 1).transpose(1, 2).cpu().numpy() * 255)
        '''

        return img1, target

    def __len__(self):
        num_images = 0
        for class_ in self.data:
            num_images += len(self.data[class_])
        return num_images
        


class MiniImagenetPairs(MiniImagenetBase):

    def __init__(self, root = './mini_imagenet', train='train',
                 datasetCompactSize = None,
                 size=None, # normally size = 84
                 transform=None, target_transform=None,
                 numTrials = 512):

        MiniImagenetBase.__init__(self, root = root, train = train, datasetCompactSize = datasetCompactSize, 
                                    size = size, transform = transform, target_transform = target_transform)
        self.data, self.label_encoder, self.label_decoder = self.load_dataset(self.train)
        self.numTrials = numTrials

    def __getitem__(self, index):

        # set the choice function to random
        np.random.seed(None)
        num_classes = len(self.data)
        
        similar_classes = True
        if index > self.numTrials/2: #different characters
            similar_classes = False

        # pick two classes
        two_class_choice = np.random.choice(list(self.data), 2, replace=False)

        idx_class = np.random.choice(range(len(self.data[two_class_choice[0]])))
        img1 = self.data[two_class_choice[0]][idx_class]

        if similar_classes:
            idx_class = np.random.choice(range(len(self.data[two_class_choice[0]])))
            img2 = self.data[two_class_choice[0]][idx_class]
        else:
            idx_class = np.random.choice(range(len(self.data[two_class_choice[1]])))
            img2 = self.data[two_class_choice[1]][idx_class]

        target = int(similar_classes)

        # If we have paths in self.data then load the image
        if type(img1).__name__ == 'str':
            img1 = self.load_img(path=img1,size=self.size)
            img1 = pil_image.fromarray(np.uint8(img1))
        
        # If we have paths in self.data then load the image
        if type(img2).__name__ == 'str':
            img2 = self.load_img(path=img2,size=self.size)
            img2 = pil_image.fromarray(np.uint8(img2))        

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.target_transform is not None:
            target = self.target_transform(target)

        '''
        import cv2
        cv2.imwrite('/home/aberenguel/tmp/arc_img/index_' + str(index) + 'img1_target_' + str(target) +  '.png',
                            img1.transpose(0, 1).transpose(1, 2).cpu().numpy() * 255)
        cv2.imwrite('/home/aberenguel/tmp/arc_img/index_' + str(index) + 'img2_target_' + str(target) +  '.png',
                            img2.transpose(0, 1).transpose(1, 2).cpu().numpy() * 255)
        '''

        return torch.stack((img1,img2)), target

    def __len__(self):
        return self.numTrials


class MiniImagenetOneShot(MiniImagenetBase):

    def __init__(self, root = './mini_imagenet', train='train',
                 datasetCompactSize = None,
                 size=None, # normally size = 84
                 transform=None, target_transform=None,
                 n_way = 20,
                 n_shot = 1,
                 numTrials = 32,
                 ):

        MiniImagenetBase.__init__(self, root = root, train = train, datasetCompactSize = datasetCompactSize, 
                                    size = size, transform = transform, target_transform = target_transform)
        self.data, self.label_encoder, self.label_decoder = self.load_dataset(self.train)
        self.n_way = n_way
        self.n_shot = n_shot
        self.numTrials = numTrials

    def __getitem__(self, index):

        # set the choice function to random
        np.random.seed(None)

        # Init variables
        channels = 3
        labels = np.zeros((self.n_way*self.n_shot + 1), dtype='float32')
        
        batches_xi = []
        for i in range(self.n_way*self.n_shot + 1):
            batches_xi.append(np.zeros((channels, self.size, self.size), dtype='float32'))

        # Select the batch
        positive_class_counter = random.randint(0, self.n_way - 1)
        # Sample random classes for this TASK
        classes_ = list(self.data.keys())
        sampled_classes = random.sample(classes_, self.n_way)
        indexes_perm = np.random.permutation(self.n_way * self.n_shot)

        counter = 0
        for class_counter, class_ in enumerate(sampled_classes):
            if class_counter == positive_class_counter:
                # We take n_shot + one sample for one class
                samples = random.sample(self.data[class_], self.n_shot+1)
                # Test sample is loaded
                batches_xi[-1] = samples[0]
                samples = samples[1::]
                # Set in the last position the information of the positive label
                labels[-1] = class_ 
            else:
                samples = random.sample(self.data[class_], self.n_shot)

            for s_i in range(0, len(samples)):
                batches_xi[indexes_perm[counter]] = samples[s_i]
                labels[indexes_perm[counter]] = class_
                counter += 1

        batches_xi = torch.stack([self.transform(batch_xi) for batch_xi in batches_xi])
        labels = torch.from_numpy(labels)

        '''
        import cv2
        [cv2.imwrite('/home/aberenguel/tmp/cedar/im_' + str(i) + '.png',
                     trial[i].transpose(0, 1).transpose(1, 2).cpu().numpy() * 255) for i in range(trial.shape[0])]
        '''

        return batches_xi, labels

    def __len__(self):
        # num characters * num samples of each character
        return self.numTrials
