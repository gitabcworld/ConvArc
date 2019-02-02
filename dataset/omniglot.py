from __future__ import print_function

import os
import os.path
import sys
import torch

import numpy as np
import torch.utils.data as data
from numpy.random import choice
from scipy.misc import imresize as resize
from PIL import Image

# if sys.version_info[0] == 2:
#     import cPickle as pickle
# else:
#     import pickle

class OmniglotBase(data.Dataset):

    def __init__(self, root = '../../data', train='train', rnd_seed = 42,
                 transform=None, target_transform=None):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training, validation or test set
        self.rnd_seed = rnd_seed

        self.chars = np.load(os.path.join(self.root,'omniglot.npy')).astype(np.uint8)

        # SPEED UP: CASE image_size == 32. DELETE!!!
        #from scipy.misc import imresize as resize
        #resized_chars = np.zeros((1623, 20, 32, 32), dtype='uint8')
        #for i in range(1623):
        #    for j in range(20):
        #        resized_chars[i, j] = resize(self.chars[i, j], (32, 32))
        #self.chars = resized_chars

        self.channels = 1
        self.nDiffCharacters, self.nSameCharacters, self.image_size, self.image_size = self.chars.shape

        # size of each alphabet (num of chars)
        self.a_size = np.array([20, 29, 26, 41, 40, 24, 46, 14, 26, 34, 33, 22, 26, 43, 24, 48, 22,
                  16, 52, 47, 40, 26, 40, 41, 33, 14, 42, 23, 17, 55, 20, 26, 26, 26,
                  26, 26, 45, 45, 41, 26, 47, 40, 30, 45, 46, 28, 23, 25, 42, 26])

        # each alphabet/language has different number of characters.
        # in order to uniformly sample all characters, we need weigh the probability
        # of sampling a alphabet by its size. p is that probability
        self.p = self.a_size.astype(np.float)/self.a_size.astype(np.float).sum()

        # Create a label list by alphabets
        self.labels  = np.concatenate([np.repeat(i, n_times) for i,n_times in enumerate(self.a_size)])

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

class Omniglot(OmniglotBase):

    def __init__(self, root = '../../data', train='train', rnd_seed = 42,
                 reduced_dataset = False,
                 transform=None, target_transform=None, 
                 partitionType = '30_10_10'):
        OmniglotBase.__init__(self, root, train, rnd_seed, 
                                 transform, target_transform)
        self.mode = train
        self.reduced_dataset = reduced_dataset
        self.partitionType = partitionType
        
        # Do the partition 30 - 10 - 10
        np.random.seed(self.rnd_seed) # Use the same seed for train - val - test
        
        if self.partitionType == '30_10_10':
            '''
            From the 50 Omniglot dataset alphabets, 30 are used for training, 10 for validation and
            the 10 for testing
            '''
            # Mix all the alphabets
            indexes_perm = np.random.permutation(len(self.a_size))
            alphabets_train = indexes_perm[0:30]
            alphabets_val = indexes_perm[30:40]
            alphabets_test = indexes_perm[40:]
            # reorder all the information by the alphabet mixing.
            selected_alphabets = []
            selected_alphabets = alphabets_train if self.train == 'train' else selected_alphabets
            selected_alphabets = alphabets_val if self.train == 'val' else selected_alphabets 
            selected_alphabets = alphabets_test if self.train == 'test' else selected_alphabets 
            self.chars = self.chars[[elem in selected_alphabets for elem in self.labels]]
            self.a_size = self.a_size[selected_alphabets]
            self.p = self.p[selected_alphabets]
            self.labels = self.labels[[elem in selected_alphabets for elem in self.labels]]
            # Make this partition probabilities to sum 1
            self.p = self.p / self.p.sum()
        
        if self.partitionType == 'Lake':
            '''
            Mix all the characters across alphabets
            '''
            indexes_perm = np.random.permutation(len(self.labels))
            self.chars = self.chars[indexes_perm]
            self.labels = self.labels[indexes_perm]
            self.chars = self.chars[:1200] if self.train == 'train' or self.train == 'val' else self.chars
            self.chars = self.chars[1200:] if self.train == 'test' else self.chars 
            self.labels = self.labels[:1200] if self.train == 'train' or self.train == 'val' else self.labels
            self.labels = self.labels[1200:] if self.train == 'test' else self.labels 
            self.a_size = None
            self.p = None
            # Train (964, 16, H, W) / Eval (964, 4, H, W) / Test (659, 20, H, W)
            self.chars = self.chars[:,:16,:,:] if self.train == 'train' else self.chars 
            self.chars = self.chars[:,16:,:,:] if self.train == 'val' else self.chars

        if self.partitionType == 'OmniglotStandard':

            # Mix all the alphabets
            indexes_perm = np.random.permutation(len(self.a_size))
            alphabets_train = indexes_perm[0:30]
            alphabets_test = indexes_perm[40:]
            # reorder all the information by the alphabet mixing.
            selected_alphabets = []
            selected_alphabets = alphabets_train if self.train == 'train' or self.train == 'val' else selected_alphabets
            selected_alphabets = alphabets_test if self.train == 'test' else selected_alphabets 
            self.chars = self.chars[[elem in selected_alphabets for elem in self.labels]]
            self.a_size = self.a_size[selected_alphabets]
            self.p = self.p[selected_alphabets]
            self.labels = self.labels[[elem in selected_alphabets for elem in self.labels]]
            # Train (964, 16, H, W) / Eval (964, 4, H, W) / Test (659, 20, H, W)
            self.chars = self.chars[:,:16,:,:] if self.train == 'train' else self.chars 
            self.chars = self.chars[:,16:,:,:] if self.train == 'val' else self.chars
            # Make this partition probabilities to sum 1
            self.p = self.p / self.p.sum()

        # set the choice function to random again
        np.random.seed(None)

    def getNumClasses(self):
        return len(list(set(self.labels)))

    def __getitem__(self, index):

        index_inter_character = int(index / self.chars.shape[1])
        index_intra_character = index - index_inter_character*self.chars.shape[1]

        img = self.chars[index_inter_character][index_intra_character]
        img = Image.fromarray(img)

        target = self.labels[index_inter_character]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.chars.shape[0]*self.chars.shape[1]

class OmniglotPairs(Omniglot):

    def __init__(self, root = '../../data', train='train', rnd_seed = 42,
                 reduced_dataset=False,
                 transform=None, target_transform=None,
                 partitionType = '30_10_10',
                 numTrials = 512):

        Omniglot.__init__(self, root, train, rnd_seed,
                                reduced_dataset, transform, target_transform,
                                partitionType)
        self.numTrials = numTrials

    def __getitem__(self, index):

        similar_characters = True
        if index > self.numTrials/2: #different characters
            similar_characters = False

        # set the choice function to random 
        np.random.seed(None)
        # characters from different alphabets
        alphbt_idx = choice(list(set(self.labels)), size=1, p=self.p)
        
        if similar_characters:
            tmp_data_alphabet = self.chars[self.labels==alphbt_idx]
            tmp_data = tmp_data_alphabet[choice(range(len(tmp_data_alphabet)))]
            tmp_selection = choice(range(len(tmp_data)), 2, replace=True)
            img1 = tmp_data[tmp_selection[0]]
            img2 = tmp_data[tmp_selection[1]]
        else:
            tmp_data_alphabet = self.chars[self.labels==alphbt_idx]
            tmp_data = tmp_data_alphabet[choice(range(len(tmp_data_alphabet)))]
            tmp_selection = choice(range(len(tmp_data)))
            img1 = tmp_data[tmp_selection]
            tmp_data_alphabet = self.chars[np.bitwise_not(self.labels==alphbt_idx)]
            tmp_data = tmp_data_alphabet[choice(range(len(tmp_data_alphabet)))]
            tmp_selection = choice(range(len(tmp_data)))
            img2 = tmp_data[tmp_selection]

        img1 = np.squeeze(img1)
        img2 = np.squeeze(img2)

        target = int(similar_characters)

        # Convert the image to PIL
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            # Case the FCN is done inside the DataLoader
            if len(img1.shape)>3:
                img1 = img1[0]
                img2 = img2[0]

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


class OmniglotOneShot(Omniglot):

    def __init__(self, root = '../../data', train='train', rnd_seed = 42,
                 reduced_dataset = False,
                 transform=None, target_transform=None,
                 partitionType = '30_10_10',
                 n_way = 20,
                 n_shot = 1,
                 numTrials = 512):

        Omniglot.__init__(self, root, train, rnd_seed, 
                                reduced_dataset, transform, target_transform,
                                partitionType
                                )
        self.n_way = n_way
        self.n_shot = n_shot
        self.numTrials = numTrials

    def __getitem__(self, index):

        # set the choice function to random 
        np.random.seed(None)
        # select positive character
        positive_label = choice(list(set(self.labels)), size=1, p=self.p)
        
        # select the position of the positive char and the negative ones.
        char_choices = list(range(self.n_way*self.n_shot))  # set of all possible chars
        key_char_idx = choice(char_choices, self.n_shot, replace=False)  # this will be the char to be matched
        # sample the other positions other chars excluding key
        for i in sorted(key_char_idx,reverse=True):
            char_choices.pop(i)
        other_char_idxs = choice(char_choices, (self.n_way-1)*self.n_shot, replace=False)

        trial = np.zeros((self.n_way*self.n_shot+1, self.chars.shape[2], self.chars.shape[3]), dtype='uint8')
        labels = np.zeros((self.n_way*self.n_shot+1), dtype='uint8')

        tmp_data_alphabet = self.chars[self.labels==positive_label]
        tmp_data = tmp_data_alphabet[choice(range(len(tmp_data_alphabet)))]
        tmp_selection = choice(range(len(tmp_data)),self.n_shot+1, replace=False)
        imgs = tmp_data[tmp_selection]
        # Set the test example
        trial[-1] = imgs[0]
        labels[-1] = positive_label
        # Set the positive example in the meta-training.
        trial[key_char_idx] = imgs[1:]
        labels[key_char_idx] = positive_label

        all_indexes_false = np.array(range(len(self.labels)))[np.bitwise_not(self.labels==positive_label)]
        selected_indexes_false = choice(all_indexes_false, (self.n_way-1)*self.n_shot,replace=False)
        
        tmp_data = self.chars[selected_indexes_false]
        imgs = np.array([tmp_data[i,choice(range(tmp_data.shape[1])),:,:] for i in range(tmp_data.shape[0])])
        # Set the negative examples in the meta-training.
        trial[other_char_idxs] = imgs
        labels[other_char_idxs] = self.labels[selected_indexes_false]

        imgs = []
        if self.transform is not None:
            for i in range(trial.shape[0]):
                img_transformed = self.transform(Image.fromarray(trial[i]))
                 # Case the FCN is done inside the DataLoader
                if len(img_transformed.shape)>3:
                    img_transformed = img_transformed[0]
                imgs.append(img_transformed)
        trial = torch.stack(imgs)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        '''
        import cv2
        [cv2.imwrite('/home/aberenguel/tmp/cedar/im_' + str(i) + '.png',
                     trial[i].transpose(0, 1).transpose(1, 2).cpu().numpy() * 255) for i in range(trial.shape[0])]
        '''

        return trial, labels

    def __len__(self):
        # num characters * num samples of each character
        return self.numTrials


