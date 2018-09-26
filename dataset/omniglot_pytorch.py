from __future__ import print_function

import os
import os.path
import sys
import torch

import numpy as np
import torch.utils.data as data
from numpy.random import choice
from scipy.misc import imresize as resize

# if sys.version_info[0] == 2:
#     import cPickle as pickle
# else:
#     import pickle


class OmniglotBase(data.Dataset):

    def __init__(self, root = os.path.join('../../data', 'omniglot.npy'), train='train',
                 transform=None, target_transform=None):

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training, validation or test set

        try:
            self.chars = np.load(self.root)
        except:
            print('Not Found Omniglot datset in : %s' % self.root)
            raise

        # SPEED UP: CASE image_size == 32. DELETE!!!
        from scipy.misc import imresize as resize
        resized_chars = np.zeros((1623, 20, 32, 32), dtype='uint8')
        for i in range(1623):
            for j in range(20):
                resized_chars[i, j] = resize(self.chars[i, j], (32, 32))
        self.chars = resized_chars

        self.channels = 1
        self.nDiffCharacters, self.nSameCharacters, self.image_size, self.image_size = self.chars.shape

        self.mean_pixel = self.chars.mean()  # used later for mean subtraction

        # starting index of each alphabet in a list of chars
        self.a_start = [0, 20, 49, 75, 116, 156, 180, 226, 240, 266, 300, 333, 355, 381,
                   424, 448, 496, 518, 534, 586, 633, 673, 699, 739, 780, 813,
                   827, 869, 892, 909, 964, 984, 1010, 1036, 1062, 1088, 1114,
                   1159, 1204, 1245, 1271, 1318, 1358, 1388, 1433, 1479, 1507,
                   1530, 1555, 1597]

        # size of each alphabet (num of chars)
        self.a_size = [20, 29, 26, 41, 40, 24, 46, 14, 26, 34, 33, 22, 26, 43, 24, 48, 22,
                  16, 52, 47, 40, 26, 40, 41, 33, 14, 42, 23, 17, 55, 20, 26, 26, 26,
                  26, 26, 45, 45, 41, 26, 47, 40, 30, 45, 46, 28, 23, 25, 42, 26]

        # each alphabet/language has different number of characters.
        # in order to uniformly sample all characters, we need weigh the probability
        # of sampling a alphabet by its size. p is that probability
        def size2p(size):
            s = np.array(size).astype('float64')
            return s / s.sum()

        self.size2p = size2p


    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

'''
From the 50 Omniglot dataset alphabets, 30 are used for training, 10 for validation and
the last 10 for testing
'''
class Omniglot_30_10_10(OmniglotBase):

    def __init__(self, root = os.path.join('../../data', 'omniglot.npy'), train='train',
                 transform=None, target_transform=None):
        OmniglotBase.__init__(self, root, train,
                                 transform, target_transform)
        self.mode = train
        self.agumentation_seed = None

        # slicing indices for splitting a_start & a_size
        i = 30
        j = 40

        num_train_chars = self.a_start[i - 1] + self.a_size[i - 1]
        num_val_chars = self.a_start[j - 1] + self.a_size[j - 1] - num_train_chars
        num_test_chars = self.a_start[-1] + self.a_size[-1] - num_train_chars - num_val_chars
        train = self.chars[:num_train_chars]  # (964, 16, H, W)
        val = self.chars[num_train_chars:num_train_chars+num_val_chars]  # (964, 4, H, W)
        test = self.chars[num_train_chars+num_val_chars:]  # (659, 20, H, W)

        self.starts = {}
        self.starts['train'], self.starts['val'], self.starts['test'] = \
                            self.a_start[:i], self.a_start[i:j], self.a_start[j:]
        self.sizes = {}
        self.sizes['train'], self.sizes['val'], self.sizes['test'] = \
                            self.a_size[:i], self.a_size[i:j], self.a_size[j:]

        self.p = {}
        self.p['train'], self.p['val'], self.p['test'] = \
            self.size2p(self.sizes['train']), self.size2p(self.sizes['val']), self.size2p(self.sizes['test'])

        self.data = {}
        self.data['train'], self.data['val'], self.data['test'] = train, val, test

        # only set the selected partition
        self.starts = self.starts[self.mode]
        self.sizes = self.sizes[self.mode]
        self.p = self.p[self.mode]
        self.data = self.data[self.mode]

    def clear(self):
        pass

    def getNumClasses(self):
        # by characters
        #return self.data.shape[0]
        # by alphabets
        return len(self.starts)

    def __getitem__(self, index):

        index_inter_character = int(index / self.data.shape[1])
        index_intra_character = index - index_inter_character*self.data.shape[1]

        img = self.data[index_inter_character][index_intra_character]
        target = index_inter_character

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        # num characters * num samples of each character
        return self.data.shape[0] * self.data.shape[1]


class Omniglot_30_10_10_Pairs(Omniglot_30_10_10):

    def __init__(self, root = os.path.join('../../data', 'omniglot.npy'), train='train',
                 transform=None, target_transform=None, numTrials = 512):

        Omniglot_30_10_10.__init__(self, root, train, transform, target_transform)
        self.numTrials = numTrials

    def __getitem__(self, index):

        # set the choice function to random
        np.random.seed(None)

        num_alphbts = len(self.starts)

        similar_characters = True
        if index > self.numTrials/2: #different characters
            similar_characters = False

        # Across alphabet task is much simpler as it is easy to distinguish characters
        # belonging to different languages, compared to distinguishing characters belonging
        # to the same language.

        # characters from different alphabets
        alphbt_idx = choice(num_alphbts, size=2, replace=False, p=self.p)
        char_offset1 = choice(self.sizes[alphbt_idx[0]], 1, replace=False)
        diff_idx1 = self.starts[alphbt_idx[0]] + char_offset1 - self.starts[0]
        char_offset2 = choice(self.sizes[alphbt_idx[1]], 1, replace=False)
        diff_idx2 = self.starts[alphbt_idx[1]] + char_offset2 - self.starts[0]
        # characters from the same alphabet
        same_idx = choice(range(self.data.shape[0]))

        if similar_characters:
            img = self.data[same_idx, choice(self.data[same_idx].shape[0], 2, replace=False)]
            img1 = img[0]
            img2 = img[1]
        else:
            img1 = self.data[diff_idx1, choice(self.data[diff_idx1].shape[0], 1)][0]
            img2 = self.data[diff_idx2, choice(self.data[diff_idx2].shape[0], 1)][0]

        img1 = np.squeeze(img1)
        img2 = np.squeeze(img2)

        target = int(similar_characters)

        # Find if there is the agumentation function with a seed. If it is,
        # then establish the seed = self.agumentation_seed
        for elem in self.transform.transforms:
            if str(type(elem)) == "<class 'util.cvtransforms.AugmentationAleju'>":
                elem.seed = self.agumentation_seed

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # set the choice function to random
        np.random.seed(None)

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


class OmniglotOS(OmniglotBase):

    def __init__(self, root = os.path.join('../../data', 'omniglot.npy'), train='train',
                 reduced_dataset = False,
                 transform=None, target_transform=None, targetsByCharacters=True):
        OmniglotBase.__init__(self, root, train,
                                 transform, target_transform)
        self.mode = train
        self.reduced_dataset = reduced_dataset
        self.targetsByCharacters = targetsByCharacters
        num_train_chars = self.a_start[29] + self.a_size[29]
        train = self.chars[:num_train_chars, :16]  # (964, 16, H, W)
        val = self.chars[:num_train_chars, 16:]  # (964, 4, H, W)
        test = self.chars[num_train_chars:]  # (659, 20, H, W)

        # slicing indices for splitting a_start & a_size
        i = 30
        self.starts = {}
        self.starts['train'], self.starts['val'], self.starts['test'] = \
            self.a_start[:i], self.a_start[:i], self.a_start[i:]
            # self.a_start[:10], self.a_start[:10], self.a_start[10:15]
        self.sizes = {}
        self.sizes['train'], self.sizes['val'], self.sizes['test'] = \
            self.a_size[:i], self.a_size[:i], self.a_size[i:]
            # self.a_size[:10], self.a_size[:10], self.a_size[10:15]

        self.p = {}
        self.p['train'], self.p['val'], self.p['test'] = \
            self.size2p(self.sizes['train']), self.size2p(self.sizes['val']), self.size2p(self.sizes['test'])

        self.data = {}
        self.data['train'], self.data['val'], self.data['test'] = train, val, test

        self.num_drawers = {}
        self.num_drawers['train'], self.num_drawers['val'], self.num_drawers['test'] = 16, 4, 20

        # only set the selected partition
        self.starts = self.starts[self.mode]
        self.sizes = self.sizes[self.mode]
        self.p = self.p[self.mode]
        self.data = self.data[self.mode]
        self.num_drawers = self.num_drawers[self.mode]

    def clear(self):
        pass

    def getNumClasses(self):
        # by characters
        #return self.data.shape[0]
        # by alphabets
        return len(self.starts)

    def __getitem__(self, index):

        index_inter_character = int(index / self.data.shape[1])
        index_intra_character = index - index_inter_character*self.data.shape[1]

        img = self.data[index_inter_character][index_intra_character]
        if self.targetsByCharacters: # by characters
            target = index_inter_character
        else: # by alphabets
            target = np.arange(len(self.starts))[np.array(self.starts) <= index_inter_character][-1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        # num characters * num samples of each character
        return self.data.shape[0] * self.data.shape[1]

class OmniglotOSPairs(OmniglotOS):

    def __init__(self, root = os.path.join('../../data', 'omniglot.npy'), train='train',
                 reduced_dataset=False,
                 transform=None, target_transform=None,
                 isWithinAlphabets = True,
                 numTrials = 512):

        OmniglotOS.__init__(self, root, train, reduced_dataset, transform, target_transform)
        self.isWithinAlphabets = isWithinAlphabets
        self.numTrials = numTrials

    def __getitem__(self, index):

        num_alphbts = len(self.starts)

        similar_characters = True
        if index > self.numTrials/2: #different characters
            similar_characters = False

        if self.isWithinAlphabets:
            # choose similar chars
            same_idx = choice(range(self.data.shape[0]))

            # choose dissimilar chars within alphabet
            alphbt_idx = choice(num_alphbts, p=self.p)
            char_offset = choice(self.sizes[alphbt_idx], 2, replace=False)
            diff_idx = self.starts[alphbt_idx] + char_offset - self.starts[0]

            img = None
            if similar_characters:
                img = self.data[same_idx, choice(self.num_drawers, 2, replace=False)]
            else:
                img = self.data[diff_idx, choice(self.num_drawers, 2)]

            img1 = img[0]
            img2 = img[1]

        else:
            # Across alphabet task is much simpler as it is easy to distinguish characters
            # belonging to different languages, compared to distinguishing characters belonging
            # to the same language.

            # characters from different alphabets
            alphbt_idx = choice(num_alphbts, size=2, replace=False, p=self.p)
            char_offset1 = choice(self.sizes[alphbt_idx[0]], 1, replace=False)
            diff_idx1 = self.starts[alphbt_idx[0]] + char_offset1 - self.starts[0]
            char_offset2 = choice(self.sizes[alphbt_idx[1]], 1, replace=False)
            diff_idx2 = self.starts[alphbt_idx[1]] + char_offset2 - self.starts[0]
            # characters from the same alphabet
            same_idx = choice(range(self.data.shape[0]))

            if similar_characters:
                img = self.data[same_idx, choice(self.num_drawers, 2, replace=False)]
                img1 = img[0]
                img2 = img[1]
            else:
                img1 = self.data[diff_idx1, choice(self.num_drawers, 1)][0]
                img2 = self.data[diff_idx2, choice(self.num_drawers, 1)][0]

        img1 = np.squeeze(img1)
        img2 = np.squeeze(img2)

        target = int(similar_characters)

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


class OmniglotOneShot(OmniglotOS):

    def __init__(self, root = os.path.join('../../data', 'omniglot.npy'), train='train',
                 reduced_dataset = False,
                 transform=None, target_transform=None,
                 isWithinAlphabets = True,
                 numTrials = 512):

        OmniglotOS.__init__(self, root, train, reduced_dataset, transform, target_transform)
        self.isWithinAlphabets = isWithinAlphabets
        self.numTrials = numTrials

    def __getitem__(self, index):

        # set the choice function to random
        np.random.seed(None)

        num_alphbts = len(self.starts)

        if self.isWithinAlphabets:
            trial = np.zeros((20+1, self.data.shape[2], self.data.shape[3]), dtype='uint8')
            alphbt_idx = choice(num_alphbts)  # choose an alphabet
            char_choices = range(self.sizes[alphbt_idx])  # set of all possible chars
            key_char_idx = choice(char_choices)  # this will be the char to be matched

            # sample 19 other chars excluding key
            char_choices.pop(key_char_idx)
            other_char_idxs = choice(char_choices, 19)

            key_char_idx = self.starts[alphbt_idx] + key_char_idx - self.starts[0]
            other_char_idxs = self.starts[alphbt_idx] + other_char_idxs - self.starts[0]

            pos = range(20)
            key_char_pos = choice(pos)  # position of the key char out of 20 pairs
            target = key_char_pos
            pos.pop(key_char_pos)
            other_char_pos = np.array(pos, dtype='int32')

            #if index == 0:
            #    print('index: %d, target: %d, alphbt_idx: %d, key_char_idx: %d' % (index, target, alphbt_idx, key_char_idx))

            trial[key_char_pos] = self.data[key_char_idx, choice(self.num_drawers)]
            trial[other_char_pos] = self.data[other_char_idxs, choice(self.num_drawers)]
            # Set in the last position the key_char_idx
            trial[-1] = self.data[key_char_idx, choice(self.num_drawers)]
        else:

            trial = np.zeros((20+1, self.data.shape[2], self.data.shape[3]), dtype='uint8')
            alphbt_idx = choice(num_alphbts)  # choose an alphabet
            char_choices = range(self.sizes[alphbt_idx])  # set of all possible chars
            key_char_idx_n = choice(char_choices)  # this will be the char to be matched
            key_char_idx = self.starts[alphbt_idx] + key_char_idx_n - self.starts[0]
            # Set in the last position the key_char_idx
            trial[-1] = self.data[key_char_idx, choice(self.num_drawers)]

            # Select the position of the equal character
            pos = range(20)
            key_char_pos = choice(pos)  # position of the key char out of 20 pairs
            target = key_char_pos
            # Set the character with the same alphabet and char type.
            trial[key_char_pos] = self.data[key_char_idx, choice(self.num_drawers)]

            # Position of the other 19 characters.
            pos.pop(key_char_pos)
            other_char_pos = np.array(pos, dtype='int32')
            # Now sample from all the alphabets any char which is not the selected target alphabet-char.
            for i in range(19):
                alphbt_idx_other = choice(num_alphbts)  # choose an alphabet
                char_choices = range(self.sizes[alphbt_idx_other])  # set of all possible chars
                if alphbt_idx == alphbt_idx_other: # exclude the key
                    char_choices.pop(key_char_idx_n)
                # sample one chars excluded key if same alphabet
                other_char_idx = choice(char_choices)
                other_char_idx = self.starts[alphbt_idx_other] + other_char_idx - self.starts[0]
                trial[other_char_pos[i]] = self.data[other_char_idx, choice(self.num_drawers)]

        imgs = []
        if self.transform is not None:
            for i in range(trial.shape[0]):
                imgs.append(self.transform(trial[i]))
        trial = torch.stack(imgs)

        if self.target_transform is not None:
            target = self.target_transform(target)

        '''
        import cv2
        [cv2.imwrite('/home/aberenguel/tmp/cedar/im_' + str(i) + '.png',
                     trial[i].transpose(0, 1).transpose(1, 2).cpu().numpy() * 255) for i in range(trial.shape[0])]
        '''

        return trial, target

    def __len__(self):
        # num characters * num samples of each character
        return self.numTrials



class OmniglotOSLake(data.Dataset):

    def __init__(self, root = './data/one_shot/', image_size=None,
                 transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        self.charsX = np.load(os.path.join(root, 'X.npy'))
        self.charsY = np.load(os.path.join(root, 'y.npy'))

        # resize the images
        if image_size is not None:
            resized_X = np.zeros((20, 800, image_size, image_size), dtype='uint8')
            for i in xrange(20):
                for j in xrange(800):
                    resized_X[i, j] = resize(self.X[i, j], (image_size, image_size))
            self.X = resized_X

        self.mean_pixel = 0.0805  # dataset mean pixel
        #self.mean_pixel = self.chars.mean()  # used later for mean subtraction

    def __len__(self):
        return 1

    def __getitem__(self, index):
        X = self.X
        y = self.y
        X = X / 255.0
        X = X - self.mean_pixel
        X = X[:, :, np.newaxis]
        X = X.astype('float64')
        y = y.astype('int32')
        return X, y


class OmniglotVinyals(OmniglotBase):

    def __init__(self, root = os.path.join('../../data', 'omniglot.npy'), train='train',
                 transform=None, target_transform=None, num_trials = None):
        OmniglotBase.__init__(self, root, train,
                                 transform, target_transform)
        self.num_trials = num_trials

    def __getitem__(self, index):

        X = np.zeros((2 * 20, self.chars.shape[2], self.chars.shape[2]), dtype='uint8')

        char_choices = range(1200, 1623)  # set of all possible chars
        key_char_idx = choice(char_choices)  # this will be the char to be matched

        # sample 19 other chars excluding key
        char_choices.remove(key_char_idx)
        other_char_idxs = choice(char_choices, 19, replace=False)

        pos = range(20)
        key_char_pos = choice(pos)  # position of the key char out of 20 pairs
        pos.remove(key_char_pos)
        other_char_pos = np.array(pos, dtype='int32')

        drawers = choice(20, 2, replace=False)
        X[key_char_pos] = self.chars[key_char_idx, drawers[0]]
        X[other_char_pos] = self.chars[other_char_idxs, drawers[0]]
        X[20:] = self.chars[key_char_idx, drawers[1]]
        y = key_char_pos

        if self.transform is not None:
            X = torch.stack([self.transform(X[i]) for i in range(len(X))])

        if self.target_transform is not None:
            y = self.target_transform(y)

        return X,y

    def __len__(self):
        if self.num_trials is None:
            # num characters * num samples of each character
            return self.chars.shape[0] * self.chars.shape[1]
        else:
            return self.num_trials