import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm
import random
from math import ceil
from sklearn.model_selection import train_test_split
import json
import argparse



from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config.json")
args = parser.parse_args()
config = json.load(open(args.config))

def set_seed(seed):
    random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)

class WikiArtImage:
    def __init__(self, imgdir, label, filename):
        self.imgdir = imgdir
        self.label = label
        self.filename = filename
        self.image = None
        self.loaded = False

    def get(self):
        if not self.loaded:
            self.image = read_image(os.path.join(self.imgdir, self.label, self.filename)).float()
            self.loaded = True

        return self.image
    


#def under_sampling():


class WikiArtDataset(Dataset):
    def __init__(self, imgdir, resampling=None, device="cpu"):
        filedict = {}
        indices = []
        #classes = set()
        
        
        print("Gathering files for {}".format(imgdir))

        if resampling:
            #a once-over to get the class freqs, since I need these to know how much of the actual data to get per class if I'm resampling
            arttype_counts = {}
            modified_arttype_counts = {}
            walking = os.walk(imgdir)
            for item in walking:
                arttype = os.path.basename(item[0])
                artfiles = item[2]
                if len(artfiles) > 0:
                    arttype_counts[arttype] = len(artfiles)

            self.arttype_counts = arttype_counts

        walking = os.walk(imgdir)
        for item in walking:
            sys.stdout.write('.')
            arttype = os.path.basename(item[0])
            artfiles = item[2]
            if len(artfiles) > 0: # only necessary for my resampling code, but seems cleaner to put here since even the original code does not use
                                # items that fail this condition

                if resampling:
                    arttype_count = arttype_counts[arttype]
                    target_freq = self.get_classfreq_target(arttype_count, sampling=resampling)
                    #print(target_freq)
                    #print(type(target_freq))
                    random.shuffle(artfiles)
                    artfiles_snapshot = artfiles
                    while len(artfiles) < target_freq:
                        artfiles.extend(artfiles_snapshot)

                    # some choices that have been made with this oversampling preprocessing:
                        # the oversampling is done by duplication
                        # the data is shuffled before being extended, so the new dataset will only have at most 1 more instance of an example
                        #   than any other
                        # the algorithm favours the number of examples for any given class to be the specified number (target_freq) over the
                        #   examples having the same number of duplicates. E.g., if get_classfreq_target returns 500 freq target (via whatever
                        #   method is specified by its 'sampling' parameter), the new dataset will have 500 examples, not 500 + whatever number
                        #   is needed for the examples to have been duplicated the same number of times.
                    count=0
                    for i in range(target_freq):
                        count+=1                        
                        art = artfiles[i]
                        filedict[art] = WikiArtImage(imgdir, arttype, art)
                        indices.append(art)
                        #classes.add(arttype)                   
                    modified_arttype_counts[arttype] = {'old': arttype_count, 'new': target_freq}

                else:
                    for art in artfiles:
                        filedict[art] = WikiArtImage(imgdir, arttype, art)
                        indices.append(art)
                        #classes.add(arttype)

        if resampling:
            print('The dataset classes have been rebalanced:')
            print(modified_arttype_counts)
        print("...finished")
        
        self.filedict = filedict
        self.imgdir = imgdir
        self.indices = indices
        #self.classes = classes
        self.device = device
        #print(arttype_counts)
        #self.arttype_max = max(arttype_counts.values())
        #self.arttype_min = min(arttype_counts.values())
        #print(self.arttype_max, self.arttype_min)


    def get_classfreq_target(self, arttype_count, sampling=None):
        '''
        args:
            arttype_count: count of examples for given class
            sampling: specification of sampling method, valid values:
                any non-truthy value
                'mean'
                'max'
                'min'
                float or int between 0 and 1 (noninclusive)
                float or int 1 or greater
                (default=None = equivalent to no changes to the dataset)

                see comments per for specifics
        
        returns:
            an integer representing the target number of examples the class should have,
            used to determine if and by how much the class should be over/undersampled
        
            ####the factor by which the given class's arttype count will be multiplied to determine
            ####if and by how much it should be up/downsampled
        
        '''
        if not sampling:
            print('not sampling')
            # not functionally differenet from no balancing, to be used for testsets
            return arttype_count
        elif sampling == 'mean':
            # target number of training examples will be the mean number of pics per category,
            # which will mean oversampling (via duplicate examples) for some classes or undersampling for some
            return round(sum(self.arttype_counts.values())/len(self.arttype_counts.keys()))
            #return round(avg_n_pics/arttype_count, ndigits=3)
        elif sampling == 'max':
            # fully oversampling: all classes will duplicate examples until they have as many as the most frequent class
            return max(self.arttype_counts.values())
        elif sampling =='min':
            # fully undersampling: all classes except the least frequent will use a random sample of examples equal in number to the least frequent
            return min(self.arttype_counts.values()) ######can get relatively small, so returns a more specific decimal value
        elif isinstance(sampling, int) or isinstance(sampling, float):
            if 0 < sampling < 1:
                # a number between 0 and 1 (bidirectionally noninclusive) is interpreted as the fraction of the most frequent class
                # that all classes will over- or undersample to. E.g, if the most frequent class has 2000 examples and sampling=0.5, 
                # all classes will end up with 1000 examples
                return round(max(self.arttype_counts.values())*sampling)
                #target = max(self.arttype_counts.values())*sampling
                #return round(target/arttype_count, ndigits=3)
            elif sampling >=1:

                # all class frequencies will be up- or downsampled to this number.
                # to be extra clear: the different handling under this condition compared to the previous one means
                # 0.8 will result in a significantly bigger dataset than 5
                return int(sampling)
        else:
            return arttype_count #equivalent to no resampling
        
    def __len__(self):
        return len(self.filedict)

    def __getitem__(self, idx):
        imgname = self.indices[idx]
        imgobj = self.filedict[imgname]
        ilabel = imgobj.label          #      class_to_tensor(imgobj.label)      #self.classes.index(imgobj.label)
        image = imgobj.get().to(self.device)

        return image, ilabel

def class_to_tensor(classlist):
    #print(classlist)
    class_to_index = {strclass: i for i, strclass in enumerate(classlist)}
    #print(class_to_index)
    #classdict = {classkey: torch.as_tensor(class_to_index[classkey]) for classkey in class_to_index}
    #print(classdict)
    return class_to_index


class WikiArtAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
