import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
import random
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config.json")
args = parser.parse_args()
config = json.load(open(args.config))

def set_seed(seed):
    """
    Sets a randomness seed for random, np, and torch
    """
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        #generic_seed = seed
        #return generic_seed

class WikiArtImage:
    def __init__(self, imgdir, label, filename, img_size_modifier=None):
        # added the option to reduce the size of the image by providing a denominator for division
        # the reason is that I ran out of cuda memory with my approach, so run it on smaller images
        # otherwise the same as Asad's baseline
        self.img_size_modifier = img_size_modifier
        self.imgdir = imgdir
        self.label = label
        self.filename = filename
        self.image = None
        self.loaded = False


    def get(self):
        if not self.loaded:
            self.image = read_image(os.path.join(self.imgdir, self.label, self.filename)).float()
            if self.img_size_modifier:
                basesize = self.image.size(dim=1)
                newsize = int(basesize*self.img_size_modifier)
                self.reduce = transforms.Compose([transforms.Resize(newsize)])
                self.image = self.reduce(self.image)

            self.loaded = True

        return self.image
    

def initialise_AE(dataset, config, artstyle_encoder=False, num_styles=None, embed_dim=300):
    '''initialises instances of classes WikiArtEncoder and WikiArtDecoder (mode 1), or
                            of classes ArtstyleEncoder, ArtstyleDecoder, ArtstyleEmbedder (mode 2)
        arguments:
            dataset: dataset of class WikiArtData. Only used for image size/channel data,
            so content (i.e., test vs train) does not matter
            config: relevant config, whether from config file or stored config associated with a saved model
            artystle_encoder: if truthy, returns mode 2 classes, otherwise returns mode 1 classes
                if truthy, requires the following:
                num_styles: number of classes in data for initialising embedder
                embed_dim: output dimension of embedding layer
        returns:
            initialised instances of mode 1 or mode 2 classes
    '''
    # created this function since the initialisation of the classes (the Decoder in particular) requires some preamble
    activation_functions = {'relu': nn.ReLU,'tanh': nn.Tanh, 'gelu': nn.GELU} #'leakyrelu': nn.LeakyReLU(0.1), 
    device = config['device']
    if config["activation_function"] not in activation_functions:
        act_fn = nn.ReLU
    else:
        act_fn = activation_functions[config['activation_function']]
    im0, _ = dataset[0]
    num_in_channels = im0.size(dim=0)
    basesize = im0.size(dim=1)
    latent_dim = get_latent_dim(config['latent_dim_specifier'], basesize=basesize, manual=config['latent_dim_manual'])

    # dummy input to give the encoder before training
    # the purpose is to initialise the encoder beforehand just to get the shape 
    # of the data just before flattening; this is used in the initialisation of the decoder
    dummy_input = torch.randn(1, num_in_channels, basesize, basesize)
    
    encoder = WikiArtEncoder(num_in_channels=num_in_channels, image_size=basesize,
                                representation_dim=latent_dim, act_fn=act_fn).to(device) #act_fn=act_fn
        #dummy_out = encoder(dummy_input.to(device))
    #else:
    
    dummy_out = encoder(dummy_input.to(device))
    
    #decoder uses this for initialisation:
    shape_before_flattening = encoder.shape_before_flattening
    
    
    #overwrite encoder with subclass encoder for part 3
    #still initialise the original (part 2) encoder to get shape_before_flattening
        #not a design decision I'd have made if I had incorporated part 3 when I first designed this function
    if artstyle_encoder:
        if not num_styles or not embed_dim:
            print("Provide the number of styles to learn embeddings for and embedding output dims")
            sys.exit() # not setting a default num_styles since this is based on the data, unlike the next two params,
                        # which are hyperparameters
        encoder = ArtstyleEncoder(num_in_channels=num_in_channels, image_size=basesize,
                                    representation_dim=latent_dim, act_fn=act_fn).to(device)
        embedder = ArtstyleEmbedder(num_in_channels=num_in_channels, image_size=basesize,
                                    representation_dim=latent_dim, act_fn=act_fn, num_styles=num_styles,
                                    embed_dim=embed_dim).to(device)
        decoder = ArtstyleDecoder(num_in_channels=num_in_channels, image_size=basesize, representation_dim=latent_dim,
                             shape_before_flattening=shape_before_flattening, act_fn=act_fn)
        return encoder, decoder, embedder

    decoder = WikiArtDecoder(num_in_channels=num_in_channels, image_size=basesize, representation_dim=latent_dim,
                             shape_before_flattening=shape_before_flattening, act_fn=act_fn).to(device)
    return encoder, decoder

class WikiArtDataset(Dataset):
    # most of this is self explanatory and/or identical to Asad's original script. I note where it differs but don't
    # give full documentation
    def __init__(self, imgdir, resampling=None, img_size_modifier=False, device="cpu", single_class=False, toy_sample=False):
        filedict = {}
        indices = []
        classes = set()
        
        print("Gathering files for {}".format(imgdir))

        if resampling:
            # for part 1, see get_classfreq_target function for details
            # a once-over to get the class freqs, since I need these to know
            # how much of the actual data to get per class if I'm resampling:
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
                        filedict[art] = WikiArtImage(imgdir, arttype, art, img_size_modifier=img_size_modifier)
                        indices.append(art)
                        classes.add(arttype)                   
                    modified_arttype_counts[arttype] = {'old': arttype_count, 'new': target_freq}

                else:
                    if toy_sample:
                        artfiles = artfiles[:50]
                    for art in artfiles:
                        if single_class:
                            if not single_class==arttype:
                                break
                        filedict[art] = WikiArtImage(imgdir, arttype, art, img_size_modifier=img_size_modifier)
                        indices.append(art)
                        classes.add(arttype)

        if resampling:
            print('The dataset classes have been rebalanced:')
            # for each class, prints previous number of images and new number of images
            print(modified_arttype_counts)
        print("...finished")
        
        self.filedict = filedict
        self.imgdir = imgdir
        self.indices = indices
        self.classes = classes
        self.device = device

    def get_classfreq_target(self, arttype_count, sampling=None):
        '''
        args:
            arttype_count: count of examples for given class
            sampling, (default=None): specification of sampling method, valid values:
                any non-truthy value
                'mean'
                'max'
                'min'
                float or int between 0 and 1 (noninclusive)
                float or int 1 or greater

                see comments below for specifics
        
        returns:
            an integer representing the target number of examples the class should have,
            used to determine if and by how much the class should be over/undersampled        
        '''
        if not sampling:
            print('not sampling')
            # not functionally differenet from no balancing, to be used for testsets
            return arttype_count
        elif sampling == 'mean':
            # target number of training examples will be the mean number of pics per category,
            # which will mean oversampling (via duplicate examples) for some classes or undersampling for some
            return round(sum(self.arttype_counts.values())/len(self.arttype_counts.keys()))
        elif sampling == 'max':
            # fully oversampling: all classes will duplicate examples until they have as many as the most frequent class
            return max(self.arttype_counts.values())
        elif sampling =='min':
            # fully undersampling: all classes except the least frequent will use a random sample of examples equal in number to the least frequent
            return min(self.arttype_counts.values()) ######can get relatively small, so returns a more specific decimal value
        elif isinstance(sampling, int) or isinstance(sampling, float):
            if 0 < float(sampling) < 1:
                # a number between 0 and 1 (bidirectionally noninclusive) is interpreted as the fraction of the most frequent class
                # that all classes will over- or undersample to. E.g, if the most frequent class has 2000 examples and sampling=0.5, 
                # all classes will end up with 1000 examples
                return round(max(self.arttype_counts.values())*float(sampling))
            elif int(sampling) >=1:
                # all class frequencies will be over- or undersampled to this number.
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
        label = imgobj.label
        image = imgobj.get().to(self.device)

        return image, label #returns class string instead of idx, which I get later. Reason being easier access to the string for printing/plot-labeling

def class_to_idx(classlist):
    class_to_index = {strclass: i for i, strclass in enumerate(classlist)}
    return class_to_index

def get_random_artstyles(classdict, batch_size):
    """
    Returns a list of random classes of len(batch_size)
    For style embedding transfer
    """
    sample = random.choices(range(0, len(classdict)), k=batch_size)

    classes = torch.as_tensor(sample)
    return classes


class WikiArtModel(nn.Module):
    # largely (but not entirely) unchanged from Asad's version. See readme for what was instead done for bonus 1
    def __init__(self, num_classes=27):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 1, (4,4), padding=2)
        self.maxpool2d = nn.MaxPool2d((4,4), padding=2)
        
        self.flatten = nn.Flatten()
        self.batchnorm1d = nn.BatchNorm1d(105*105)
        self.linear1 = nn.Linear(105*105, 300)
        self.dropout = nn.Dropout(config["dropout"])
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(300, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, image):
        output = self.conv2d(image)
        output = self.maxpool2d(output) 
        output = self.flatten(output)
        output = self.batchnorm1d(output)    
        output = self.linear1(output)
        output = self.dropout(output)
        output = self.relu(output)
        output = self.linear2(output)
        return self.softmax(output)
    

# modified from pyimagesearch.com/2023/07/17/implementing-a-convolutional-autoencoder-with-pytorch/
# another helpful source: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
class WikiArtEncoder(nn.Module):
    """
    Encoder part of autoencoder. Stored as separate model, but trained alongside decoder.
    """
    def __init__(self, num_in_channels, image_size, representation_dim, act_fn):
        """
        Specifies layer sizes/dimensions and activation function.
        Args:
            num_in_channels: number of channels in image (3 in WikiArt)
            image_size: size of side of image. Single int: assumes same-size height and width
            representation_dim: size of latent representation
            act_fn: torch activation function to use
        """
        # see readme regarding the lack of pooling layer
        super().__init__()
        self.act_fn = act_fn()
        self.representation_dim = representation_dim

        self.conv1 = nn.Conv2d(num_in_channels, image_size, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(image_size, image_size*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(image_size*2, image_size*4, kernel_size=3, stride=2, padding=1)
        self.shape_before_flattening = None,
        self.flatten = nn.Flatten()
        flattened_size = (image_size // 8) * (image_size // 8) * image_size*4
        self.representation_layer = nn.Linear(flattened_size, representation_dim)

    def forward(self, x):
        """
        Takes input image, returns latent representation
        """
        x = self.act_fn(self.conv1(x))
        x = self.act_fn(self.conv2(x))
        x = self.act_fn(self.conv3(x))

        self.shape_before_flattening = x.shape[1:] # shape of final conv layer, disregarding batch dim
        x = self.flatten(x)
        representation = self.representation_layer(x) # latent representations
        return representation

class WikiArtDecoder(nn.Module):
    """
    Inverse of Encoder
        same args as encoder except
            shape_before_flattening: shape of final conv layer in encoder (minus batch dimension)
    """
    def __init__(self, num_in_channels, image_size, representation_dim, shape_before_flattening, act_fn):
        super().__init__()
        
        self.act_fn = act_fn()
        self.shape_before_flattening = shape_before_flattening
        self.unrepresentation_layer = nn.Linear(representation_dim, np.prod(shape_before_flattening))
        self.deconv1 = nn.ConvTranspose2d(
            image_size*4, image_size*4, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            image_size*4, image_size*2, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            image_size*2, image_size, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.reconstruction_conv = nn.Conv2d(image_size, num_in_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        """
        Takes latent representation, returns reconstructed image
        """
        x  = self.unrepresentation_layer(x)
        x = x.view(x.size(0), *self.shape_before_flattening)
        x = self.act_fn(self.deconv1(x))
        x = self.act_fn(self.deconv2(x))
        x = self.act_fn(self.deconv3(x))
        x = self.reconstruction_conv(x)
        return x

class ArtstyleEmbedder(WikiArtEncoder):
    """
    subclass of WikiArtEncoder that tries to do what the encoder does from only a class label, hopefully capturing some commonalities
    of images of the class
    """
    def __init__(self, num_in_channels, image_size, representation_dim, act_fn, num_styles, embed_dim=300):
        """
        Same as WikiArtEncoder except:
            num_styles: number of possible input classes
            embed_dim: output size of embedding layer        
        """
        super().__init__(num_in_channels, image_size, representation_dim, act_fn)
        self.embeddings = nn.Embedding(num_styles, embed_dim)
        self.imspec = (num_in_channels, image_size, image_size) # size of first conv in Encoder
        self.linear = nn.Linear(embed_dim, np.prod(self.imspec)) # outputs the flat size of first image of encoder,
                                                                # to be turned into a conv matrix of equal shape of first conv
        self.style_layer_ids = None

    def style_layer_selection(self, style_layers):
        """
        Tells the embedder to return outputs of specified layers. Not necessary for embedding training but used turing 
        embedding style transfer training for style loss.
        """
        self.style_layer_ids = style_layers

    def forward(self, style):
        """
        Takes style tensor (single int), returns tensor in the shape of image latent representation
        if self.style_layer_ids = truthy:
            also returns
               style_outs: list of outputs of layers specified in the config to represent style
                style_rep: flattened representation of of final conv layer
        """
        style_outs = []
        x = self.embeddings(style)
        x = self.linear(x)
        x = x.view(x.size(0), *self.imspec)
        x = self.act_fn(self.conv1(x))
        if self.style_layer_ids:
            if 1 in self.style_layer_ids:
                style_outs.append(x)

        x = self.act_fn(self.conv2(x))

        if self.style_layer_ids:
            if 2 in self.style_layer_ids:
                style_outs.append(x) 

        x = self.act_fn(self.conv3(x))
        if self.style_layer_ids:
            if 3 in self.style_layer_ids:
                style_outs.append(x)
        

        self.shape_before_flattening = x.shape[1:]

        style_rep = self.flatten(x) # to be concatenated with the equivalent feature in the encoder
 

        representation = self.representation_layer(style_rep)
        
        if self.style_layer_ids: # Will be None while training the embedder
                                # Will be truthy when training the style transfer
            return representation, style_outs, style_rep 
        return representation




class ArtstyleEncoder(WikiArtEncoder):
    """
    Same as WikiArtEncoder except where noted
    """
    def __init__(self, num_in_channels, image_size, representation_dim, act_fn):
        super().__init__(num_in_channels, image_size, representation_dim, act_fn)
        
        self.style_layer_ids = None

        # combines the pre-latent-representation of the embedder and the encoder
        combined_size = ((image_size // 8) * (image_size // 8) * image_size*4)*2
        
        self.combrep_layer = nn.Linear(combined_size, representation_dim)

    def style_layer_selection(self, style_layers):
        self.style_layer_ids = style_layers
    
    def forward(self, x, style_representation):
        #content_outs = []
        style_outs = []

        x = self.act_fn(self.conv1(x))
        #if 1 in self.content_layers:
            #content_outs.append(x)
        if self.style_layer_ids:
            if 1 in self.style_layer_ids:
                style_outs.append(x)
            

        x = self.act_fn(self.conv2(x))
        #if 2 in self.content_layers:
            #content_outs.append(x)
        if self.style_layer_ids:
            if 2 in self.style_layer_ids:
                style_outs.append(x)

        x = self.act_fn(self.conv3(x))
       # if 3 in self.content_layers:
            #content_outs.append(x)
        if self.style_layer_ids:
            if 3 in self.style_layer_ids:
                style_outs.append(x)
       

        self.shape_before_flattening = x.shape[1:]

        x = self.flatten(x)
        x = torch.cat((x, style_representation), 1)

        representation = self.combrep_layer(x)
        if self.style_layer_ids:
            return representation, style_outs # also returns outputs of layers specified in config to represent style features"""
        return representation

class ArtstyleDecoder(WikiArtDecoder):
    """
    Actually the same as WikiArtDecoder
    """
    # shouldn't exist, see readme for why it does.

    def __init__(self, num_in_channels, image_size, representation_dim, shape_before_flattening, act_fn):
        super().__init__(num_in_channels, image_size, representation_dim, shape_before_flattening, act_fn)

    def style_layer_selection(self, style_layers):
        self.style_layer_ids = style_layers

    def forward(self, x):
        x  = self.unrepresentation_layer(x)
        x = x.view(x.size(0), *self.shape_before_flattening)
        x = self.act_fn(self.deconv1(x))
        x = self.act_fn(self.deconv2(x))
        x = self.act_fn(self.deconv3(x))
        x = self.reconstruction_conv(x)
        return x


def gram(x):
    """
    Takes input feature and returns its gram matrix for computing style loss
    """
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G



def gram_matrix(input):
    #alternate gram matrix, from https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    #not sure which works best, not using this one
    a, b, c, d = input.size()  
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t()) 
    return G.div(a * b * c * d)

            
def get_latent_dim(latent_dim_specifier, basesize=416, manual=False): #

    '''
    returns the number of dims of the compressed representation to be returned by the encoder
    args:
        latent_dim_specifier:
            if manual=True (or any truthy value): latent_dim_specifier will be used directly as the latent dim
            if manual=False (or any falsy value, default): latent_dim_specifier will be used as the denominator by which
            (base_channel_size*base_channel_size) will be divided to get the latent dim, 
            e.g., the dimension size of the compressed representation of Wikiart's images would be (416*416)/latent_dim_specifier
            ''' 
    if manual:           
        return int(latent_dim_specifier)
            
    return (basesize*basesize)//latent_dim_specifier