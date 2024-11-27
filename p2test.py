import os
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import tqdm
from wikiart import set_seed
import json
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config2", help="configuration file", default="config2.json")

args = parser.parse_args()

config = json.load(open(args.config2))

parentdir = config["parentdir"]
testingdir = config["testingdir"]
#testsampling = config["testsampling"] # equivalent to original, i.e. no modifications to testset
device = config["device"]
testingdir = os.path.join(parentdir, config["testingdir"])
seed = config["seed"]
set_seed(seed)

def validate(encoder, decoder, test_loader, criterion):
    """
    Validates model during training to determine which model to save    
    """
    # largely from:
    # https://pyimagesearch.com/2023/07/17/implementing-a-convolutional-autoencoder-with-pytorch/
    # set the encoder and decoder to evaluation mode

    encoder.eval()
    decoder.eval()
    # initialize the running loss to 0.0
    accumulated_loss = 0.0
    # disable gradient calculation during validation
    with torch.no_grad():
        # iterate through the test loader
        for batch_id, batch in enumerate(tqdm.tqdm(test_loader)):
            X, _ = batch
            X = X.to(device)
            encoded = encoder(X)

            decoded = decoder(encoded)
            loss = criterion(decoded, X)

            accumulated_loss += loss.item()
    # calculate the average loss over all batches
    # and return to the calling function
    return accumulated_loss / len(test_loader)

def save_image_comparison(image_recon, labels, ae_num, midtrain_epoch=False, style_transfer_id=None):
    '''
    Saves input images in a grid with 2 columns (original, reconstruction) for comparison
    
    Args:
        image_recon: tensor of alternating original and reconstructed images. Default output 0 of extract_random_images
        labels: list of labels corresponding to image_recon. Default output 1 of extract_random_images
        ae_num: autoencoder model id; specifies autoencoder directory to save grid in. E.g., ae_num=2 saves comparison in
            ./autoencoders/ae2.
        midtrain_epoch: if provided, creates and saves image grid in ./looseimages and names file per provided midtrain_epoch value
            not supported for style transfer
        style_transfer_id: id of style_transfer model. E.g., if ae_num=2 and style_transfer_id=1, saves comparison in
            ./autoencoders/ae2/transfermodel1
    
    '''
    # modified from https://pyimagesearch.com/2023/07/17/implementing-a-convolutional-autoencoder-with-pytorch/

    # calculate the number of rows needed to display all the images
    num_rows = len(image_recon) // 2
    # create a grid of images using torchvision's make_grid function
    grid = make_grid(
        image_recon.cpu(), nrow=2, padding=2, normalize=True
    )
    # convert the grid to a NumPy array and transpose it to
    # the correct dimensions
    grid_np = grid.numpy().transpose((1, 2, 0))
    # create a new figure with the appropriate size
    plt.figure(figsize=(2 * 2, num_rows * 2))
    # show the grid of images
    plt.imshow(grid_np)
    # remove the axis ticks
    plt.axis("off")
    # set the title of the plot
    plt.title("Originals, Reconstructions", fontsize=16)
        # add labels for each image in the grid
    for i in range(len(image_recon)):
        # calculate the row and column of the current image in the grid
        row = i // 2
        col = i % 2
        # get the name of the label for the current image
        #label_name = config.CLASS_LABELS[labels[i].item()]
        label_name = labels[i]
        # add the label name as text to the plot
        if i % 2 == 0 or style_transfer_id: # adds every other label if not style transferring, since other cases simply have duplicate labels
            plt.text(
                col * (image_recon.shape[3] + 2) + image_recon.shape[3] // 2,
                (row + 1) * (image_recon.shape[2] + 2) - 5,
                label_name,
                fontsize=12,
                ha="center",
                va="center",
                color="white",
                bbox=dict(facecolor="black", alpha=0.5, lw=0),
            )

    print('saving')

    # various save locations
    if midtrain_epoch:
        #not supported for style transfer
        if not os.path.exists('./looseimages/'):
            os.mkdir('./looseimages')
        imagedir = './looseimages'
        imagepath = os.path.join(imagedir, 'epoch'+str(midtrain_epoch)+'.png')
    elif style_transfer_id:
        dirpath = os.path.join('./autoencoders', 'ae'+ae_num, 'transfermodel'+str(style_transfer_id))
        num_comps = ' '.join(os.listdir(dirpath)).count('comparison')
        imagepath = os.path.join(dirpath, 'comparison'+str(num_comps+1)+'.png')
    else:
        dirpath = os.path.join('./autoencoders', 'ae'+ae_num)
        num_comps = ' '.join(os.listdir(dirpath)).count('comparison')
        imagepath = os.path.join(dirpath, 'comparison'+str(num_comps+1)+'.png')
        

    plt.savefig(imagepath, bbox_inches="tight")
    plt.close()

def extract_random_images(shuffled_dataloader, encoder, decoder, num_images, device, out_set='both'):
    """
    returns list of images to be saved and labels to go with those images
    args:
        shuffled_dataloader: pytorch dataloader with batch size 1
        encoder: WikiArtEncoder class
        decoder: WikiArtDecoder class
        num_images: number of images to return
        out_set: set of images to return
            originals: only original images
            reconstructions: only reconstructions
            anything else: both
        if both are returned, the out list is num_images*2 (10 originals and a reconstruction per image)
    """ # never ended up using anything but both since the comparison is kind of important, but figured I'd leave the code in

    out = []
    sample_labels = []
    if not out_set == 'originals' and not out_set == 'reconstructions':
        num_images = num_images*2
    
    for image, label in shuffled_dataloader:

        image = image.to(device)
        if out_set == 'originals':
            out.append(image)
        else:
            encoded = encoder(image)
            reconstruction = decoder(encoded)
            if out_set == 'reconstructions':
                out.append(reconstruction)
            else:
                out.append(image)
                out.append(reconstruction)      
                sample_labels.append(label[0])
                # labels duplicated if out_set does not specify only image or recon to match the alternating
                # og/recon image in the out list
        sample_labels.append(label[0])

        if len(out) == num_images:
            break
    
    out = torch.cat(out, dim=0)

    return out, sample_labels # if out is both originals and recons, it alternates between original and recons: [o1, r1, o2, r2...]
                                #this is what my saving function expects.

def plot_pca_kmeans(test_loader, encoder, ae_id):
    #used https://www.askpython.com/python/examples/plot-k-means-clusters-python
    """
    Saves a plot with the latent space data reduced by PCA, and another with that PCA representation clustered by K-means classes
        ae_id: autoencoder ID; e.g., if ae_id=2, saves plots in ./autoencoders/ae2
    """
    representations = []
    labels = []
    for im, lb in test_loader:
        rep = encoder(im)
        representations.extend(rep.detach().cpu().numpy())
        labels.extend(lb)
        del im, lb
    
    representations = np.array(representations)
    class_labels = np.array(labels)
    u_class_labels = np.unique(class_labels)
    pca = PCA()

    #inilitalise kmeans to create num_classes clusters
    kmeans = KMeans(n_clusters=len(u_class_labels))

    pca_reps = pca.fit_transform(representations)

    cluster_labels = kmeans.fit_predict(pca_reps)

    u_cluster_labels = np.unique(cluster_labels)

    dirpath = os.path.join('./autoencoders', 'ae'+ae_id)
    num_plots = ' '.join(os.listdir(dirpath)).count('pca_kmeans')
    filepath = os.path.join(dirpath, 'pca_kmeans'+str(num_plots+1))

    for lb in u_cluster_labels:
        plt.scatter(pca_reps[cluster_labels == lb , 0] , pca_reps[cluster_labels == lb , 1] , label = lb)
    plt.legend()
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()

    num_plots = ' '.join(os.listdir(dirpath)).count('pca_class')
    filepath = os.path.join(dirpath, 'pca_class'+str(num_plots+1))

    for lb in u_class_labels:
        plt.scatter(pca_reps[class_labels == lb , 0] , pca_reps[class_labels == lb , 1] , label = lb)
    #plt.legend()
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()