import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
import tqdm
from wikiart import initialise_AE, set_seed
from p2test import validate, extract_random_images, save_image_comparison
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config2", help="configuration file", default="config2.json")

args = parser.parse_args()
config = json.load(open(args.config2))
parentdir = config["parentdir"]
trainingdir = os.path.join(parentdir, config["trainingdir"])
testingdir = os.path.join(parentdir, config["testingdir"])
modelname = config["modelname"]
device = config["device"]
seed = config["seed"]
latent_dim_spec = config["latent_dim_specifier"]
manual_ld_bool = config["latent_dim_manual"]
batch_size=config["batch_size"]
activation_functions = {'relu': nn.ReLU,'tanh': nn.Tanh, 'gelu':nn.GELU} #'leakyrelu': nn.LeakyReLU(0.1), 

set_seed(seed)

def train(trainset, testset, epochs=3, per_epoch_comparison=False, device=device):
    """
    Trains autoencoder (Encoder + Decoder together)
    args:
        trainset: WikiArtDataset class
        testset: WikiArtDataset class, used for training validation
        per_epoch_comparison: if truthy, creates and saves images in ./looseimages every epoch to see recon progression
            recommended (for clarity, not to be able to run) to run clearim.py
            between instances of per_epoch_comparison=True runs of this function
    Saves best validation model and final model (one or two models depending on whether final is best)
    Saves in latest ae directory +1 and +2 respectively; e.g., if ./autoencoders/ae1 and ./autoencoders/ae2 already exist,
    the current run's best val model is saved in ae3, and (if not the same model) the last epoch model is saved in ae4.
    Saves checkpoint data text files in respective ae directory for easily accessible metadeta.

    Returns ae_ids for model loading
    """
    loader = DataLoader(trainset, batch_size=batch_size, shuffle=True )
    testloader = DataLoader(testset, batch_size=1, shuffle=True)

    if not os.path.exists('./autoencoders'):
        os.mkdir('./autoencoders')
    modeldir = "./autoencoders"
    encoder, decoder = initialise_AE(trainset, config)
    optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), config["learning_rate"]) #weight_decay = 1e-8
    criterion = torch.nn.MSELoss().to(device)
    best_val_loss = float("inf")
    num_images = config["num_image_samples"]

    # id used to save the autoencoder; the autoencoder and its checkpoint data (see below) will be saved in
    # a subdirectory in the "autoencoders" directory called ae+str(num+1) (e.g., "autoencoders/ae1", "ae2", etc.)
    # Requires that the autoencoder subdirs always remain incrementally named, so if ae2 is deleted from "autoencoders"
    # with subdirs ['ae1', 'ae2', 'ae3'], ae3 must be renamed to ae2
    ae_id = str(len(os.listdir(modeldir))+1)
    best_epoch = None

    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch+1))
        encoder.train()
        decoder.train()
        accumulated_loss = 0
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            X, _ = batch
            optimizer.zero_grad(set_to_none=True)
            encoded = encoder(X)
            decoded = decoder(encoded)
            loss = criterion(decoded, X)
            loss.backward()
            accumulated_loss += loss.item()
            optimizer.step()
            
            # for memory purposes
            del loss, encoded, decoded

        print("In epoch {}, loss = {}".format(epoch+1, accumulated_loss))

        train_loss = accumulated_loss / len(loader)
        val_loss = validate(encoder, decoder, testloader, criterion)
        print(
        f"Epoch {epoch + 1} | Average Train Loss: {train_loss:.4f} "
        f"| Average Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch+1
            aepath = os.path.join(modeldir, 'ae'+ae_id)
            if not os.path.exists(aepath):
                os.mkdir(aepath)
            config["checkpoint_at_epoch"] = epoch+1
            config["average_val_loss"] = val_loss
            torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "epoch": epoch+1, "config": config},
                    os.path.join(aepath, modelname))
            checkpoint_data = list(config.items())[-10:]
            checkpoint_data = [': '.join((str(d[0]), str(d[1])))+'\n' for d in checkpoint_data]
            checkpoint_path = os.path.abspath(os.path.join(os.getcwd(), 'autoencoders', 'ae'+ae_id, 'checkpointdata.txt'))
            with open(checkpoint_path, 'w') as f:
                f.writelines(checkpoint_data)
        if per_epoch_comparison:
            with torch.no_grad():
                samples, sample_labels = extract_random_images(testloader, encoder, decoder, num_images, device)
                save_image_comparison(samples, sample_labels, ae_id, midtrain_epoch=epoch+1)
    ae_ids = [ae_id]

    if not best_epoch == epochs:
        ae_id2 = str(int(ae_id)+1)
        aepath = os.path.join(modeldir, 'ae'+str(ae_id2))
        if not os.path.exists(aepath):
            os.mkdir(aepath)
        torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "epoch": epoch+1, "config": config},
                os.path.join(aepath, modelname))
            
        checkpoint_data = list(config.items())[-9:]
        checkpoint_data = [': '.join((str(d[0]), str(d[1])))+'\n' for d in checkpoint_data]
        checkpoint_path = os.path.abspath(os.path.join(os.getcwd(), 'autoencoders', 'ae'+ae_id2, 'checkpointdata.txt'))
        with open(checkpoint_path, 'w') as f:
            f.writelines(checkpoint_data)
        ae_ids.append(ae_id2)

    return ae_ids
