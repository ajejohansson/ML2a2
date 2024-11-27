import sys
import os
import torch
from torch.utils.data import DataLoader
from wikiart import WikiArtDataset, initialise_AE, class_to_idx, set_seed
from p2train import train
from p2test import save_image_comparison, extract_random_images, plot_pca_kmeans
import json
import argparse



# see readme for config description and running instructions

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config2", help="configuration file", default="config2.json")
    args = parser.parse_args()
    config = json.load(open(args.config2))
    device = config["device"]
    num_images = config["num_image_samples"]
    train_new = config["train_new"]
    toy_sample = config["toy_sample"]
    compare = config["compare"]
    plot_latent = config["plot_latent"]
    per_epoch_comparison = config["per_epoch_comparison"]
    
    # possible overriding of config with the one stored with the model
    # device/num_images will always be from the global config file
    # 
    if not train_new:
        # ae_ids is an id list of the autoencoders to be tested.
        # When there is new training, that is a list of len 1 or 2 (the model after final epoch,
        # or the model after final epoch+the model after best validation loss, if those are different models)
        # When loading the id, I make it a list of len 1 
        ae_ids = [config["load_ae_id"]]
        ae_id = ae_ids[0] # note the difference between ae_ids and ae_id. The former will be iterated over later, the latter is used here for modelpath
        modelpath = os.path.join('./autoencoders', 'ae'+str(ae_id), config['modelname'])                                
        autoencoder_dict = torch.load(modelpath)
        config = autoencoder_dict["config"]


    parentdir = config["parentdir"]
    trainingdir = os.path.join(parentdir, config["trainingdir"])
    testingdir = os.path.join(parentdir, config["testingdir"])
    seed = config["seed"]
    img_size_modifier = config["img_size_modifier"]

    set_seed(seed)

    traindataset = WikiArtDataset(trainingdir, device=device, img_size_modifier=img_size_modifier, toy_sample=toy_sample)

    testdataset = WikiArtDataset(testingdir, device=device, img_size_modifier=img_size_modifier)
    trainclasses = traindataset.classes
    testclasses = testdataset.classes
    classes = list(trainclasses.union(testclasses))
    classdict = class_to_idx(classes)



    if train_new:
        ae_ids = train(traindataset, testdataset, epochs=config["epochs"], per_epoch_comparison=per_epoch_comparison,
                    device=device)

        
    for ae_id in ae_ids:
        ae_id = str(ae_id)
        modelpath = os.path.join('./autoencoders', 'ae'+str(ae_id), config['modelname'])                                
        autoencoder_dict = torch.load(modelpath)
        config = autoencoder_dict["config"]
        encoder, decoder = initialise_AE(testdataset, config)
        encoder.load_state_dict(autoencoder_dict["encoder"])
        decoder.load_state_dict(autoencoder_dict["decoder"])

        testloader = DataLoader(testdataset, batch_size=1, shuffle=True)

        encoder.eval()
        decoder.eval()
        
        with torch.no_grad():
            if compare:
                sample, sample_labels = extract_random_images(testloader, encoder, decoder, num_images, device)
                save_image_comparison(sample, sample_labels, ae_id)
            if plot_latent:
                plot_pca_kmeans(testloader, encoder, ae_id)



if __name__ == "__main__":
    main()
    
