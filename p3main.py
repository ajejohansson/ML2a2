import sys
import os
import torch
from torch.utils.data import DataLoader
from wikiart import WikiArtDataset, initialise_AE, class_to_idx, set_seed
import json
import argparse
from p3funcs import train_style_embeddings, style_transfer_train, style_transfer
from random import randint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config2", help="configuration file", default="config2.json")
    args = parser.parse_args()
    config = json.load(open(args.config2))
    device = config["device"]
    num_images = config["num_image_samples"]
    toy_sample = config["toy_sample"]
    parentdir = config["parentdir"]
    trainingdir = os.path.join(parentdir, config["trainingdir"])
    testingdir = os.path.join(parentdir, config["testingdir"])
    embedder_epochs = config["embedder_epochs"]
    transfer_epochs = config["transfer_epochs"]
    content_weight = config["content_weight"]
    style_layers = config["style_layers"]
    style_weight = config["style_weight"]
    embed_dim = config["embed_dim"]
    device = config['device']
    seed = config["seed"]
    train_embedder = config["new_embedder"]
    train_transfer = config["train_transfer"]
    style_transfer_id = config["transfer_id"]
    single_image = config["single_image_comparison"]
  

    ae_id = config["load_ae_id"] # select autoencoder id. Must be trained beforehand with p2main.py
    if not train_transfer:
        transfer_id = config["transfer_id"] # select transfermodel id
        transferpath = os.path.join('./autoencoders', 'ae'+str(ae_id), 'transfermodel'+str(transfer_id), 'StyleEmbeddingTransfer.pth')  
    
    modelpath = os.path.join('./autoencoders', 'ae'+str(ae_id), config['modelname'])

    try:
        autoencoder_dict = torch.load(modelpath)
    except:
        print(modelpath, "does not exist. Set config load_ae_id to an id that exists in ./autoencoders (e.g., 2 for ae2)")
        print("or train a new autoencoder with p2main.py")
        sys.exit()
    config = autoencoder_dict["config"]

    img_size_modifier = config["img_size_modifier"]    
    set_seed(seed)

    traindataset = WikiArtDataset(trainingdir, device=device, img_size_modifier=img_size_modifier, toy_sample=toy_sample)
    testdataset = WikiArtDataset(testingdir, device=device, img_size_modifier=img_size_modifier)
    trainclasses = traindataset.classes
    testclasses = testdataset.classes
    classes = list(trainclasses.union(testclasses))
    classdict = class_to_idx(classes)
    ae_id = str(ae_id)
    modelpath = os.path.join('./autoencoders', 'ae'+ae_id, config['modelname'])  
    autoencoder_dict = torch.load(modelpath)
    config = autoencoder_dict["config"]
    loader = DataLoader(traindataset, batch_size=config['batch_size'], shuffle=True )
    encoder, decoder, embedder = initialise_AE(traindataset, config, artstyle_encoder=True, num_styles=len(classdict), embed_dim=embed_dim)
    encoder.load_state_dict(autoencoder_dict["encoder"], strict=False)
    decoder.load_state_dict(autoencoder_dict["decoder"])
    
    if train_embedder:
        # train and save embedder and update metadata with embedding specs
        # note: updates existing autoencoder model dict, so if specified autoencoder (e.g. ae2, ae3) already has
        # an associated embedder, this embedder is overwritten
        embedder = train_style_embeddings(loader, embedder, decoder, classdict, config, device, epochs=embedder_epochs)
        autoencoder_dict["config"]["embedder_epochs"] = embedder_epochs
        autoencoder_dict["embedder"] = embedder
        autoencoder_dict["emb_dim"] = embed_dim
        checkpoint_path = os.path.abspath(os.path.join(os.getcwd(), 'autoencoders', 'ae'+ae_id, 'checkpointdata.txt'))

        # Get rid of final line, add embedder epoch metadata
        # Final line is either for overwritten embedder, or a blank line from my previous code,
        # so should be fine either way (though I've not tested extensively):
        with open(checkpoint_path, 'r') as f:
            d = f.read()
            d = d.split('\n')
            d = d[:-1]
            d.append("embedder epochs: "+str(embedder_epochs))
            d = "\n".join(d)
        with open(checkpoint_path, 'w+') as f:
            for i in range(len(d)):
                f.write(d[i])

        torch.save(autoencoder_dict, modelpath)
    else:
        try:
            embedder = autoencoder_dict["embedder"]
            embed_dim = autoencoder_dict["emb_dim"] #overwrite whatever is in the config with what the given embedder was actually trainied with
        except:
            # the rest of the script cannot run without an embedder, so is interrupted if the config is set to not train a new one
            # and there isn't an existing one
            print('Specified autoencoder does not have an associated embedder model.')
            print("Set 'new_embedder' in the config file to 'true', or specify another autoencoder ID.")
            sys.exit()


    if train_transfer:
        # trains and saves transfermodel in specified ae directory (e.g. ae2). If specified ae has no existing transfermodel,
        # new model will be called transfermodel1, if there is one existing, new one will be called transfermodel2, etc.
        embedder, encoder, decoder = style_transfer_train(loader, embedder, encoder, decoder, classdict, config,
                            device, style_layers, content_weight, style_weight, epochs = transfer_epochs)
        
        dirpath = os.path.join('./autoencoders', 'ae'+ae_id)
        num = ' '.join(os.listdir(dirpath)).count('transfermodel')
        style_transfer_id = str(num+1)
        modelname = 'transfermodel'+style_transfer_id
        transferpath = os.path.join(dirpath, modelname)
        os.mkdir(transferpath)

        torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "embedder": embedder.state_dict(), "style_layers": style_layers},
                        os.path.join(transferpath, 'StyleEmbeddingTransfer.pth'))
        metadata = [("epochs", transfer_epochs), ("content weight", content_weight), 
                    ("style layers", style_layers), ("style weight", style_weight)]
        metadata = [': '.join((str(d[0]), str(d[1])))+'\n' for d in metadata]
        metapath = os.path.abspath(os.path.join(os.getcwd(), 'autoencoders', 'ae'+ae_id, modelname, 'metadata.txt'))
        with open(metapath, 'w') as f:
            f.writelines(metadata)
        transferpath = os.path.join(transferpath, 'StyleEmbeddingTransfer.pth')


    with torch.no_grad():
        set_seed(randint(1, 10000))
        
        testloader = DataLoader(testdataset, batch_size=1, shuffle=True )
        
        try:
            transfer_dict = torch.load(transferpath)
        except:
            print(transferpath, "does not exist. Set config load_ae_id to an id that exists in ./autoencoders (e.g., 2 for ae2)")
            print("and transfer_id to a transfermodel that exists within that ae directory (e.g. 1 for transfermodel1).")
            print("Alternatively, train a new autoencoder with p2main.py and/or a new transfermodel (train_transfer=true).")
            sys.exit()

        style_layers = transfer_dict["style_layers"]
        encoder, decoder, embedder = initialise_AE(traindataset, config, artstyle_encoder=True, num_styles=len(classdict), embed_dim=embed_dim)
        embedder.load_state_dict(transfer_dict["embedder"])
        encoder.load_state_dict(transfer_dict["encoder"], strict=False)
        decoder.load_state_dict(transfer_dict["decoder"])

        # create comparison of original images and new style-altered images
        # comparison files are created in ./autoencoders/aeI/transfermodelJ, where I is config-specified autoencoder id,
        # and J is either the id of the newly trained transfermodel (if a new one was trained) or the config-specified transfer_id 
        # spoiler: model does not learn function
        style_transfer(testloader, embedder, encoder, decoder, classdict, style_layers,
                    ae_id=ae_id, style_transfer_id=style_transfer_id, num_images=num_images, single_image=single_image)

if __name__ == "__main__":
    main()
    
