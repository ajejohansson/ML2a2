import torch
from torch.optim import AdamW, Adam
import tqdm
from wikiart import gram, get_random_artstyles
from p2test import save_image_comparison
import json
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config2", help="configuration file", default="config2.json")
args = parser.parse_args()
config = json.load(open(args.config2))
content_weight = config["content_weight"]
style_layers = config['style_layers']
style_weight = config["style_weight"]
embed_dim = config["embed_dim"]
device = config['device']

def train_style_embeddings(trainloader, embedder, decoder, classdict, config, device, epochs=3):
    """
    Trains style embedder: an alternate encoding that does not get an input image, only a class label
        args:
            ArtstyleEmbedder class
    """
    # the rest of the arguments have been described elsewhere
    # don't have super much faith that this embedder actually learns much, but (to be fair) I think whatever commonality exists
    # between artstyles is a hard function to master. See part 2 pca plotting to see how at least my autoencoder doesn't capture features
    # across classes, so the embedder might be doomed from the start.

    embedder = embedder.to(device)
    decoder = decoder.to(device)
    
    loader = trainloader
    optimizer = Adam(embedder.parameters(), config["learning_rate"]) #trains only embedder, using pretrained decoder
    MSE_loss = torch.nn.MSELoss().to(device)

    print("Training style embedder")
    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch+1))
        embedder.train()

        accumulated_loss = 0
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            X, labels = batch #not calling labels 'y' since I use it as input here

            input_lb = [classdict[lb] for lb in labels]
            input_lb = torch.as_tensor(input_lb).to(device)

            optimizer.zero_grad(set_to_none=True) #https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
            encoded = embedder(input_lb)
            decoded = decoder(encoded.to(device))
            loss = MSE_loss(decoded, X)
            loss.backward()
            accumulated_loss += loss.item()
            optimizer.step()

            del loss, encoded, decoded
        
        train_loss = accumulated_loss / len(loader)

        print("In epoch {}, loss = {}".format(epoch+1, accumulated_loss))
        print(f"Epoch {epoch + 1} | Average Train Loss: {train_loss:.4f} ")

    return embedder


def style_transfer_train(loader, embedder, encoder, decoder, classdict, config,
                         device, style_layer_ids, content_weight, style_weight, epochs=3):
    """
    Args:
        embedder, encoder, decoder: Artstyle versions.
        style_layer_ids: list of integers 1-3. Each number represents a conv layer output in the encoder/embedder for which will
        be used to compute style loss. If style_layer_ids = [2,3], conv layers 2 and 3 will be used.
        content_weight: factor to multiply content loss with
        style_weight: factor to multiply style loss with
    """
    embedder = embedder.to(device)
    embedder.style_layer_selection(style_layer_ids) # sets embedder to not only return final representation, but also those specified in style_layer_ids
    encoder = encoder.to(device)
    encoder.style_layer_selection(style_layer_ids) # same
    decoder = decoder.to(device)

    embedder.style_layer_selection(style_layer_ids)
    
    optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()) +list(embedder.parameters()), config["learning_rate"])

    MSE_loss = torch.nn.MSELoss().to(device)

    print("Training style transfer")
    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch+1))
        embedder.train()
        encoder.train()
        decoder.train()

        accumulated_style_loss = 0.0
        accumulated_content_loss = 0.0
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            X, _ = batch
            optimizer.zero_grad(set_to_none=True)

            # Getting random labels to input to the embedder as I think this will generalise better than just passing the matching
            # label every time.
            # Might be even better to train every batch with every label, but that's a lot of extra computation for something
            # that I assume will not work very well in the end anyway
            input_labels = get_random_artstyles(classdict, X.size(0)).to(device) # Note that this weighs all class styles equally regardless of frequency
            emb_encoded, emb_style_outs, style_rep = embedder(input_labels)

            encoded, style_outs = encoder(X, style_rep) # encode content images and get features to be used for style loss
            decoded = decoder(encoded)
            content_loss = content_weight*MSE_loss(decoded, X)# making the naive assumption of just using the decoder output
                                               # to get content loss. A more sophisticated approach would be to
                                               # also specify layers to use (or weigh heaver) for content loss
            accumulated_content_loss +=content_loss.item()

            style_loss = 0
            for i in range(len(style_outs)):
                enc_gram_i = gram(style_outs[i])
                emb_gram_i = gram(emb_style_outs[i])
                style_loss += MSE_loss(enc_gram_i, emb_gram_i)
            
            style_loss = style_weight*style_loss

            accumulated_style_loss+=style_loss.item()

            total_loss = style_loss + content_loss
            total_loss.backward()
            optimizer.step()

            del style_loss, content_loss, total_loss, encoded, decoded, emb_encoded, style_rep #del anything that requires grad
        
        train_loss = (accumulated_content_loss+accumulated_style_loss)/ len(loader)

        print("In epoch {}, content loss = {}, style loss = {}".format(epoch+1, accumulated_content_loss, accumulated_style_loss))
        print(f"Epoch {epoch + 1} | Average combined train Loss: {train_loss:.4f} ")

    return embedder, encoder, decoder


def style_transfer(loader, embedder, encoder, decoder, classdict, style_layer_ids, ae_id=2,
                   style_transfer_id=1, num_images=5, single_image=False):

    """
    Applies class style transfer to images. Note that this does not attempt to style transfer from a style image to a content image,
    but from an artstyle to a content image. Saves a grid with num_images originals, each compared to a (not very successfully) attempted
    ST reconstruction 

    Args:
        style_transfer_id: style transfer model identifier. E.g., if ae_id=2 and style_transfer_id=1, the transfer model will be transfer model
        1 associated with autoencoder 2, i.e., located in ./autoencoders/ae2/transfermodel1. Any transfer model is associated with an autoencoder
        since the embedder will be trained using parts of a previous autoencoder, and the transfermodel will consist of further trained
        embedder, encoder, decoder
        single_image: if truthy, the saved grid will use the same original image every time, compared to style transfer attempts of different styles
        (e.g. ae2, transfermodel2, comparison2)
        Otherwise, the grid will have num_images random images with a style transfer image (e.g., ae2, transfermodel1, comparison1)
    """
    embedder = embedder.to(device)
    embedder.style_layer_selection(style_layer_ids)
    encoder = encoder.to(device)
    encoder.style_layer_selection(style_layer_ids)
    decoder = decoder.to(device)

    embedder.style_layer_selection(style_layer_ids)
    embedder.eval()
    encoder.eval()
    decoder.eval()


    images = []
    og_styles = []
    new_styles = []

    # for generating transfers of num_images different styles for one image:
    if single_image:
        for image, og_style in loader:
            while len(images) < num_images:
                images.append(image)
                og_styles.append(og_style)
                new_style = og_style

                ## get a random style; if it's the same as the original or already in new_styles, try again:
                while new_style == og_style or new_style in new_styles:
                    new_style_idx = random.randint(0, len(classdict)-1)
                    new_style = [list(classdict.keys())[new_style_idx]]
                new_styles.append(new_style)
    # for generating a transfer of a random style for num_images images
    else: 
        for image, og_style in loader:
            #working with batch size = 1 items here, will unpack this later since my models assume a batch dim
            og_styles.append(og_style)
            images.append(image)
            new_style = og_style

            # get a random style; if it's the same as the original, try again:
            while new_style == og_style:
                new_style_idx = random.randint(0, len(classdict)-1)
                new_style = [list(classdict.keys())[new_style_idx]]
            new_styles.append(new_style)
            if len(images) == num_images:
                break

    ordered_images = []
    ordered_labels = []
    
    for image, og_style, new_style in zip(images, og_styles, new_styles):
        input_label = [classdict[style] for style in new_style]
        input_label = torch.as_tensor(input_label).to(device)
        _, _, style_rep = embedder(input_label)
        encoded, _ = encoder(image, style_rep)
        decoded = decoder(encoded)
        ordered_images.append(image)
        ordered_images.append(decoded)
        ordered_labels.append(og_style[0])
        ordered_labels.append(new_style[0])


    ordered_images = torch.cat(ordered_images, dim=0)

    save_image_comparison(ordered_images, ordered_labels, ae_id, style_transfer_id=style_transfer_id)