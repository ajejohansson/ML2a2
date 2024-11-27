import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import tqdm
from wikiart import WikiArtModel, set_seed
from p1b1test import test
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config.json")

args = parser.parse_args()
config = json.load(open(args.config))
parentdir = config["parentdir"]
trainingdir = os.path.join(parentdir, config["trainingdir"])
testingdir = os.path.join(parentdir, config["testingdir"])
trainsampling = config["trainsampling"]
testsampling = config["testsampling"]
device = config["device"]
seed = config["seed"]
set_seed(seed)

def train(trainset, classdict, testset_for_epoch_eval=None, filelog=False, epochs=3, batch_size=32, modelfile=None, device="cpu"):
    """
    Trains a WikiArtModel for style classification.

    Args:
        trainset: dataset of class WikiArtDataset
        classdict: dictionary from input strings to corresponding integer index
        testset_for_epoch_eval: dataset of class WikiArtDataset. If provided, will print or log (see filelog description in readme)
                                a testrun per epoch
        modelfile: if provided, saves trained model at this location
    Saves:
        trained model's state dict at modelfile location
    Returns:
        trained model

    """
    loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    model = WikiArtModel(len(classdict)).to(device)
    optimizer = Adam(model.parameters(), config["learning_rate"])
    criterion = nn.NLLLoss().to(device)

    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        accumulate_loss = 0
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            X, y = batch
            y = [classdict[label] for label in y]
            y = torch.as_tensor(y).to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            accumulate_loss += loss
            optimizer.step()

        if filelog:
            log = open(filelog, 'a')
            sys.stdout = log

        print("In epoch {}, loss = {}".format(epoch, accumulate_loss))

        if filelog:
            log.close()
            sys.stdout = sys.__stdout__
        if testset_for_epoch_eval:
            print('Testing model in epoch {}:'.format(epoch))
            test(testset_for_epoch_eval, classdict, inputmodel=model, device=device, filelog=filelog)
        model.train()
        print()

    if modelfile:
        torch.save(model.state_dict(), modelfile)
    return model