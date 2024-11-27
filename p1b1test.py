import sys
import os
import torch
from torch.utils.data import DataLoader
import tqdm
from wikiart import WikiArtDataset, WikiArtModel, set_seed
import torcheval.metrics as metrics
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config.json")
args = parser.parse_args()
config = json.load(open(args.config))
parentdir = config["parentdir"]
testingdir = config["testingdir"]
testsampling = config["testsampling"]
device = config["device"]
testingdir = os.path.join(parentdir, config["testingdir"])
seed = config["seed"]
set_seed(seed)

testset = WikiArtDataset(testingdir, resampling=testsampling, device=device)

def test(testset, classdict, modelfile=None, inputmodel=None, device="cpu", filelog=False):
    """
    Tests classification accuracy of WikiArtModel

    Args:
        testset: dataset of class WikiArtDataset
        classdict: dictionary from input strings to corresponding integer index
        modelfile: if provided, loads trained model from this location
        inputmodel: if provided, uses this model for evaluation
        (if both modelfile and inputmodel are provided, uses loaded model)
        filelog:
            if false-like: prints evaluation
            if truthy: instead logs evaluation to ./results/b1p1
    """

    loader = DataLoader(testset, batch_size=1)
    
    if modelfile: 
        model = WikiArtModel(len(classdict))
        model.load_state_dict(torch.load(modelfile, weights_only=True))
        #model = torch.load(modelfile, weights_only=False)
    elif inputmodel:
        model = inputmodel
    else:
        print('No model given for evaluation')
        return None
    
    model = model.to(device)
    model.eval()

    predictions = []
    truth = []
    for batch_id, batch in enumerate(tqdm.tqdm(loader)):
        X, y = batch
        y = [classdict[label] for label in y]
        y = torch.as_tensor(y).to(device)
        #y = y.to(device)
        output = model(X)
        predictions.append(torch.argmax(output).unsqueeze(dim=0))
        truth.append(y)
    

    #print("predictions {}".format(predictions))
    #print("truth {}".format(truth))
    predictions = torch.concat(predictions)
    truth = torch.concat(truth)

    metric = metrics.MulticlassAccuracy()
    metric.update(predictions, truth)
    #print('Testing loaded model:')
    if filelog:
        log = open(filelog, 'a')
        sys.stdout = log
    
    print("Accuracy: {}".format(metric.compute()))

    if filelog:
        print()
        log.close()
        sys.stdout = sys.__stdout__

    # took this out from the original script since there in my opinion are too many classes for a confusion matrix
    # to be very sensible:
    #confusion = metrics.MulticlassConfusionMatrix(27)
    #confusion.update(predictions, truth)
    #print("Confusion Matrix\n{}".format(confusion.compute()))
