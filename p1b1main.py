import sys
import os
import torchvision.transforms.functional as F
from wikiart import WikiArtDataset, WikiArtModel, class_to_idx, set_seed
from p1b1train import train
from p1b1test import test
import json
import argparse


"""
See readme for config explanation and running instructions
"""

def main():
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
    filelog=config["filelog"]

    if filelog:
        filelog='./results/b1p1/results_log'+str(len(os.listdir('./results/b1p1'))+1)+'.txt'
        print('Logging results to file instead of printing: see most recent results log')
        log = open(filelog, 'a')
        sys.stdout = log

    print('Config:')
    print(config)
    print()

    if filelog:
        log.close()
        sys.stdout = sys.__stdout__

    log = open(filelog, 'a')
    

    traindataset = WikiArtDataset(trainingdir, resampling=trainsampling, device=device)    
    testdataset = WikiArtDataset(testingdir, resampling=testsampling, device=device)
    trainclasses = traindataset.classes
    testclasses = testdataset.classes
    classes = list(trainclasses.union(testclasses))
    classdict = class_to_idx(classes)

    model = train(traindataset, classdict, testset_for_epoch_eval=testdataset, epochs=config["epochs"],
                  batch_size=config["batch_size"], modelfile=config["modelfile"], device=device, filelog=filelog)
    
    #print('Testing returned model:')
    #test(testdataset, classdict, inputmodel=model, device=device, filelog=filelog)
    #print()

    if filelog:
        log = open(filelog, 'a')
        sys.stdout = log
        print('Final accuracy:')
        log.close()
        sys.stdout = sys.__stdout__

    print('Testing loaded model:')
    test(testdataset, classdict, modelfile=config["modelfile"], device=device, filelog=filelog)

if __name__ == "__main__":
    main()
    
