# ML2a2
Andreas Johansson submission for LT2326, assignment 2

General note: I generate files/directories in several places, and I use a rudimentary system to deal with multiple of the same kinds of files/dirs in the same location. The scripts will generate e.g. "comparison1", then "comparison2", etc, or "ae1", "ae2" etc. The system checks for example how many "comparison" files are at the location, and then tries to create comparison+str(comparison count+1). So if a dir has files "comparison1", "comparison2", and comparison 1 is deleted, comparison 2 must be renamed to comparison1, or the system would try to generate comparison2, which would already exist.

Parts:

All parts use wikiart.py

Bonus A and part 1
These parts are combined and use the following files
 - p1b1main
 - p1b1train
 - p1b1test
 - config

Only p1b1main needs to ever be run (without arguments) and only config needs to be altered
Config parameters (I will skip obvious ones, like hyperparams):
- modelfile, parentdir, trainingdir, testingdir
  - these only exist as a leftover from Asad's original script. They should not be altered and should probably be removed and baked into the scripts.
-seed
  - mostly obvious, but I made a function that sets random, np, and torch seeds in one go, altered with this config
- train/testsampling
  - explained in more detail in code documentation, but can be set to the following values:
   - "mean": counts for all classes will be set to the mean number of examples per class
   - "max": all classcounts will be set to the count of the most frequent class
   - "min": reverse of max, least frequent class
   - float between 0 and 1 noninclusive: counts will be set to this proportion of most frequent class (e.g. .5 = max/2
   - float or int 1 or higher: counts will be set to this number
   -  any non-truthy value: no dataset change
  - any oversampled class does so via duplication until the target count is achieved. No image will be duplicated twice until all images have been duplicated once, and none three times before all have been duplicated twice, etc.
  - undersampling is done via random removal of images
- filelog: if truthy, evaluation (accuracy testing, alongside the current config data) is logged to file instead of printed. Logged in ./results/b1p1

p1b1main trains the model per config parameters, evaluates it every epoch (also stored in the log if filelog), then does a final evaluation. If there is any rebalancing, WikiArtDataset will print the changed data counts.

Bonus A:
The low accuracy in class was not strictly due to the model; it seems to have been due to the classes indices not being consistent across training and testing. I fixed this with a label-to-index dictionary. See results log 1 (10 epochs, no rebalancing, fluctuates around ~17%). The model still isn't great (it is not quite but close to a most-likely-class predictor), so it still is not learning the desired function. More training does not seem to help (at least up to 20 epochs). The dip to 14% at the end of log 1 does not seem to be a local minumum followed by a better fit; see log 2, with the same parameters and a similar dip around ep10, but which still never goes much above 20. I played around with some different model parameters, but nothing did much, and the ~17 is still well above the 5% threshold.

Part 1
Mostly documented in the above config description (sampling parameters), specific implementation in the code. Overall, the rebalancing does not help the model make better predictions, at least as far as accuracy is concerned. Logs 3-5 ('trainsampling': 500, 'testsampling': False), ('trainsampling': 'mean', 'testsampling': 'mean') and ('trainsampling': 'mean', 'testsampling': False), respectively, are some examples of different rebalancing strategies. If the model were actually able to learn the proper functions to properly classify the data, rebalancing would certainly help. Given that the model still is not able to do this, it makes sense that resampling strategies actually hurt the model, as is the case here, since it might mess up spurious correlations the model actually use to classify the data instead of the proper function. Log 6 ('trainsampling': '500', 'testsampling': '500') is entirely balanced across classes (as far as count at least), and the model seems to be learning at least a little bit in spite of not just being able to rely on most frequent (ends at ~19% after 15 epochs, without as much fluctuations as before).


Both parts 2 and 3 use config2
Part 2 uses
- p2main
- p2train
- p2test
Only p2main needs to be run (without arguments), with configurations set by config2.
General note on the autoencoder: it really should have had a pooling layer (the flattened linear sizes get very big, and cuda memory is a problem). By the time I figured out I should implement this it was a bit of a hassle, and since the autoencoder still works (if only runnable at resized image sizes), I did not end up implementing this. The autoencoder produces relatively fine reconstructions, if quite blurry. See for example any comparison png in ae1 or ae2 (note: look at those directly in these directories, not in a transfermodel directory). The model 













