# ML2a2
Andreas Johansson submission for LT2326, assignment 2

General note: I generate files/directories in several places, and I use a rudimentary system to deal with multiple of the same kinds of files/dirs in the same location. The scripts will generate e.g. "comparison1", then "comparison2", etc, or "ae1", "ae2" etc. The system checks for example how many "comparison" files are at the location, and then tries to create comparison+str(comparison count+1). So if a dir has files "comparison1", "comparison2", and comparison 1 is deleted, comparison 2 must be renamed to comparison1, or the system would try to generate comparison2, which would already exist.

Running instructions:

All parts use wikiart.py

Bonus A and part 1
These parts are combined and use the following files
 - p1b1main.py
 - p1b1train.py
 - p1b1test.py
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
  - any oversampled class does so via duplication until the target count is achieved. No image will be duplicated twice until all images have been duplicated once.
  - undersampling is done via random removal of images
- filelog: if truthy, evaluation (accuracy testing, alongside the current config data) is logged to file instead of printed. Logged in ./results/b1p1

- 
