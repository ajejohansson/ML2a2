# ML2a2
Andreas Johansson submission for LT2326, assignment 2

General note: I generate files/directories in several places, and I use a rudimentary system to deal with multiple of the same kinds of files/dirs in the same location. The scripts will generate e.g. "comparison1", then "comparison2", etc, or "ae1", "ae2" etc. The system checks for example how many "comparison" files are at the location, and then tries to create comparison+str(comparison count+1). So if a dir has files "comparison1", "comparison2", and comparison 1 is deleted, comparison 2 must be renamed to comparison1, or the system would try to generate comparison2, which would already exist.

Running instructions:

All parts use wikiart.py

Bonus A and part 1
These are combined and use the following files
 - p1b1main.py
 - p1b1train.py
 - p1b1test.py
 - config
Only p1b1main needs to ever be run (without arguments) and only config needs to be altered
Config parameters (I will skip obvious ones, like hyperparams):
- modelfile, parentdir, trainingdir, testingdir
  these only exist as a leftover from Asad's original script. They should not be altered and should probably be removed and baked into the scripts.
-seed
  mostly obvious, but I made a function that sets random, np, and torch seeds in one go, altered with this config
- train/testsampling
  
