# ML2a2
Andreas Johansson submission for LT2326, assignment 2

See very bottom for simple running instructions.

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

Only p2main needs to be run (without arguments), with configurations set by config2 (see bottom)
General note on the autoencoder: it really should have had a pooling layer (the flattened linear sizes get very big, and cuda memory is a problem). By the time I figured out I should implement this it was a bit of a hassle, and since the autoencoder still works (if only runnable at resized image sizes and relatively small batch sizes.), I did not end up implementing this. The autoencoder produces relatively fine reconstructions, if quite blurry. See for example any comparison png in ae1 or ae2 (note: look at those directly in these directories, not in a transfermodel directory) Any autoencoder folder also has hyperparameter/metadata (checkpointdata.txt) . A bit more greedy hyperparameters (though not necessarily many more epochs) seemed to increase performance, so there is some room to grow. The latent space does not cluster very well across the artstyles (see ./autoencoders/ae2/pcaclass1.png, which is shows the latent representations reduced by PCA. There is no label legend since there are too many classes, but each colour is a label). This aligns with the difficulty of classifying the classes found in part 1; it might be that whatever commonalities exist across classes is very hard to capture in this problem, compared to, for example, OCR. Of course, that does not mean the problem is insolvable and that there is no pattern, but that the part 1 classification model and the part 2 latent reprersentations are not sophisticated enough to capture it. Compare the mentioned PCA plot with one clustered by kmeans (same directory, pca_kmeans1.png), which shows that a pattern is picked up by this clustering algorithm, but it is not one corresponding to our classes.

Part 3 uses
- p3main
- p3funcs

Only p3main needs to be run, with configurations from config2.
The style embedding transfer implemented does not work very well, probably because the implementation could be better (I could not find someone implementing what we did; only regular style transfer, not transfer of a style embedding), but also because, as we have seen, the artstyle features are not easy to get to. In short this is the implementation: train an Embedder (a version of the encoder part of the autoencoder that does not see an input image, only a class label); train this with a part 2 decoder against an image (i.e., loss function takes the output of Embedder --> Decoder --> against the original image); hope SOMETHING relating to the style was captured; train the embedder and both parts of the autoencoder (encoder, decoder). Instead of just getting loss(image, image), use style loss and content loss, weighted by hyperparameters style_weight, content_weight. Style loss is taken to be MSE loss from the gram matrix of the output of pre-specified layers (another hyperparameter). The idea is that the gram matrix (which in our case is applied to outputs of conv layers) enhances the features that co-occur across the piece without focusing on the individual features of an image. Content loss is (possibly naively) taken to be very similar to the loss we used in p2, though it might be a good idea to also look at specific layers for this. For example, it might be reasonable to assume that early layers have more concrete features and later layers have more abstract (style) features.* I was not sure how to combine the class style with the content image and settled relatively arbitrarily on just concatenating the flattened representation after the final conv layer in the embedder and encoder, before combining them in the latent representation. The decoder was not altered for this, but it seems likely it would have been better off with some alterations to accommodate what is going on in the embedder/encoder. The final product is simply a bit more blurry version of what the p2 autoencoder already output. See ./autoencoders/ae2/transfermodel1/comparison1.png for an example comparison, and ./autoencoders/ae2/transfermodel2/comparison2.png for different styles being applied to the same image, showing that the model does not really differentiate between different styles.

*Some sources for this implementation:
https://github.com/dxyang/StyleTransfer/blob/master/style.py
https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
https://www.youtube.com/watch?v=AJOyMJjPDtE
(though none did exactly what we did)

config2 explanation:
(skipping obvious/things already covered for the first config)
#Can be disregarded/does not have to be changed, see below
toy_sample: makes everything in parts 2 and 3 run on a toy amount of data
train_new: if truthy, trains new autoencoder model, saves in ./autoencoders/aeI, where if ae1 and ae2 already exists, I = 3 (CAN possibly save another ae, see code documentation)
per_epoch_comparison: if truthy, creates and saves images in ./looseimages directory every epoch of training, for development comparison. Directory is removed by running clearim.py
compare: saves image comparisons either in selected ae directory (see below) if not training new model, or in newly model directory (if training)
plot_latent: plots pca and kmeans-clustered pca (in selected ae dir or newly trained ae dir)
load_ae_id: selects autoencoder do any of the above (if not training new) or to use for style transfer
new_embedder: if truthy, trains new embedder for style transfer, saves alongside the autoencoder that is used to train it
train_transfer: train new transfer model. Saves in ./autoencoders/aeI/transfermodelJ, where I = loaded ae_id and if transfermodel1 and transfermodel2 exists in selected ae directory, J = 3
transfer_id: selects transfermodel. If load_ae_id=2 and transfer_id=1, selected model is ./autoencoders/ae2/transfermodel1. Not used if new transfermodel is trained
single_image_comparison: transfers different styles to one image, instead of using different content images
content_layers: (not used)
style_layers: layers used for style loss
style_ and content_weight (factors for their respective loss)
epochs: refers to initial autoencoder epochs (cf. embedder_epochs, transfer_epochs
batch_size, learning_rate: used for all trainings in p2 and p3
latent_dim_specifier: latent dim size
latent_dim_manual: matters, but just leave as true, affects how previous line is interpreted
img_size_modifier": factor for image size. Don't change, as might otherwise lead to cuda memory issues, and is also not robust to some errors due to size compatability


Simple running instructions:

If run as of submission:
Run p1b1main.py
- will train a part 1 model on only two epochs (but full amounts of data), relatively quick
- will test the model every epoch and once at the end
- will print (not log) results
Run p2main.py:
- Will train autoencoder on a toy sample of data for two epochs (including a validation step per epoch, so 4 total progress bars
- Will create ./autoencoders/ae3/, and inside this dir: checkpointdata (metadata), comparison1, pca_class1, pca_kmeans1, WikiArtAE (model itself)
Run p3main.py
- Will train an embedder using ae3 and save it in the WikiArtAE created in p2.
- Will create ./autoencoders/ae3/transfermodel1, and inside this dir: comparison1 (10 images to the left and each image's style transfered reconstruction to the right, will be very bad with the toy sample), metadata, StyleEmbeddingTransfer (model: an embedder, encoder, decoder)

Further runs of p2 without changing config will create ae4 etc., but further runs of p3 will still select ae3. Note that models are not included in the repository.











