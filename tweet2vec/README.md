We provide and off-the-shelf encoder trained on the _medium_ dataset described in the paper (used for generating Table 3 results), as well as training/testing code to train your own model and compute the performance metrics.

1. __Encoder__ - First preprocess your data and put it in the encoding file format described on the main page. Then specify the model path, data file and the output result path in `tweet2vec_encoder.sh` and run. Output embeddings will be stored in .npy format and predicted hashtags in .txt format.

2. __Trainer__ - Preprocess your data and put it in the training file format (for training data) and testing file format (for validation data) described on main page. Then specify their locations and path to store model in `tweet2vec_trainer.sh` and run.

3. __Tester__ - Preprocess your data and put it in testing file format described on main page. Specify its location and model path in `tweet2vec_tester.sh` and run. Precision @1, Recall @10 and Mean Rank will be printed on the screen.
