Prerequisites
======================
- Python 2.7
- GuineaPig - http://curtis.ml.cmu.edu/w/courses/index.php/Guinea_Pig (latest)
- Theano and all dependencies (latest)
- Lasagne 0.1

Data and Preprocessing
=======================
1. Start with files containing tweets in json format. Ex:

2. Preprocessing - 
```sh
    java -jar twitterFilter.jar input_folder tweet_file_int
    python clean_data.py tweet_file_int tweet_file
```
input_folder contains the raw tweet files, tweet_file will be one file containing processed tweets in the following format - hashtag_1>,hashtag_2,... \t tweet_text\n

3. Create dictionary of hashtags by - 
```sh
    python hash_dict_gpig.py --params input:tweet_file --store hashdict
```
A folder gpig_views will be created with the file hashdict.gp

4. Run - 
```sh
    python select_hashtags.py gpig_views/hashdict.gp MIN_COUNT MAX_COUNT out_path
```
This will filter all hashtags with less than MIN_COUNT and more than MAX_COUNT and store the output in out_path/data.txt

5. Combine same tweets with different hashtags - 
```sh
    python combine_tags.py --params input:out_path/data.txt --store combined
```
The final dataset will be stored in gpig_views/combined.gp

6. Split the above into training, testing and validation files

7. Finally, split the hashtags again for training and validation files
```sh
    python hash_dict_gpig.py --params input:train_file --store splittags
```
This will store the training file ready for use in gpig_views/splittags.gp. Do the same for validation file

Training
========================
For each model enter the appropriate settings in settings.py.

- Word model:
```sh
    python char_word.py training_file validation_file model_save_path
```
- 1 Layer model:
```sh
    python char.py training_file validation_file model_save_path
```
- 2 Layer model:
```sh
    python char_c2w2s.py training_file validation_file model_save_path
```

Training will be performed for the number of epochs specified in settings.py. Choose the best model by tracking the validation cost output on the screen. You can redirect the outputs to a file and run the following to see training and validation errors after each epoch:
```sh
    grep Training < log_file
```

Testing
========================
First, generate predictions over the test set (make sure the settings match training when running the following) -
- Word model:
```sh
    python test_word.py test_file model_save_path result_save_path epoch_num
```
Epoch number denotes the model after the specific training epoch that you want to generate predictions from. Omit this to test on last saved model.
- 1 Layer model:
```sh
    python test.py test_file model_save_path result_save_path epoch_num
```
- 2 Layer model:
```sh
    python test_c2w2s.py test_file model_save_path result_save_path epoch_num
```

Next, run the evaluation script to generate performance metrics - 
```sh
    python eval.py result_save_path model_save_path
```

Contact
==========================
Report bugs and missing info to bdhingraATandrewDOTcmuDOTedu (replace AT, DOT appropriately).
