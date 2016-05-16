Tweet2Vec
======================
This repository provides a character-level encoder/trainer for social media posts. See [Tweet2Vec](https://arxiv.org/abs/1605.03481) paper for details.

There are two models implemented in the paper - the character level _tweet2vec_ and a word level baseline. They can be found in their respective directories, with instructions on how to run. General information about prerequisites and data format can be found below.

Prerequisites
======================
- Python 2.7
- Theano and all dependencies (latest)
- Lasagne (latest)
- Numpy
- Maybe more, just use `pip install` if you get an error


Data and Preprocessing
=======================
Unfortunately we are not allowed to release the data used in experiments from the paper, due to licensing restrictions. Hence, we describe the data format and preprocessing here -

1. __Preprocessing__ - We replace HTML tags, usernames, and URLs from tweet text with special tokens. Hashtags are also removed from the body of a tweet, and re-tweets are discarded. Example code is provided in `misc/preprocess.py`. 

2. __Encoding File Format__ - If you have a bunch of posts that you want to embed into a vector space, use the `_encoder.sh` scripts provided. The input file must contain one tweet per line (make sure you preprocess these first). An example is provided in `misc/encoder_example.txt`.

3. __Training File Format__ - To train the models from scratch, use the `_trainer.sh` scripts provided. The input file must contain one _(hashtag,tweet)_ pair per line separated by a tab. There should be only one tag per line - for tweets with multiple tags split them into separate line. See `misc/trainer_example.txt` for an example. 

4. __Test/Validation File Format__ - After training the model, you can test it on a held-out set using `_tester.sh` scripts provided. It has the same format as the training file format, except it can have multiple tags per separated by a comma. Example in `misc/tester_example.txt`.

Note
==========================
Make sure to add `THEANO_FLAGS=device=cpu` before any command if you are running on a CPU.

Contributors
==========================
Bhuwan Dhingra, Dylan Fitzpatrick, Zhong Zhou, Michael Muehl. Special thanks to Yun Fu for the preprocessing JAR-file.

Report bugs and missing info to bdhingraATandrewDOTcmuDOTedu (replace AT, DOT appropriately).

License
==========================
[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

<!---
1. Start with files containing tweets in json format. Ex:

2. Preprocessing - 
    ```sh
        java -jar twitterFilter.jar input_folder tweet_file_int
        python clean_data.py tweet_file_int tweet_file
    ```
    input_folder contains the raw tweet files, tweet_file will be one file containing processed tweets in the following format - 
    ```
        hashtag_1>,hashtag_2,... \t tweet_text\n
    ```

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
    ```
        python char_word.py training_file validation_file model_save_path
    ```
- 1 Layer model:
    ```
        python char.py training_file validation_file model_save_path
    ```
- 2 Layer model:
    ```
        python char_c2w2s.py training_file validation_file model_save_path
    ```

Training will be performed for the number of epochs specified in settings.py. Choose the best model by tracking the validation cost output on the screen. You can redirect the outputs to a file and run the following to see training and validation errors after each epoch:
    ```
        grep Training < log_file
    ```

Testing
========================
First, generate predictions over the test set (make sure the settings match training when running the following) -
- Word model:
    ```
        python test_word.py test_file model_save_path result_save_path epoch_num
    ```
    Epoch number denotes the model after the specific training epoch that you want to generate predictions from. Omit this to test on last saved model.
- 1 Layer model:
    ```
        python test.py test_file model_save_path result_save_path epoch_num
    ```
- 2 Layer model:
    ```
        python test_c2w2s.py test_file model_save_path result_save_path epoch_num
    ```

Next, run the evaluation script to generate performance metrics - 
    ```
        python eval.py result_save_path model_save_path
    ```
-->
