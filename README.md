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
Make sure to add `THEANO_FLAGS=device=cpu,floatX=float32` before any command if you are running on a CPU.

Contributors
==========================
Bhuwan Dhingra, Dylan Fitzpatrick, Zhong Zhou, Michael Muehl. Special thanks to Yun Fu for the preprocessing JAR-file.

If you end up using this code, please cite the following paper - 

Dhingra, Bhuwan, Zhong Zhou, Dylan Fitzpatrick, Michael Muehl, and William W. Cohen. "Tweet2Vec: Character-Based Distributed Representations for Social Media." ACL (2016).

```
@InProceedings{dhingra-EtAl:2016:P16-2,
  author    = {Dhingra, Bhuwan  and  Zhou, Zhong  and  Fitzpatrick, Dylan  and  Muehl, Michael  and  Cohen, William},
  title     = {Tweet2Vec: Character-Based Distributed Representations for Social Media},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  month     = {August},
  year      = {2016},
  address   = {Berlin, Germany},
  publisher = {Association for Computational Linguistics},
  pages     = {269--274},
  url       = {http://anthology.aclweb.org/P16-2044}
}
```

Report bugs and missing info to bdhingraATandrewDOTcmuDOTedu (replace AT, DOT appropriately).
