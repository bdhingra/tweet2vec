Preprocessing
=======================

1. If you have twitter data in the format provided by their API (one json object per line), you can use `twitterFilter.jar` to extract tweet text and hashtags. This will also perform the required preprocessing - remove HTML tags, and replace usernames and URLs with special symbols, and remove re-tweets. Note that non-english posts and posts without hashtags will be discarded.

Usage:
```
java -jar twitterFilter.jar in_folder out_file
```

`in_folder` contains the raw json files. `out_file` will contain the tweets one per line and their associated hashtags.

2. If you have a bunch of tweets already extracted to a text file, and would only like to process the text, use `preprocess.py`. Note that this uses a better tokenization than the one used in the paper.

Usage:
```
python preprocess.py in_file out_file
```

`in_file` contains the posts one per line. `out_file' will contain the processed version of these, one per line.
