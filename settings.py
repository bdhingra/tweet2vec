# Define all global params here

# Set true for word-level model, false for character-level
WORD_LEVEL = True
# Number of epochs
NUM_EPOCHS = 30
# Batch size
N_BATCH = 64
# Max sequence length
MAX_LENGTH = 20
# Number of unique characters
N_CHAR = 20000
# Dimensionality of character lookup
CHAR_DIM = 150
# Initialization scale
SCALE = 0.1
# Dimensionality of C2W hidden states
C2W_HDIM = 500
# Dimensionality of word vectors
WDIM = 500
# Number of classes
MAX_CLASSES = 6000
# Learning rate
LEARNING_RATE = .01
# Display frequency
DISPF = 5
# Save frequency
SAVEF = 1000
# Validation set
N_VAL = 512
# Regularization
REGULARIZATION = 0.0001
# Reload
RELOAD_DATA = False
RELOAD_MODEL = False
# Maximum word length
MAX_WORD_LENGTH = 8
# Maximum sequence length
MAX_SEQ_LENGTH = 20
# Sequence dimension
SDIM = 150
# 2nd layer hidden dim
W2S_HDIM = 150
# Debugging
DEBUG=False
# NAG
MOMENTUM = 0.9
# clipping
GRAD_CLIP = 5.
