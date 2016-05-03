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
N_WORD = 20000
# Dimensionality of word lookup
WORD_DIM = 150
# Initialization scale
SCALE = 0.1
# Dimensionality of C2W hidden states
C2W_HDIM = 200
# Dimensionality of word vectors
WDIM = 200
# Number of classes
MAX_CLASSES = 6000
# Learning rate
LEARNING_RATE = .01
# Display frequency
DISPF = 5
# Save frequency
SAVEF = 1000
# Regularization
REGULARIZATION = 0.0001
# Reload
RELOAD_MODEL = False
# NAG
MOMENTUM = 0.9
# clipping
GRAD_CLIP = 5.
