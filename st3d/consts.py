import torch

NUM_TRAINING_SCENES = 500
NUM_VALIDATION_SCENES = 25
NUM_OBJS_IN_SCENE = 10
NUM_POINTS_IN_CLOUD = N = 16000

NUM_NEIGHBORS = K = 8
DEPTH = D = 64
NUM_RESIDUALS = 4

HIDDEN_LAYER_D = 128
NUM_DECODED_POINTS = 1024
MLP_LEAKY_SLOPE = 0.2
HIDDEN_LAYER_LEAKY_SLOPE = 0.05

DECODER_SAMPLE_SIZE = 16

NUM_PRETRAINING_EPOCHS = 250
PRETRAINING_LR = 1e-3
PRETRAINING_WD = 1e-6

NUM_TRAINING_EPOCHS = 100
TRAINING_LR = 1e-3
TRAINING_WD = 1e-5

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'