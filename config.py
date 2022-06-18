"""
MODEL_NAME = X3D OR I3D
SAMPLE_MODE = OVERSAMPLE OR CENTER_CROP

IF X3D_M:
    MODEL_NAME = X3D
    FREQUENCY = 16, 
    INPUT_CLIP_LENGTH = 16, 
    CHUNK_SIZE = 16, 
    CROP_SIZE = 256, 
    PRETRAINED_PATH = X3D_M_extract_features.pth

IF X3D_S:
    MODEL_NAME = X3D
    FREQUENCY = 13, 
    INPUT_CLIP_LENGTH = 13, 
    CHUNK_SIZE = 13, 
    CROP_SIZE = 182, 
    PRETRAINED_PATH = X3D_S_extract_features.pth

IF I3D:
    MODEL_NAME = I3D
    FREQUENCY = 16, 
    INPUT_CLIP_LENGTH = 16, 
    CHUNK_SIZE = 16, 
    CROP_SIZE = 224, 
    PRETRAINED_PATH = i3d_r50_kinetics.pth
"""

MODEL_NAME = "X3D"

# FOR X3D
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]

# FOR I3D
# mean = [114.75, 114.75, 114.75]
# std = [57.375, 57.375, 57.375]

SAMPLE_MODE = "OVERSAMPLE"

FREQUENCY = 13
BATCH_SIZE = 1
CROP_SIZE = 182
INPUT_CLIP_LENGTH = 13

CHUNK_SIZE = 13

PRETRAINED_PATH = "X3D_S_extract_features.pth"
FRAMES_DIR = "F:\\Time Measurement\\01_0138"
