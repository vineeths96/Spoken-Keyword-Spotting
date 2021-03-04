NUM_CLASSES = 31

AUDIO_SR = 16000
AUDIO_LENGTH = 16000
LIBROSA_AUDIO_LENGTH = 22050

EPOCHS = 25

categories = {
    "stop": 0,
    "nine": 1,
    "off": 2,
    "four": 3,
    "right": 4,
    "eight": 5,
    "one": 6,
    "bird": 7,
    "dog": 8,
    "no": 9,
    "on": 10,
    "seven": 11,
    "cat": 12,
    "left": 13,
    "three": 14,
    "tree": 15,
    "bed": 16,
    "zero": 17,
    "happy": 18,
    "sheila": 19,
    "five": 20,
    "down": 21,
    "marvin": 22,
    "six": 23,
    "up": 24,
    "wow": 25,
    "house": 26,
    "go": 27,
    "yes": 28,
    "two": 29,
    "_background_noise_": 30,
}


inv_categories = {
    0: "stop",
    1: "nine",
    2: "off",
    3: "four",
    4: "right",
    5: "eight",
    6: "one",
    7: "bird",
    8: "dog",
    9: "no",
    10: "on",
    11: "seven",
    12: "cat",
    13: "left",
    14: "three",
    15: "tree",
    16: "bed",
    17: "zero",
    18: "happy",
    19: "sheila",
    20: "five",
    21: "down",
    22: "marvin",
    23: "six",
    24: "up",
    25: "wow",
    26: "house",
    27: "go",
    28: "yes",
    29: "two",
    30: "_background_noise_",
}

# Marvin model
INPUT_SHAPE = (99, 40)
TARGET_SHAPE = (99, 40, 1)
PARSE_PARAMS = (0.025, 0.01, 40)
filters = [16, 32, 64, 128, 256]

DROPOUT = 0.25
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
DENSE_1 = 512
DENSE_2 = 256

BATCH_SIZE = 128
PATIENCE = 5
LEARNING_RATE = 0.001
