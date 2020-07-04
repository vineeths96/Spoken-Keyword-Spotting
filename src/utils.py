import os
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score, confusion_matrix
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from python_speech_features import logfbank
from src.parameters import *


def getDataset(df,
               batch_size,
               cache_file=None,
               shuffle=True,
               parse_param=(0.025, 0.01, 40),
               scale=False):
    """
    Creates a Tensorflow Dataset containing filterbanks, labels
    :param df: Dataframe with filenames and labels
    :param batch_size: Batch size of the input
    :param cache_file: Whether to cache the dataset during run
    :param shuffle: Whether to shuffle the dataset
    :param parse_param: Window parameters
    :param scale: Whether to scale filterbank levels
    :return: TF Dataset, Steps per epoch
    """

    data = tf.data.Dataset.from_tensor_slices(
        (df['files'].tolist(), df['labels'].tolist())
    )

    data = data.map(
        lambda filename, label: tuple(
            tf.py_function(
                _parse_fn,
                inp=[filename, label, parse_param, scale],
                Tout=[tf.float32, tf.int32]
                )
        ),
        num_parallel_calls=os.cpu_count()
    )

    if cache_file:
        data = data.cache('../input/' + cache_file)

    if shuffle:
        data = data.shuffle(buffer_size=df.shape[0])

    data = data.batch(batch_size).prefetch(buffer_size=1)
    steps = df.shape[0] // batch_size

    return data, steps


def _loadfile(filename):
    """
    Return a np array containing the wav
    :param filename: Filename of wav
    """

    _, wave = wavfile.read(filename)

    # Pad with noise if audio is short
    if len(wave) < AUDIO_LENGTH:
        silence_part = np.random.normal(0, 5, AUDIO_LENGTH - len(wave))
        wave = np.append(np.asarray(wave), silence_part)

    return np.array(wave, dtype=np.float32)


def _logMelFilterbank(wave,
                      parse_param=(0.025, 0.01, 40)):
    """
    Computes the log Mel filterbanks
    :param wave: Audio as an array
    :param parse_param: Window Parameter
    :return: Filterbanks
    """

    fbank = logfbank(
        wave,
        samplerate=AUDIO_SR,
        winlen=float(parse_param[0]),
        winstep=float(parse_param[1]),
        highfreq=AUDIO_SR / 2,
        nfilt=int(parse_param[2])
        )

    fbank = np.array(fbank, dtype=np.float32)

    return fbank


def _normalize(data):
    """
    Normalizes the data (z-score)
    :param data: Data to be normalized
    :return: Nomralized data
    """

    mean = np.mean(data, axis=0)
    sd = np.std(data, axis=0)

    # If Std Dev is 0
    if not sd:
        sd = 1e-7

    return (data - mean) / sd


def _parse_fn(filename,
              label,
              parse_param=(0.025, 0.01, 40),
              scale=False):
    """
    Calculates filterbank energies for a given file
    :param filename: File name
    :param label: Class label
    :param parse_param: Window parameters
    :param scale: Whether to normalize the filterbanks
    :return: Filterbanks, Label
    """

    wave = _loadfile(filename.numpy())
    fbank = _logMelFilterbank(wave, parse_param)

    if scale:
        fbank = _normalize(fbank)

    return fbank, np.asarray(label, dtype=np.int32)


def plot_confusion_matrix(y_pred, y_true):
    """
    Plots the confusion matrix for given data
    :param y_pred: Predicted labels
    :param y_true: True labels
    :return: None
    """

    cm = confusion_matrix(y_pred, y_true, labels=[0, 1])
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=['Other', 'Marvin']).plot()

    plt.show()


def OC_Statistics(y_pred,
                  y_true):
    """
    Print performance statistics for One Class problem
    :param y_pred: Predicted labels
    :param y_true: True labels
    :return: None
    """

    print("Accuracy: {:.4f}".format(accuracy_score(y_true, y_pred)))
    print("Precision: {:.4f}".format(precision_score(y_true, y_pred)))
    print("Recall: {:.4f}".format(recall_score(y_true, y_pred)))
    print("F1-score: {:.4f}".format(f1_score(y_true, y_pred)))

    plot_confusion_matrix(y_pred, y_true)
