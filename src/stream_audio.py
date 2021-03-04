import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.models import Model, load_model

import pickle
import pyaudio
import numpy as np
from queue import Queue
import matplotlib.pyplot as plt
from python_speech_features import logfbank


class StreamPrediction:
    """
    Class for predicting streaming data. Heavily adapted from the implementation:
    """

    def __init__(self, model_path):
        # Load model
        self.feature_extractor = None
        self.pca = None
        self.marvin_svm = None
        self.load_models(model_path)

        # Recording parameters
        self.sr = 16000
        self.chunk_duration = 0.75
        self.chunk_samples = int(self.sr * self.chunk_duration)
        self.window_duration = 1
        self.window_samples = int(self.sr * self.window_duration)
        self.silence_threshold = 100

        # Data structures and buffers
        self.queue = Queue()
        self.data = np.zeros(self.window_samples, dtype="int16")

        # Plotting parameters
        self.change_bkg_frames = 2
        self.change_bkg_counter = 0
        self.change_bkg = False

    def load_models(self, model_path):
        """
        Loads the models for hotword detection
        :param model_path: Path to model directory
        :return: None
        """

        # Load model structure
        model = load_model(model_path + "/marvin_kws.h5")

        layer_name = "features256"
        self.feature_extractor = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

        # Load trained PCA object
        with open(model_path + "/marvin_kws_pca.pickle", "rb") as file:
            self.pca = pickle.load(file)

        # Load trained SVM
        with open(model_path + "/marvin_kws_svm.pickle", "rb") as file:
            self.marvin_svm = pickle.load(file)

        print("Loaded models from disk")

    def start_stream(self):
        """
        Start audio data streaming from microphone
        :return: None
        """

        stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sr,
            input=True,
            frames_per_buffer=self.chunk_samples,
            input_device_index=6,
            stream_callback=self.callback,
        )

        stream.start_stream()

        try:
            while True:
                data = self.queue.get()
                fbank = logfbank(data, samplerate=self.sr, nfilt=40)
                pred = self.detect_keyword(fbank)

                self.plotter(data, fbank, pred)

                if pred == 1:
                    print("Marvin!", sep="", end="", flush=True)

        except (KeyboardInterrupt, SystemExit):
            stream.stop_stream()
            stream.close()

    def detect_keyword(self, fbank):
        """
        Detect hotword presence in current window
        :param fbank: Log Mel filterbank energies
        :return: Prediction
        """

        fbank = np.expand_dims(fbank, axis=0)
        feature_embeddings = self.feature_extractor.predict(fbank)

        feature_embeddings_scaled = self.pca.transform(feature_embeddings)
        prediction = self.marvin_svm.predict(feature_embeddings_scaled)

        return prediction

    def callback(self, in_data, frame_count, time_info, status):
        """
        Obtain the data from buffer and load it to queue
        :param in_data: Daa buffer
        :param frame_count: Frame count
        :param time_info: Time information
        :param status: Status
        :return:
        """

        data0 = np.frombuffer(in_data, dtype="int16")

        if np.abs(data0).mean() < self.silence_threshold:
            print(".", sep="", end="", flush=True)
        else:
            print("-", sep="", end="", flush=True)

        self.data = np.append(self.data, data0)

        if len(self.data) > self.window_samples:
            self.data = self.data[-self.window_samples :]
            self.queue.put(self.data)

        return in_data, pyaudio.paContinue

    def plotter(self, data, fbank, pred):
        """
        Plot waveform, filterbank energies and hotword presence
        :param data: Audio data array
        :param fbank: Log Mel filterbank energies
        :param pred: Prediction
        :return:
        """

        plt.clf()

        # Wave
        plt.subplot(311)
        plt.plot(data[-len(data) // 2 :])
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.ylabel("Amplitude")

        # Filterbank energies
        plt.subplot(312)
        plt.imshow(fbank[-fbank.shape[0] // 2 :, :].T, aspect="auto")
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().invert_yaxis()
        plt.ylim(0, 40)
        plt.ylabel("$\log \, E_{m}$")

        # Hotword detection
        plt.subplot(313)
        ax = plt.gca()

        if pred == 1:
            self.change_bkg = True

        if self.change_bkg and self.change_bkg_counter < self.change_bkg_frames:
            ax.set_facecolor("lightgreen")

            ax.text(
                x=0.5,
                y=0.5,
                s="MARVIN!",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=30,
                color="red",
                fontweight="bold",
                transform=ax.transAxes,
            )

            self.change_bkg_counter += 1
        else:
            ax.set_facecolor("salmon")
            self.change_bkg = False
            self.change_bkg_counter = 0

        plt.tight_layout()
        plt.pause(0.01)


if __name__ == "__main__":
    audio_stream = StreamPrediction("../models")
    audio_stream.start_stream()
