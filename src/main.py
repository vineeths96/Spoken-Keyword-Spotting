import os
from model_train import model_train, marvin_kws_model
from model_test import marvin_model_test


def main():
    trained = os.path.isfile("../models/marvin_kws_svm.pickle") and os.path.isfile("../models/marvin_kws_pca.pickle")

    if not trained:
        print("Training model")
        model_train()
        marvin_kws_model()
    else:
        print("Testing model")
        marvin_model_test()


if __name__ == "__main__":
    main()
