import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt

from parameters import *
from utils import OC_Statistics
from models import create_model
from utils import getDataset, plot_history
from get_data import downloadData, getDataDict, getDataframe


def model_train():
    """
    Trains model which is used as a feature extractor
    :return:None
    """

    # Download data
    downloadData(data_path="/input/speech_commands/")

    # Get data dictionary
    dataDict = getDataDict(data_path="/input/speech_commands/")

    # Obtain dataframe for each dataset
    trainDF = getDataframe(dataDict["train"])
    valDF = getDataframe(dataDict["val"])
    devDF = getDataframe(dataDict["dev"])
    testDF = getDataframe(dataDict["test"])

    print("Dataset statistics")
    print("Train files: {}".format(trainDF.shape[0]))
    print("Validation files: {}".format(valDF.shape[0]))
    print("Dev test files: {}".format(devDF.shape[0]))
    print("Test files: {}".format(testDF.shape[0]))

    # Use TF Data API for efficient data input
    train_data, train_steps = getDataset(df=trainDF, batch_size=BATCH_SIZE, cache_file="train_cache", shuffle=True)

    val_data, val_steps = getDataset(df=valDF, batch_size=BATCH_SIZE, cache_file="val_cache", shuffle=False)

    model = create_model()
    model.summary()

    # Stop training if the validation accuracy doesn't improve
    earlyStopping = EarlyStopping(monitor="val_loss", patience=PATIENCE, verbose=1)

    # Reduce LR on validation loss plateau
    reduceLR = ReduceLROnPlateau(monitor="val_loss", patience=PATIENCE, verbose=1)

    # Compile the model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["sparse_categorical_accuracy"],
    )

    # Train the model
    history = model.fit(
        train_data.repeat(),
        steps_per_epoch=train_steps,
        validation_data=val_data.repeat(),
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=[earlyStopping, reduceLR],
    )

    # Save model
    print("Saving model")
    model.save("../models/marvin_kws.h5")

    # Save history data
    print("Saving training history")
    with open("../models/marvin_kws_history.pickle", "wb") as file:
        pickle.dump(history.history, file, protocol=pickle.HIGHEST_PROTOCOL)

    plot_history(history=history)


def marvin_kws_model():
    """
    Trains an One Class SVM for hotword detection
    :return: None
    """

    # Download data
    downloadData(data_path="/input/speech_commands/")

    # Get dictionary with files and labels
    dataDict = getDataDict(data_path="/input/speech_commands/")

    # Obtain dataframe for each dataset
    trainDF = getDataframe(dataDict["train"])
    valDF = getDataframe(dataDict["val"])

    # Obtain Marvin data from training data
    marvin_data, _ = getDataset(
        df=trainDF.loc[trainDF["category"] == "marvin", :],
        batch_size=BATCH_SIZE,
        cache_file="kws_marvin_cache",
        shuffle=False,
    )

    # Obtain Marvin - Other separated data from validation data
    valDF["class"] = valDF.apply(lambda row: 1 if row["category"] == "marvin" else -1, axis=1)
    valDF.drop("category", axis=1)
    val_true_labels = valDF["class"].tolist()

    val_data, _ = getDataset(df=valDF, batch_size=BATCH_SIZE, cache_file="kws_val_cache", shuffle=False)

    # Load model and create feature extractor
    model = load_model("../models/marvin_kws.h5")

    layer_name = "features256"
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    # Obtain the feature embeddings
    X_train = feature_extractor.predict(marvin_data, use_multiprocessing=True)
    X_val = feature_extractor.predict(val_data, use_multiprocessing=True)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=32)
    pca.fit(X_train)
    print("Variance captured = ", sum(pca.explained_variance_ratio_))

    X_train_transformed = pca.transform(X_train)
    X_val_transformed = pca.transform(X_val)

    # SVM hyper-parameter tuning using Gaussian process
    marvin_svm = svm.OneClassSVM()

    svm_space = [
        Real(10 ** -5, 10 ** 0, "log-uniform", name="gamma"),
        Real(10 ** -5, 10 ** 0, "log-uniform", name="nu"),
    ]

    @use_named_args(svm_space)
    def svm_objective(**params):
        marvin_svm.set_params(**params)

        marvin_svm.fit(X_train_transformed)
        val_pred_labels = marvin_svm.predict(X_val_transformed)

        score = f1_score(val_pred_labels, val_true_labels)

        return -1 * score

    res_gp_svm = gp_minimize(
        func=svm_objective, dimensions=svm_space, n_calls=100, n_jobs=-1, verbose=False, random_state=1
    )

    print("Best F1 score={:.4f}".format(-res_gp_svm.fun))

    ax = plot_convergence(res_gp_svm)
    plt.savefig("../docs/results/marvin_svm.png", dpi=300)
    plt.show()

    # Instantiate a SVM with the optimal hyper-parameters
    best_params_svm = {k.name: x for k, x in zip(svm_space, res_gp_svm.x)}
    marvin_kws = svm.OneClassSVM()
    marvin_kws.set_params(**best_params_svm)

    marvin_kws.fit(X_train_transformed)

    # Performance on training set
    val_pred_labels = marvin_kws.predict(X_val_transformed)
    OC_Statistics(val_pred_labels, val_true_labels, "marvin_cm_training")

    print("Saving PCA object")
    with open("../models/marvin_kws_pca.pickle", "wb") as file:
        pickle.dump(pca, file, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saving Marvin SVM")
    with open("../models/marvin_kws_svm.pickle", "wb") as file:
        pickle.dump(marvin_svm, file, protocol=pickle.HIGHEST_PROTOCOL)
