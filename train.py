from math import ceil
from time import time
import os
import warnings
import argparse
import numpy as np
import mlflow as mlflow
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from urllib.parse import urlparse
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, \
    Activation, \
    Flatten, \
    Dropout, \
    Convolution2D, \
    MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, \
    ModelCheckpoint, \
    TensorBoard
from tensorflow.keras.metrics import Precision, \
    Recall, \
    AUC, \
    TruePositives, \
    TrueNegatives, \
    FalsePositives, \
    FalseNegatives

size = 200
batch_size = 32
start_run_time = time()

def createModel(filter_number, learning_rate, dropout, regularisation):
    model = Sequential()
    model.add(Convolution2D(filter_number, (3, 3), use_bias=True, activation='relu', input_shape=(size, size, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(filter_number * 2, (3, 3), use_bias=True, activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    if regularisation is not None:
        model.add(
            Convolution2D(filter_number * 4, (3, 3),
                          use_bias=True,
                          activation='relu',
                          kernel_regularizer=regularisation))
    else:
        model.add(
            Convolution2D(filter_number * 4, (3, 3), use_bias=True, activation='relu'))

    model.add(MaxPooling2D((2, 2)))
    if regularisation is not None:
        model.add(
            Convolution2D(filter_number * 8, (3, 3),
                          use_bias=True,
                          activation='relu',
                          kernel_regularizer=regularisation))
    else:
        model.add(
            Convolution2D(filter_number * 8, (3, 3), use_bias=True, activation='relu'))

    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(filter_number * 16, activation='relu'))
    if dropout:
        model.add(Dropout(dropout))
    model.add(Dense(11, activation='softmax'))

    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            AUC(),
            Precision(),
            Recall(),
            TruePositives(),
            TrueNegatives(),
            FalsePositives(),
            FalseNegatives()
        ])

    return model


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=120, type=int, help="epochs")
    parser.add_argument("--filter-number", default=32, type=int, help="filter number")
    parser.add_argument("--learning-rate", default=0.001, type=float, help="learning rate")
    parser.add_argument("--dropout", default=0, type=float, help="dropout")
    parser.add_argument('--regularization-type', default='l1', type=str, help="regularization type: l1, l2")
    parser.add_argument('--regularization-rate', default=0, type=float, help="regularization rate")
    args = parser.parse_args()
    epochs = int(args.epochs)
    filter_number = int(args.filter_number)
    learning_rate = float(args.learning_rate)
    dropout = float(args.dropout)
    regularization_type = args.regularization_type
    regularization_rate = float(args.regularization_rate)

    regularisation = None
    if regularization_rate != 0:
        if regularization_type == 'l1':
            regularisation = l1(l=regularization_rate)
        if regularization_type != 'l2':
            regularisation = l2(l=regularization_rate)


    train_datagen = ImageDataGenerator(rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(f"../dataset-split/train",
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        class_mode='categorical',
                                                        target_size=(size, size))

    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(f"../dataset-split/validation",
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  class_mode='categorical',
                                                                  target_size=(size, size))

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(f"../dataset-split/test",
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      class_mode='categorical',
                                                      target_size=(size, size))

    steps_per_epoch = ceil(train_generator.samples/train_generator.batch_size)
    validation_steps = ceil(validation_generator.samples/validation_generator.batch_size)
    test_steps = ceil(test_generator.samples/test_generator.batch_size)

    RUN_NAME = f"cnn-all-loss-s{size}-bs{batch_size}-e{epochs}-spe{steps_per_epoch}-dr{dropout}-fn{filter_number}-lr{learning_rate}-rt{regularization_type}-rr{regularization_rate}"

    target_dir = './models/'

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    if not os.path.exists(f"{target_dir}{RUN_NAME}"):
        os.mkdir(f"{target_dir}{RUN_NAME}")

    target_dir = f"{target_dir}{RUN_NAME}"

    best_model_file = f"{target_dir}/best-weights.h5"

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto'),
        ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True),
        TensorBoard(f"logs/{RUN_NAME}-{start_run_time}")
    ]

    with mlflow.start_run():
        mlflow.log_param('input_size', size)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('steps_per_epoch', steps_per_epoch)
        mlflow.log_param('dropout', dropout)
        mlflow.log_param("filter_number", filter_number)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("regularization_type", regularization_type)
        mlflow.log_param("regularization_rate", regularization_rate)

        model = createModel(filter_number, learning_rate, dropout, regularisation)
        history = model.fit_generator(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=1,
            callbacks=callbacks)
        loss = model.evaluate_generator(validation_generator, steps=validation_steps, verbose=1)

        model.save(f"{target_dir}/model.h5")
        model.save_weights(f"{target_dir}/weights.h5")

        predictions = model.predict_generator(test_generator, steps=test_steps, verbose=1)
        predictions = np.argmax(predictions, axis=1)


        def log_metric(metric_name):
            for idx in range(len(history.history[metric_name])):
                mlflow.log_metric(metric_name, history.history[metric_name][idx], step=idx + 1)


        log_metric("accuracy")
        log_metric("val_accuracy")
        log_metric("loss")
        log_metric("val_loss")
        log_metric("auc")
        log_metric("val_auc")
        log_metric("precision")
        log_metric("val_precision")
        log_metric("recall")
        log_metric("val_recall")
        log_metric("true_positives")
        log_metric("val_true_positives")
        log_metric("true_negatives")
        log_metric("val_true_negatives")
        log_metric("false_positives")
        log_metric("val_false_positives")
        log_metric("false_negatives")
        log_metric("val_false_negatives")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.tensorflow.log_model(model, "models", registered_model_name=RUN_NAME)
        else:
            mlflow.keras.log_model(model, "models")

        predictions = model.predict_generator(test_generator, steps=test_steps, verbose=1)
        predictions = np.argmax(predictions, axis=1)

        print('Confusion Matrix')
        cm = confusion_matrix(test_generator.classes, predictions)
        fig, ax = plt.subplots(figsize=(20, 16))
        sns.heatmap(cm, annot=True, ax=ax, fmt='g', annot_kws={"fontsize": 18})
        fig.savefig(f"{target_dir}/confusion-matrix.png")
        mlflow.log_artifact(f"{target_dir}/confusion-matrix.png")

        print('Classification Report')
        target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        class_report = classification_report(test_generator.classes, predictions, target_names=target_names)
        print(class_report)
        with open(f"{target_dir}/classification-report.txt", 'w') as f:
            f.write(class_report)
        mlflow.log_artifact(f"{target_dir}/classification-report.txt")

        dict_class_report = classification_report(
            test_generator.classes,
            predictions,
            target_names=target_names,
            output_dict=True)

        for i in range(11):
            mlflow.log_metric(f"class_{i}_precision", dict_class_report[f"{i}"]['precision'])
            mlflow.log_metric(f"class_{i}_recall", dict_class_report[f"{i}"]['recall'])
            mlflow.log_metric(f"class_{i}_f1_score", dict_class_report[f"{i}"]['f1-score'])
            mlflow.log_metric(f"class_{i}_support", dict_class_report[f"{i}"]['support'])
        mlflow.log_metric('test_accuracy', dict_class_report['accuracy'])
        mlflow.log_metric('macro_avg_precision', dict_class_report["macro avg"]['precision'])
        mlflow.log_metric('macro_avg_recall', dict_class_report["macro avg"]['recall'])
        mlflow.log_metric('macro_avg_f1_score', dict_class_report["macro avg"]['f1-score'])
        mlflow.log_metric('macro_avg_support', dict_class_report["macro avg"]['support'])
        mlflow.log_metric('weighted_avg_precision', dict_class_report["weighted avg"]['precision'])
        mlflow.log_metric('weighted_avg_recall', dict_class_report["weighted avg"]['recall'])
        mlflow.log_metric('weighted_avg_f1_score', dict_class_report["weighted avg"]['f1-score'])
        mlflow.log_metric('weighted_avg_support', dict_class_report["weighted avg"]['support'])
