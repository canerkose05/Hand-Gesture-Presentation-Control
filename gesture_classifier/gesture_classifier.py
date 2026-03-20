import json
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from gesture_classifier.data_preprocessing import normalize_position_and_distances
from nn_framework.Neural_Network import NeuralNetwork


class GestureClassifier:
    def __init__(self, parameters: dict):
        self.parameters = parameters

        for key, value in parameters.items():
            setattr(self, key, value)

        self.nn = None

    def normalize_position_and_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        return normalize_position_and_distances(df)

    def train(self, files_train, files_test=None):
        X_train, y_train, X_validation, y_validation = self.train_validation_files_to_tensors(files_train)

        if files_test is not None:
            X_test, y_test = self.test_files_to_tensor(files_test)
        else:
            print("No test set provided, using validation data for testing.")
            X_test, y_test = X_validation, y_validation

        if self.scaler is not None:
            self.scaler.fit(X_train)
            X_train = self.scaler.transform(X_train)
            X_validation = self.scaler.transform(X_validation)
            X_test = self.scaler.transform(X_test)

        self.nn = self.train_model(X_train, y_train, X_validation, y_validation)
        self.test_model(X_test, y_test)

    def train_validation_files_to_tensors(self, files_train):
        dfs = self.create_dataframe_from_files(files_train)
        X, y = self.dfs_to_tensor(dfs)
        X, y = self.balance_classes(X, y)

        X_train, X_validation, y_train, y_validation = self.train_test_split(
            X=X,
            y=y,
            test_size=self.test_size,
            random_state=42,
        )
        return X_train, y_train, X_validation, y_validation

    def test_files_to_tensor(self, files_test):
        dfs_test = self.create_dataframe_from_files(files_test)
        X_test, y_test = self.dfs_to_tensor(dfs_test)
        X_test, y_test = self.balance_classes(X_test, y_test)
        return X_test, y_test

    def create_dataframe_from_files(self, files):
        dfs = []

        for file in files:
            df = pd.read_csv(file)
            df["from_file"] = file
            df = self.normalize_position_and_distances(df)
            dfs.append(df)

        return dfs

    def train_test_split(self, X, y, test_size=0.2, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)

        indices = np.arange(len(X))
        np.random.shuffle(indices)

        split_index = int(len(X) * (1 - test_size))

        X_train = X[indices[:split_index]]
        X_test = X[indices[split_index:]]
        y_train = y[indices[:split_index]]
        y_test = y[indices[split_index:]]

        return X_train, X_test, y_train, y_test

    def apply_multiframe(self, X, y):
        X_new = np.hstack([
            X[i:X.shape[0] - (self.n_frames - 1) + i, :]
            for i in range(self.n_frames)
        ])
        y_new = y[self.n_frames - 1:, :]
        return X_new, y_new

    def dfs_to_tensor(self, dfs):
        Xs = []
        ys = []

        for df in dfs:
            df = df[df["ground_truth"].isin(self.classes)]

            X = df[self.selected_features].values
            y = np.array(df["ground_truth"].map(self.one_hot_encoding).values.tolist())

            if len(X) < self.n_frames:
                continue

            X, y = self.apply_multiframe(X, y)
            Xs.append(X)
            ys.append(y)

        if not Xs or not ys:
            raise ValueError("No valid training samples were created. Check your data and class labels.")

        X = np.vstack(Xs)
        y = np.vstack(ys)
        return X, y

    def balance_classes(self, X, y):
        unique_values, counts = np.unique(y, axis=0, return_counts=True)
        min_count = np.min(counts)

        balanced_X = []
        balanced_y = []

        for value in unique_values:
            indices = np.where((y == value).all(axis=1))[0]
            np.random.shuffle(indices)
            balanced_X.extend(X[indices[:min_count]])
            balanced_y.extend(y[indices[:min_count]])

        return np.array(balanced_X), np.array(balanced_y)

    def one_hot_encoding(self, label):
        one_hot = np.zeros(len(self.classes))
        one_hot[self.classes.index(label)] = 1
        return one_hot

    def one_hot_decoding(self, one_hot_vector):
        return self.classes[np.argmax(one_hot_vector)]

    def export_results(self, prediction, groundtruth, costs):
        output_dir = Path("results/basic_gesture_classifier")
        output_dir.mkdir(parents=True, exist_ok=True)

        overall_accuracy = np.sum(groundtruth == prediction) / len(groundtruth)

        export_parameters = dict(self.parameters)
        export_parameters["overall_accuracy"] = float(overall_accuracy)
        export_parameters["scaler"] = str(export_parameters.get("scaler"))
        export_parameters["cost_funct"] = str(export_parameters.get("cost_funct"))
        export_parameters["layers"] = [
            (layer.input_size, layer.output_size, layer.activation_funct)
            for layer in self.layers
        ]

        confusion_matrix = np.zeros((len(self.classes), len(self.classes)))

        for i in range(len(self.classes)):
            for j in range(len(self.classes)):
                confusion_matrix[i, j] = np.sum((prediction == j) & (groundtruth == i))

        row_sum = np.sum(confusion_matrix, axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        confusion_matrix_percentage = confusion_matrix / row_sum * 100

        dict_string = "\n".join([f"{key}: {value}" for key, value in export_parameters.items()])
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        fig, ax = plt.subplots(figsize=(15, 5))
        fig.subplots_adjust(right=0.8)

        heatmap = sns.heatmap(
            confusion_matrix_percentage,
            xticklabels=self.classes,
            yticklabels=self.classes,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            ax=ax,
        )
        heatmap.set_ylabel("Actual Class")
        heatmap.set_xlabel("Predicted Class")
        plt.title("Confusion Matrix (%)")

        plt.text(
            1.15,
            0.5,
            dict_string,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        plt.savefig(output_dir / f"cm_{timestamp}.png", bbox_inches="tight")
        plt.show()

        iterations = range(1, len(costs) + 1)
        plt.figure()
        plt.ylim(0, 1)
        plt.plot(iterations, costs, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Cost Function Value")
        plt.title("Cost Function per Iteration")
        plt.savefig(output_dir / f"cost_function_{timestamp}.png", bbox_inches="tight")
        plt.show()

        with open(output_dir / f"{timestamp}.json", "w") as f:
            json.dump(export_parameters, f, indent=2)

    def test_model(self, X_test, y_test):
        AL, _ = self.nn.forward_propagation(X_test.T)
        predictions = self.nn.get_predictions(AL)
        groundtruth = self.nn.get_predictions(y_test.T)

        test_accuracy = np.sum(groundtruth == predictions) / len(y_test)
        print("Test accuracy:", test_accuracy)

        print("Ground truth counts:")
        for i, class_name in enumerate(self.classes):
            print(class_name, np.sum(groundtruth == i))

        print("Prediction counts:")
        for i, class_name in enumerate(self.classes):
            print(class_name, np.sum(predictions == i))

        self.export_results(predictions, groundtruth, self.nn.costs)

    def train_model(self, X_train, y_train, X_validation, y_validation):
        nn = NeuralNetwork(
            self.layers,
            self.cost_funct,
            self.softmax,
            self.adam_optimizer,
        )
        nn.gradient_descent(
            X_train.T,
            y_train.T,
            X_validation.T,
            y_validation.T,
            self.learning_rate,
            self.epochs,
            self.batch_size,
        )
        return nn

    def save_classifier(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)