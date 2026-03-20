from collections import deque
import numpy as np

from gesture_classifier.gesture_classifier import GestureClassifier
from nn_framework.buffer_predictor import BufferPredictor


class LiveGestureClassifier:
    def __init__(self, gesture_classifier: GestureClassifier, buffer_size=1):
        self.gesture_classifier = gesture_classifier
        self.n_frames = gesture_classifier.n_frames
        self.n_features = len(gesture_classifier.selected_features)
        self.n_classes = len(gesture_classifier.classes)

        self.buffer_predictor = BufferPredictor(buffer_size, self.n_classes)

        self.prediction_product_since_idle = np.ones(self.n_classes)
        self.current_prediction = "idle"
        self.previous_prediction = "idle"

        self.last_n_frames = deque(
            [0.0] * (self.n_features * self.n_frames),
            maxlen=self.n_features * self.n_frames,
        )

        self.idle_counter = 0

    def predict(self, frame_data, send=False):
        normalized_frame_data = self.gesture_classifier.normalize_position_and_distances(frame_data)

        current_features = normalized_frame_data[
            self.gesture_classifier.selected_features
        ].to_numpy()[0]

        self.last_n_frames.extend(current_features)
        X = np.array([self.last_n_frames])

        if self.gesture_classifier.scaler is not None:
            X = self.gesture_classifier.scaler.transform(X)

        AL, _ = self.gesture_classifier.nn.forward_propagation(X.T)

        self.previous_prediction = self.current_prediction
        self.current_prediction = self.gesture_classifier.classes[
            self.buffer_predictor.predict(AL)
        ]

        prediction_since_idle = self.gesture_classifier.classes[
            np.argmax(self.prediction_product_since_idle)
        ]

        if prediction_since_idle == "idle":
            self.prediction_product_since_idle = np.ones(self.n_classes)

        if self.current_prediction == "idle":
            self.idle_counter += 1
        else:
            self.prediction_product_since_idle *= AL.T[0]
            self.idle_counter = 0

        if self.idle_counter > 3 and prediction_since_idle != "idle":
            self.prediction_product_since_idle = np.ones(self.n_classes)

        return self.current_prediction, prediction_since_idle