from pathlib import Path
import pickle

from nn_framework.NN_Layer import Layer
from nn_framework.Cost_Functions import CrossEntropy
from nn_framework.PCA import PCA
from gesture_classifier.gesture_classifier import GestureClassifier


TRAIN_DATA_DIR = Path("data/train")
VALIDATION_DATA_DIR = Path("data/validation")
MODEL_OUTPUT_PATH = Path("trained_model/classifier.pkl")

SELECTED_FEATURES = [
    "left_shoulder_x",
    "left_shoulder_y",
    "right_shoulder_x",
    "right_shoulder_y",
    "left_elbow_x",
    "left_elbow_y",
    "right_elbow_x",
    "right_elbow_y",
    "left_wrist_x",
    "left_wrist_y",
    "right_wrist_x",
    "right_wrist_y",
]

CLASSES = [
    "idle",
    "swipe_right",
    "swipe_left",
    "rotate_right",
]

N_FRAMES = 7
TEST_SIZE = 0.2
SCALER = PCA(70)

LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EPOCHS = 25


def collect_csv_files(folder: Path) -> list[str]:
    return sorted(str(path) for path in folder.rglob("*.csv"))


def build_parameters() -> dict:
    input_size = (
        SCALER.num_components
        if isinstance(SCALER, PCA)
        else N_FRAMES * len(SELECTED_FEATURES)
    )
    output_size = len(CLASSES)

    return {
        "classes": CLASSES,
        "selected_features": SELECTED_FEATURES,
        "n_frames": N_FRAMES,
        "test_size": TEST_SIZE,
        "scaler": SCALER,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "layers": [
            Layer(input_size, input_size * 10),
            Layer(input_size * 10, output_size),
        ],
        "cost_funct": CrossEntropy(binary=False),
        "softmax": True,
        "adam_optimizer": True,
    }


def main() -> None:
    train_files = collect_csv_files(TRAIN_DATA_DIR)
    test_files = collect_csv_files(VALIDATION_DATA_DIR)

    if not train_files:
        raise FileNotFoundError(f"No training CSV files found in {TRAIN_DATA_DIR}")
    if not test_files:
        raise FileNotFoundError(f"No validation CSV files found in {VALIDATION_DATA_DIR}")

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    parameters = build_parameters()
    classifier = GestureClassifier(parameters)
    classifier.train(train_files, test_files)

    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(classifier, f)


if __name__ == "__main__":
    main()