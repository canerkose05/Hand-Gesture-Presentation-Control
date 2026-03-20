import asyncio
import pathlib
import pickle

import cv2
import mediapipe as mp
import pandas as pd
import yaml

from gesture_classifier.data_preprocessing import ALL_FEATURES
from gesture_classifier.live_gesture_classifier import LiveGestureClassifier
from socket_communication import send_command


MODEL_PATH = pathlib.Path("trained_model/classifier.pkl")
KEYPOINT_MAPPING_PATH = pathlib.Path("keypoint_mapping.yml")


async def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    script_dir = pathlib.Path(__file__).parent

    show_video = True
    show_data = True
    flip_image = True
    camera_index = 0
    gesture_buffer_size = 7

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam with index {camera_index}")

    with open(script_dir / KEYPOINT_MAPPING_PATH, "r", encoding="utf-8") as yaml_file:
        mappings = yaml.safe_load(yaml_file)
        keypoint_names = mappings["face"] + mappings["body"]

    with open(MODEL_PATH, "rb") as f:
        gesture_classifier = pickle.load(f)

    live_gesture_classifier = LiveGestureClassifier(
        gesture_classifier,
        buffer_size=gesture_buffer_size,
    )

    success = True
    gesture_prediction = "idle"
    gesture_prediction_since_idle = "idle"
    last_sent_gesture = "idle"

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened() and success:
            success, image = cap.read()
            if not success:
                break

            if flip_image:
                image = cv2.flip(image, 1)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

            if show_data and results.pose_landmarks is not None:
                person_in_frame = True
                frame_list = []

                for joint_index, _joint_name in enumerate(keypoint_names):
                    joint_data = results.pose_landmarks.landmark[joint_index]
                    frame_list.extend([
                        joint_data.x,
                        joint_data.y,
                        joint_data.z,
                        joint_data.visibility,
                    ])

                frame_data = pd.DataFrame([frame_list], columns=ALL_FEATURES)

                gesture_prediction, gesture_prediction_since_idle = (
                    live_gesture_classifier.predict(frame_data, send=False)
                )

                if (
                    gesture_prediction_since_idle != "idle"
                    and gesture_prediction_since_idle != last_sent_gesture
                ):
                    print(f"Sending command: {gesture_prediction_since_idle}")
                    send_command(gesture_prediction_since_idle)
                    last_sent_gesture = gesture_prediction_since_idle

                if gesture_prediction_since_idle == "idle":
                    last_sent_gesture = "idle"

            else:
                person_in_frame = False
                gesture_prediction = "idle"
                gesture_prediction_since_idle = "idle"
                last_sent_gesture = "idle"

            if show_video:
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                font = cv2.FONT_HERSHEY_SIMPLEX
                first_text_pos = (10, 50)
                second_text_pos = (10, 100)
                third_text_pos = (10, 150)
                font_scale = 1
                font_color = (0, 0, 255)
                thickness = 2
                line_type = 2

                if person_in_frame:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )

                    cv2.putText(
                        image,
                        f"Current Gesture: {gesture_prediction}",
                        second_text_pos,
                        font,
                        font_scale,
                        font_color,
                        thickness,
                        line_type,
                    )

                    cv2.putText(
                        image,
                        f"Detected Gesture: {gesture_prediction_since_idle}",
                        third_text_pos,
                        font,
                        font_scale,
                        font_color,
                        thickness,
                        line_type,
                    )
                else:
                    cv2.putText(
                        image,
                        "Please move into the frame",
                        first_text_pos,
                        font,
                        font_scale,
                        font_color,
                        thickness,
                        line_type,
                    )

                cv2.imshow("MediaPipe Pose", image)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())