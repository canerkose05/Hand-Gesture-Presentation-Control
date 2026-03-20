ALL_FEATURES = [
    "nose_x",
    "nose_y",
    "nose_z",
    "nose_confidence",
    "left_eye_inner_x",
    "left_eye_inner_y",
    "left_eye_inner_z",
    "left_eye_inner_confidence",
    "left_eye_x",
    "left_eye_y",
    "left_eye_z",
    "left_eye_confidence",
    "left_eye_outer_x",
    "left_eye_outer_y",
    "left_eye_outer_z",
    "left_eye_outer_confidence",
    "right_eye_inner_x",
    "right_eye_inner_y",
    "right_eye_inner_z",
    "right_eye_inner_confidence",
    "right_eye_x",
    "right_eye_y",
    "right_eye_z",
    "right_eye_confidence",
    "right_eye_outer_x",
    "right_eye_outer_y",
    "right_eye_outer_z",
    "right_eye_outer_confidence",
    "left_ear_x",
    "left_ear_y",
    "left_ear_z",
    "left_ear_confidence",
    "right_ear_x",
    "right_ear_y",
    "right_ear_z",
    "right_ear_confidence",
    "left_mouth_x",
    "left_mouth_y",
    "left_mouth_z",
    "left_mouth_confidence",
    "right_mouth_x",
    "right_mouth_y",
    "right_mouth_z",
    "right_mouth_confidence",
    "left_shoulder_x",
    "left_shoulder_y",
    "left_shoulder_z",
    "left_shoulder_confidence",
    "right_shoulder_x",
    "right_shoulder_y",
    "right_shoulder_z",
    "right_shoulder_confidence",
    "left_elbow_x",
    "left_elbow_y",
    "left_elbow_z",
    "left_elbow_confidence",
    "right_elbow_x",
    "right_elbow_y",
    "right_elbow_z",
    "right_elbow_confidence",
    "left_wrist_x",
    "left_wrist_y",
    "left_wrist_z",
    "left_wrist_confidence",
    "right_wrist_x",
    "right_wrist_y",
    "right_wrist_z",
    "right_wrist_confidence",
    "left_pinky_x",
    "left_pinky_y",
    "left_pinky_z",
    "left_pinky_confidence",
    "right_pinky_x",
    "right_pinky_y",
    "right_pinky_z",
    "right_pinky_confidence",
    "left_index_x",
    "left_index_y",
    "left_index_z",
    "left_index_confidence",
    "right_index_x",
    "right_index_y",
    "right_index_z",
    "right_index_confidence",
    "left_thumb_x",
    "left_thumb_y",
    "left_thumb_z",
    "left_thumb_confidence",
    "right_thumb_x",
    "right_thumb_y",
    "right_thumb_z",
    "right_thumb_confidence",
]


def normalize_around_nose(df):
    df = df.copy()

    for axis in ["x", "y", "z"]:
        nose_column = f"nose_{axis}"
        target_columns = [
            col for col in df.columns
            if col.endswith(f"_{axis}") and col != nose_column
        ]

        df[target_columns] = df[target_columns].sub(df[nose_column], axis=0)
        df[nose_column] = 0

    return df


def normalize_distances(df):
    df = df.copy()
    x_reference = "left_shoulder_x"
    y_reference = "left_shoulder_y"

    x_columns = [col for col in df.columns if col.endswith("_x") and col != x_reference]
    y_columns = [col for col in df.columns if col.endswith("_y") and col != y_reference]

    safe_x = df[x_reference].replace(0, 1e-8)
    safe_y = df[y_reference].replace(0, 1e-8)

    df[x_columns] = df[x_columns].div(safe_x, axis=0)
    df[y_columns] = df[y_columns].div(safe_y, axis=0)

    df[x_reference] = 1
    df[y_reference] = 1

    return df


def normalize_position_and_distances(df):
    df = normalize_around_nose(df)
    df = normalize_distances(df)
    return df