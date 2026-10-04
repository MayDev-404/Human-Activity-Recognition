"""Dataset constants shared by every module (BUILD_SPEC 5.1). Defined once here; import, never copy."""

from __future__ import annotations

# Raw inertial channels in the fixed order used for the (N, 128, 9) raw representation.
# Loaders must stack files in this order, never in directory-listing order.
CHANNELS: list[str] = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

# Class index i (0..5) corresponds to label value i + 1 in the dataset's y_*.txt files.
ACTIVITY_NAMES: list[str] = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING",
]

N_CLASSES = 6
WINDOW_LEN = 128   # 2.56 s at 50 Hz, 50% overlap (pre-segmented by the dataset authors)
N_FEATURES = 561
