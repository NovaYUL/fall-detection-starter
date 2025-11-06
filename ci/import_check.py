"""
Minimal import self-check for CI.
Avoids heavy installs (mmpose/mmcv) by using ci/requirements-ci.txt.
"""

def main():
    # Third-party deps
    import numpy as np  # noqa: F401
    import torch  # noqa: F401
    import sklearn  # noqa: F401
    import cv2  # noqa: F401
    import tsai  # noqa: F401

    # Project imports
    import src  # noqa: F401
    import src.utils.feature_engineering as fe  # noqa: F401
    import src.datasets.fall_dataset as fd  # noqa: F401
    import src.models.lstm_fall as lstm_fall  # noqa: F401
    import src.models.inception_time as inc_time  # noqa: F401
    import src.models.tsai_lstm as tsai_lstm  # noqa: F401
    import src.models.tsai_inception as tsai_inc  # noqa: F401

    print("Import check passed.")


if __name__ == "__main__":
    main()
