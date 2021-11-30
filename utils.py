import os
import argparse
import logging


def get_logger():
    PATH = "output/"
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    log_file = os.path.join(PATH, "output.log")
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="%(levelname)s:%(message)s",
        level=logging.DEBUG,
    )

    return logging


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--information",
        help="Get informtions about your dataset.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--display",
        help="Plot batch of dataset images.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--process_data",
        help="Process coordinates from pascal voc to yolo, "
        " split dataset to train and validation and"
        "then create yolo labes files.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--train",
        help="Train yolov5 on custom data.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--inference",
        help="Test yolov5 on custom data.",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    return args
