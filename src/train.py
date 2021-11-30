import os


def training(config):
    os.system(
        "python3 yolov5/train.py --img {} --batch {} --epochs {} "
        "--data makerere.yaml --weights {} --cache".format(
            config.DATASET.IMAGE_SIZE,
            config.TRAIN.BATCH_SIZE,
            config.TRAIN.EPOCHS,
            config.TRAIN.MODEL_WEIGHTS
        )
    )
