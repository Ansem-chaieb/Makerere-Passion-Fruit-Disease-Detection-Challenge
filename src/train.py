import os


def training(config):
    os.system(
        "python3 yolov5/train.py --img {} --batch {} --epochs {} "
        "--data makerere.yaml --weights yolov5s.pt --cache".format(
            config.DATASET.IMAGE_SIZE,
            config.TRAIN.BATCH_SIZE,
            config.TRAIN.EPOCHS,
        )
    )
