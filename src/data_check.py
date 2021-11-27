import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects

from utils import get_logger


def get_information(config, train_df):
    logger = get_logger()
    unique_ID = len(train_df[config.DATASET.ID].unique())
    targets = train_df[config.DATASET.TARGET_COL].unique()
    target_count = [
        train_df[
            train_df[config.DATASET.TARGET_COL] == config.DATASET.TARGETS[0]
        ].shape[0],
        train_df[
            train_df[config.DATASET.TARGET_COL] == config.DATASET.TARGETS[1]
        ].shape[0],
        train_df[
            train_df[config.DATASET.TARGET_COL] == config.DATASET.TARGETS[2]
        ].shape[0],
    ]

    logger.info("Columns : {}".format(train_df.columns))
    logger.info("Unique ID: {}".format(unique_ID))
    logger.info("Targets: {}".format(targets))
    logger.info("{}: {}".format(config.DATASET.TARGETS[0], target_count[0]))
    logger.info("{}: {}".format(config.DATASET.TARGETS[1], target_count[1]))
    logger.info("{}: {}".format(config.DATASET.TARGETS[2], target_count[2]))


def get_target_ds(config, name, df):
    rows = df[df[config.DATASET.ID] == name[:-4]]
    return (
        rows[config.DATASET.TARGET_COL].values,
        rows[config.DATASET.BBOXES_COLS].values,
    )


def get_bbox(bboxes):
    boxes = bboxes.copy()
    if boxes.shape[0] == 1:
        return boxes

    return np.squeeze(boxes)


def image_show(image, ax, figsize=(7, 11)):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.xaxis.tick_top()
    ax.imshow(image)
    return ax


def draw_outline(obj):
    obj.set_path_effects(
        [
            patheffects.Stroke(linewidth=4, foreground="black"),
            patheffects.Normal(),
        ]
    )


def draw_box(ax, bb):
    patch = ax.add_patch(
        patches.Rectangle(
            (bb[0], bb[1]), bb[2], bb[3], fill=False, edgecolor="red", lw=2
        )
    )
    draw_outline(patch)


def draw_text(ax, bb, txt, disp):
    text = ax.text(
        bb[0],
        (bb[1] - disp),
        txt,
        verticalalignment="top",
        color="white",
        fontsize=10,
        weight="bold",
    )
    draw_outline(text)


def show_sample(image, bboxes, classes, ax=None):
    bb = get_bbox(bboxes)
    ax = image_show(image, ax=ax)
    for i in range(len(bboxes)):
        draw_box(ax, bb[i])
        draw_text(ax, bb[i], str(classes[i]), image.shape[0] * 0.05)


def show_samples(config, shape, df, images, figsize=(18, 10)):
    ids = np.random.randint(0, len(images) - 1, shape[0] * shape[1])
    fig, ax = plt.subplots(shape[0], shape[1], figsize=figsize)
    plt.subplots_adjust(wspace=0.1, hspace=0)
    fig.tight_layout()
    for i in range(shape[0]):
        for j in range(shape[1]):
            img = images[ids[(i + 1) * j]]
            labels, bboxes = get_target_ds(config, img, df)
            img = cv2.imread(
                config.DATA_PATHS.TRAIN_DATA_PATH + str(img),
                cv2.IMREAD_UNCHANGED,
            )
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            show_sample(img, bboxes, labels, ax=ax[i][j])
    plt.show()
