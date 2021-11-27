import os
import ast
import pandas as pd
import numpy as np
import shutil
from tqdm import tqdm
from sklearn import preprocessing


def pascal_to_yolo(config, train_df):
    le = preprocessing.LabelEncoder()
    train_df[config.DATASET.TARGET_COL] = le.fit_transform(
        train_df[config.DATASET.TARGET_COL]
    )

    x_center, y_center, width, height = [], [], [], []
    for r in range(len(train_df)):
        x, y = (
            train_df[config.DATASET.BBOXES_COLS[0]].iloc[r],
            train_df[config.DATASET.BBOXES_COLS[1]].iloc[r],
        )
        w, h = (
            train_df[config.DATASET.BBOXES_COLS[2]].iloc[r],
            train_df[config.DATASET.BBOXES_COLS[3]].iloc[r],
        )
        x_c = x + w / 2
        y_c = y + h / 2
        x_c /= config.DATASET.IMAGE_SIZE
        y_c /= config.DATASET.IMAGE_SIZE
        w /= config.DATASET.IMAGE_SIZE
        h /= config.DATASET.IMAGE_SIZE

        x_center.append(x_c)
        y_center.append(y_c)
        width.append(w)
        height.append(h)

    d = {
        "Image_ID": list(train_df[config.DATASET.ID]),
        "class": list(train_df[config.DATASET.TARGET_COL]),
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
    }
    data = pd.DataFrame(d)
    return data


def generate_label_txt(config, df, data_type="train"):
    classes = []
    bboxes = []
    for Id in tqdm(df[config.DATASET.ID].unique()):
        class_number = len(
            df[config.DATASET.TARGET_COL][df[config.DATASET.ID] == Id].values
        )
        box = []
        C = []
        for i in range(class_number):
            x = df["x_center"][df[config.DATASET.ID] == Id].values[i]
            y = df["y_center"][df[config.DATASET.ID] == Id].values[i]
            w = df["width"][df[config.DATASET.ID] == Id].values[i]
            h = df["height"][df[config.DATASET.ID] == Id].values[i]
            c = df[config.DATASET.TARGET_COL][
                df[config.DATASET.ID] == Id
            ].values[i]
            box.append([x, y, w, h])
            C.append(c)
        bboxes.append(box)
        classes.append(str(C))

    data = pd.DataFrame(
        {
            config.DATASET.ID: df[config.DATASET.ID].unique(),
            config.DATASET.TARGET_COL: classes,
            "bboxes": bboxes,
        }
    )

    for r in tqdm(range(len(data))):
        yolo_data = []
        image_id = data.iloc[r][config.DATASET.ID]
        bboxes = data.iloc[r]["bboxes"]
        bboxes = ast.literal_eval(str(bboxes))
        classes = ast.literal_eval(data.iloc[r][config.DATASET.TARGET_COL])

        num_obj = len(bboxes)

        for nb in range(num_obj):
            x_center, y_center = bboxes[nb][0], bboxes[nb][1]
            w, h = bboxes[nb][2], bboxes[nb][3]
            yolo_data.append([classes[nb], x_center, y_center, w, h])
        yolo_data = np.array(yolo_data)

        np.savetxt(
            os.path.join(
                config.DATA_PATHS.OUTPUT_PATH,
                f"labels/{data_type}/{image_id}.txt",
            ),
            yolo_data,
            fmt=["%d", "%f", "%f", "%f", "%f"],
        )
        shutil.copyfile(
            os.path.join(
                config.DATA_PATHS.TRAIN_DATA_PATH, f"{image_id}.jpg"
            ),
            os.path.join(
                config.DATA_PATHS.OUTPUT_PATH,
                f"images/{data_type}/{image_id}.jpg",
            ),
        )
