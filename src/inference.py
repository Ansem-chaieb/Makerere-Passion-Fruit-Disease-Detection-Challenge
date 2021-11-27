import os
import pandas as pd


def detect(config):
    os.system(
        "python3 yolov5/detect.py --weights {} --source {} --save-txt --save-conf".format(
            config.INFERENCE.WEIGHTS_PATH, config.DATA_PATHS.TEST_DATA_PATH
        )
    )


def generate_yolo_sub(config):
    files = os.listdir(config.INFERENCE.LABELS_PATH)
    Image_id, classe, confidence, x, y, w, h = [], [], [], [], [], [], []
    for f in files:
        with open(config.INFERENCE.LABELS_PATH + f, "rt") as myfile:
            for myline in myfile:
                bb = myline.rstrip("\n").split(" ")
                Image_id.append(f.split(".")[0])
                classe.append(config.DATASET.TARGETS[0]) if bb[
                    0
                ] == "0" else (
                    classe.append(config.DATASET.TARGETS[1])
                    if bb[0] == "1"
                    else classe.append(config.DATASET.TARGETS[2])
                )
                confidence.append(float((bb[-1])))
                x.append(float(bb[1]))
                y.append(float(bb[2]))
                w.append(float(bb[3]))
                h.append(float(bb[4]))

    data = pd.DataFrame(
        {
            "Image_ID": Image_id,
            "class": classe,
            "confidence": confidence,
            "x_center": x,
            "y_center": y,
            "width": w,
            "height": h,
        }
    )
    data.to_csv(config.INFERENCE.OUTPUT_PATH + "sub_yolo.csv")
    return data


def convert_yolo_sub(config, yolo_data):
    Image_id, classe, confidence, xmin, ymin, xmax, ymax = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for r in range(len(yolo_data)):
        Image_id.append(yolo_data["Image_ID"].iloc[r])
        classe.append(yolo_data["class"].iloc[r])
        confidence.append(yolo_data["confidence"].iloc[r])
        x_c, y_c = (
            yolo_data["x_center"].iloc[r],
            yolo_data["y_center"].iloc[r],
        )
        width, height = (
            yolo_data["width"].iloc[r],
            yolo_data["height"].iloc[r],
        )

        xmin.append(((2 * x_c * 512) - (width * 512)) / 2)
        xmax.append(((2 * x_c * 512) + (width * 512)) / 2)

        ymin.append(((2 * y_c * 512) - (height * 512)) / 2)
        ymax.append(((2 * y_c * 512) + (height * 512)) / 2)

    data = pd.DataFrame(
        {
            "Image_ID": Image_id,
            "class": classe,
            "confidence": confidence,
            "ymin": ymin,
            "xmin": xmin,
            "ymax": ymax,
            "xmax": xmax,
        }
    )

    data.to_csv(config.INFERENCE.OUTPUT_PATH + "sub_pascal.csv")
