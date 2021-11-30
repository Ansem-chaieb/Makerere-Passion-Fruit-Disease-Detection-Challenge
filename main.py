import os
import yaml
import pandas as pd
from box import Box

from src.data_check import get_information, show_samples
from src.data_process import pascal_to_yolo, generate_label_txt
from src.train_val_split import train_test_split
from src.train import training
from src.inference import detect, generate_yolo_sub, convert_yolo_sub


from utils import get_logger, get_args
import warnings

warnings.filterwarnings("ignore")
logger = get_logger()
args = get_args()
with open("src/config.yaml", "r") as ymlfile:
    config = Box(yaml.safe_load(ymlfile))


train_df = pd.read_csv(config.DATA_PATHS.TRAIN_PATH)

if args.information:
    logger.info("------------ Data checking-----------------")
    get_information(config, train_df)
if args.display:
    files = os.listdir(config.DATA_PATHS.TRAIN_DATA_PATH)
    image_path = []
    for f in files:
        image_path.append(f)
    show_samples(config, (2, 4), train_df, image_path, figsize=(20, 8))
if args.process_data:
    logger.info("------------Transform data----------------")
    yolo_data = pascal_to_yolo(config, train_df)
    logger.info("---------Train validation split--------------")
    df_train, df_val = train_test_split(config, yolo_data)
    logger.info("---------Create Yolo labels files-----------")
    generate_label_txt(config, df_train, "train")
    generate_label_txt(config, df_val, "validation")
if args.train:
    logger.info("------------Train yolov5--------------------")
    training(config)
if args.test:
    logger.info("------------Test yolov5--------------------")
    detect(config)
    yolo_data = generate_yolo_sub(config)
    convert_yolo_sub(config, yolo_data)
