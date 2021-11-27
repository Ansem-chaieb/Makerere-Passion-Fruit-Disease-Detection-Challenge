from utils import get_logger

logger = get_logger()


def train_test_split(config, data):
    df_train = data.iloc[:2608]
    df_val = data.iloc[2608:]

    logger.info(
        "--------------------------Train data-------------------------------"
    )
    logger.info("df_train shape {}".format(df_train.shape))
    target_dist(config, df_train)
    logger.info(
        "--------------------------val data-------------------------------"
    )
    logger.info("df_val shape {}".format(df_val.shape))
    target_dist(config, df_val)
    return df_train, df_val


def target_dist(config, df):
    target_count = [
        df[df["class"] == 2].shape[0],
        df[df["class"] == 0].shape[0],
        df[df["class"] == 1].shape[0],
    ]
    logger.info("{}: {}".format(config.DATASET.TARGETS[0], target_count[0]))
    logger.info("{}: {}".format(config.DATASET.TARGETS[1], target_count[1]))
    logger.info("{}: {}".format(config.DATASET.TARGETS[2], target_count[2]))
