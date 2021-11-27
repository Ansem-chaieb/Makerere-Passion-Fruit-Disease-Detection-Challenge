import os
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
