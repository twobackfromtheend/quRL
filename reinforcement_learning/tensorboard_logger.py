import logging
import os
import pathlib
from typing import List

import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard

LOG_FOLDER = "tensorboard_logs"
FOLDER_PREFIX = "run"

logger = logging.getLogger(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(dir_path, LOG_FOLDER)


# See https://justttry.github.io/justttry.github.io/tensorboard-on-train_on_batch/
def tf_log(callback: TensorBoard, names: List[str], logs: List[float], batch_no: int):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def create_callback(model) -> TensorBoard:
    i = find_max_run_number() + 1
    _log_path = os.path.join(log_path, f"{FOLDER_PREFIX}_{i}")
    pathlib.Path(_log_path).mkdir(parents=True, exist_ok=True)
    callback = TensorBoard(_log_path)
    callback.set_model(model)
    logger.info(f"Created {_log_path} in {LOG_FOLDER}")

    return callback


def find_max_run_number() -> int:
    files = os.listdir(log_path)
    run_numbers = []
    for file in files:
        if file.startswith(FOLDER_PREFIX):
            run_number = int(file[len(FOLDER_PREFIX) + 1:])
            run_numbers.append(run_number)
    return max(run_numbers)


if __name__ == '__main__':
    find_max_run_number()
