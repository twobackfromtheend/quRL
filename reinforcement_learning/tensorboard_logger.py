import os
import pathlib
from typing import List

import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard

LOG_FOLDER = "tensorboard_logs"

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


i = 0


def create_callback(model) -> TensorBoard:
    global i
    _log_path = os.path.join(log_path, f"run_{i}")
    pathlib.Path(_log_path).mkdir(parents=True, exist_ok=True)
    callback = TensorBoard(_log_path)
    callback.set_model(model)
    i += 1
    return callback
