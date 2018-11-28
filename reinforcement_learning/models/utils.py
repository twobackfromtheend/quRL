import tensorflow as tf
from tensorflow.keras.layers import LSTM, CuDNNLSTM


def check_if_using_gpu() -> bool:
    return bool(tf.test.gpu_device_name())


def get_LSTM_layer():
    if check_if_using_gpu():
        return CuDNNLSTM
    else:
        return LSTM


if __name__ == '__main__':
    print(check_if_using_gpu())
