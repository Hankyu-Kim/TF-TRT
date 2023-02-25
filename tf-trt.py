import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
tf.config.experimental.set_virtual_device_configuration(
    gpu_devices[0],
    [tf.config.experimental.VirtualDeviceConfiguration(
        memory_limit=3072)])

SAVED_MODEL_DIR = "."
OUTPUT_SAVED_MODEL_DIR = "saved_model_TFTRT_uint8"


def input_fn():
    yield [np.random.randint(low=0, high=255, size=(16, 360, 640, 3), dtype=np.uint8)]


# Instantiate the TF-TRT converter
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=SAVED_MODEL_DIR,
    max_workspace_size_bytes=1 << 26,
    precision_mode=trt.TrtPrecisionMode.FP16,
    maximum_cached_engines=1,
)

# Convert the model into TRT compatible segments
trt_func = converter.convert()
converter.summary()

converter.build(input_fn=input_fn)
converter.save(output_saved_model_dir=OUTPUT_SAVED_MODEL_DIR)
