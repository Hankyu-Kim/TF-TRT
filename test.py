import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
import time

batch_size = 1
batched_input = np.zeros((batch_size, 360, 640, 3), dtype=np.uint8)

for i in range(batch_size):
    img_path = './data/img%d.JPG' % (i % 4)
    img = image.load_img(img_path, target_size=(360, 640))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    batched_input[i, :] = x
batched_input = tf.constant(batched_input)
print('batched_input shape: ', batched_input.shape)


def benchmark_tftrt(input_saved_model):
    saved_model_loaded = tf.saved_model.load(
        input_saved_model, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    N_warmup_run = 50
    N_run = 1000
    elapsed_time = []

    for i in range(N_warmup_run):
        labeling = infer(batched_input)

    for i in range(N_run):
        start_time = time.time()
        labeling = infer(batched_input)
        end_time = time.time()
        elapsed_time = np.append(elapsed_time, end_time - start_time)
        if i % 50 == 0:
            print('Step {}: {:4.1f}ms'.format(
                i, (elapsed_time[-50:].mean()) * 1000))

    print('Throughput: {:.0f} images/s'.format(N_run *
                                               batch_size / elapsed_time.sum()))


benchmark_tftrt(".")
