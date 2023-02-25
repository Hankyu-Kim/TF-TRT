# Running object detection model on jetson platform using TF-TRT

- Jetpack 4.6.3 (https://developer.nvidia.com/jetpack-sdk-463) 

- SD Card Image Method 

** 1. docker installation (L4T R32.7.1) **

We have to install l4t r32.7.1 which is for Jetpack 4.6.3. 
```
$ sudo docker pull nvcr.io/nvidia/l4t-tensorflow:r35.2.1-tf2.11-py3 
$ sudo docker run -it --rm --runtime nvidia --network host -v /home/user/project:/location/in/container nvcr.io/nvidia/l4t-tensorflow:r35.2.1-tf2.11-py3 
```
 

  

 ※ (Optional) Tensorflow installation locally.
```
$ sudo apt-get update 
$ sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev $ liblapack-dev libblas-dev gfortran                                                                                                  
$ sudo apt-get install python3-pip 
$ sudo pip3 install -U pip testresources setuptools 
$ sudo ln -s /usr/include/locale.h /usr/include/xlocale.h 
$ sudo pip3 install -U numpy==1.19.4 future mock keras_preprocessing keras_applications gast==0.2.1 protobuf pybind11 cython pkgconfig packaging 
$ sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 tensorflow 
```
 

 

** 2. Frozen graph – saved model coversion 'frozen_graph_to_saved_model.py' **

Normally, we can save information of our model / training with .pb filename extension from Tensorflow 

There is two ways - 1) Frozen graph 2) saved model

If you currently have frozen graph form, we cannot change parameter inside, so we do have to change it to saved model form.
```
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
```
when we change frozen graph to saved model, we have to know information about output node.  

On our example, model is ssd mobilenet v1, 'num_detections', 'detection_scores', 'detection_boxes', 'detection_classes' are output nodes.

(If you don't have any knowledge of your model's output node, you may have to use some method to figure out.)

 

** 3. tf-trt ptimization from saved model **

build file 'tf-trt.py'

※ parameter

VirtualDeviceConfiguration.memory_limit : Maximum memory (MB) to allocate on virtual device(here, GPU)-

Input_fn : Input of converter build, you need to make the size of the image all equal. 

max_workspace_size_bytes: integer, Maximum GPU memory for TF-TRT

  

※ command example

after put 'saved mode.pb' from Tensorflow on SAVED_MODEL_DIR location,  
```
$ python3 tf-trt.py$ python3 tf-trt.py 
```
after that, you may check new tensorRT optimized .pb file on OUTPUT_SAVED_MODEL_DIR. 

 
```
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
```
 

2. test file for checking speed - 'test.py' 
```
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
```
<br/><br/>

results of 'test.py'

Nano : 5 fps -> 7 fps, NX : 7 fps -> 12 fps, AGX : 15 fps -> 30 fps 
 
<br/><br/><br/><br/>

※ when you do not have enough memory

1. zram (memory compaction) extension

 	 

to make zram 2gb(default) to 4gb  

you can easily change through shell script file from the link
 	
https://github.com/JetsonHacksNano/resizeSwapMemory  

 	 

 	 

2. zram cancellation

  

if you allocate zram too much, it might cause poor performance, if you need more memory, use swap file.

  
```
sudo systemctl disable nvzramconfig  
```
  

https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-transfer-learning.md  

 	 

 	 

3. swap file setting

 	 

Tensorflow usually allocate gpu memory with virtual mcachine form.

But for example, Jetson Nano share cpu, gpu memory 4GB  

And also 'TensorRT build' use cpu, gpu both a lot of memory (RAM)  

If 'build' allocate gpu a lot of memory, swaps occur too often to handle all cpu operations with a small amount of memory left, resulting in very slow overall operations  

Therefore, we need to temporarily releases the space used by zram and replaces it with swap files to reduce the frequency of swaps.  

You can easily modify using shell script files in the link.
 	 

https://github.com/JetsonHacksNano/installSwapfile  

 	 

 	 

4. GUI deactivation (will help you to save more memory)

  
```
sudo systemctl set-default multi-user.target  

sudo reboot  
```
 

 ---

※ Reference Site

Jetpack installation : https://developer.nvidia.com/jetpack-sdk-463  

https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorflow  

(Optional) Tensorflow installation locally : https://forums.developer.nvidia.com/t/tensorflow-for-jetson-tx2/64596  

Tf-trt : https://www.tensorflow.org/api_docs/python/tf/experimental/tensorrt/Converter  

Frozen graph – saved model : https://stackoverflow.com/questions/44329185/convert-a-graph-proto-pb-pbtxt-to-a-savedmodel-for-use-in-tensorflow-serving-o/44329200#44329200  
