# Using frozen graph through tensorRT for object detection from jetson platform

- Jetpack 4.6.3 (https://developer.nvidia.com/jetpack-sdk-463) 

- SD Card Image Method 

**1. docker installation (L4T R32.7.1)**

We have to install l4t r32.7.1 which is for Jetpack 4.6.3. 
```
sudo docker pull nvcr.io/nvidia/l4t-tensorflow:r35.2.1-tf2.11-py3 
sudo docker run -it --rm --runtime nvidia --network host -v /home/user/project:/location/in/container nvcr.io/nvidia/l4t-tensorflow:r35.2.1-tf2.11-py3 
```
 

  

 ※ (Optional) Tensorflow installation locally.
```
sudo apt-get update 
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev $ liblapack-dev libblas-dev gfortran
sudo apt-get install python3-pip 
sudo pip3 install -U pip testresources setuptools 
sudo ln -s /usr/include/locale.h /usr/include/xlocale.h 
sudo pip3 install -U numpy==1.19.4 future mock keras_preprocessing keras_applications gast==0.2.1 protobuf pybind11 cython pkgconfig packaging 
sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 tensorflow 
```
 
<br/>
 

**2. Frozen graph – saved model coversion 'frozen_graph_to_saved_model.py'**

<br/>

Normally, we can save information of our model / training with .pb filename extension from Tensorflow 

There is two ways - 1) Frozen graph 2) saved model

If you currently have frozen graph form, we cannot change parameter inside, so we do have to change it to saved model form.

<br/>

※ command example

```
python3 frozen_graph_to_saved_model.py
```

when we change frozen graph to saved model, we have to know information about output node.  

On our example, model is ssd mobilenet v1, 'num_detections', 'detection_scores', 'detection_boxes', 'detection_classes' are output nodes.

(If you don't have any knowledge of your model's output node, you may have to use some method to figure out.)

 <br/><br/>

**3. tf-trt optimization from saved model**

build file 'tf-trt.py'

※ parameter

VirtualDeviceConfiguration.memory_limit : Maximum memory (MB) to allocate on virtual device(here, GPU)-

Input_fn : Input of converter build, you need to make the size of the image all equal. 

max_workspace_size_bytes: integer, Maximum GPU memory for TF-TRT

<br/>

※ command example

after put 'saved mode.pb' from Tensorflow on SAVED_MODEL_DIR location,  
```
python3 tf-trt.py
```
after that, you may check new tensorRT optimized .pb file on OUTPUT_SAVED_MODEL_DIR. 

  <br/><br/>

2. test file for checking speed - 'test.py' 

<br/>

※ command example

```
python3 test.py
```

<br/>

results of 'test.py'

Nano : 5 fps -> 7 fps, NX : 7 fps -> 12 fps, AGX : 15 fps -> 30 fps 
 
<br/><br/>

### ※ when you do not have enough memory

#### 1. zram (memory compaction) extension

This makes zram 2GB(default) to 4GB  

You can easily modify by following shell script file from the link!
 	
https://github.com/JetsonHacksNano/resizeSwapMemory  

<br/>

#### 2. zram cancellation

if you allocate zram too much, it might cause poor performance, if you need more memory, use swap file.
  
```
sudo systemctl disable nvzramconfig  
```
https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-transfer-learning.md  

<br/>

#### 3. swap file setting

Tensorflow usually allocate gpu memory with virtual mcachine form.

But for example, Jetson Nano share cpu, gpu memory 4GB  

And also 'TensorRT build' use cpu, gpu both a lot of memory (RAM)  

If 'build' allocate gpu a lot of memory, swaps occur too often to handle all cpu operations with a small amount of memory left, resulting in very slow overall operations  

Therefore, we need to temporarily releases the space used by zram and replaces it with swap files to reduce the frequency of swaps.  

You can easily modify using shell script files in the link.
 	 

https://github.com/JetsonHacksNano/installSwapfile  

<br/>

#### 4. GUI deactivation (will help you to save more memory)
  
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
