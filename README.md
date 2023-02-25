# TF-TRT 를 이용한 Jetson 환경에서의 객체 인식 모델 구동 

- Jetpack 4.6.3 (https://developer.nvidia.com/jetpack-sdk-463) 

- SD Card Image Method 

** 1. 도커 L4T R32.7.1 설치 **

Jetpack 4.6.3에 해당하는 l4t r32.7.1을 설치해야한다. 
```
$ sudo docker pull nvcr.io/nvidia/l4t-tensorflow:r35.2.1-tf2.11-py3 
$ sudo docker run -it --rm --runtime nvidia --network host -v /home/user/project:/location/in/container nvcr.io/nvidia/l4t-tensorflow:r35.2.1-tf2.11-py3 
```
 

  

 ※ (Optional) Tensorflow 직접 설치 
```
$ sudo apt-get update 
$ sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev $ liblapack-dev libblas-dev gfortran                                                                                                  
$ sudo apt-get install python3-pip 
$ sudo pip3 install -U pip testresources setuptools 
$ sudo ln -s /usr/include/locale.h /usr/include/xlocale.h 
$ sudo pip3 install -U numpy==1.19.4 future mock keras_preprocessing keras_applications gast==0.2.1 protobuf pybind11 cython pkgconfig packaging 
$ sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 tensorflow 
```
 

 

** 2. Frozen graph – saved model 변환 파일 frozen_graph_to_saved_model.py **

Tensorflow에서 모델에 대한 정보 / 학습에 대한 정보를 .pb확장자명으로 저장할 수 있다. 

두가지 방법으로 나뉜다. 1) Frozen graph 2) saved model이다. 

현재 본인이 가지고 있는 파일이frozen graph 형식인 경우 파라미터 변경이 불가능하기 때문에, tensorRT를 사용하기위해 saved model형식으로 변환하여야 한다. 
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
frozen graph에서 saved model로 변환하기 위해 모델의 출력 노드를 알아야 한다.  

예제에서 적용한 모델은 ssd mobilenet v1을 이용한 것으로 'num_detections', 'detection_scores', 'detection_boxes', 'detection_classes' 4개의 노드를 출력으로 가진다. 

 

** 3. saved model 파일을 이용한 tf-trt 최적화 **

빌드 파일 tf-trt.py 

※ 파라미터 

VirtualDeviceConfiguration.memory_limit : 가상 기기(여기에서는 GPU)에 할당할 최대 메모리(MB) 

Input_fn : converter build의 입력으로 이미지의 크기를 맞춰주어야한다. 

max_workspace_size_bytes: 정수, TensorRT에 사용 가능한 최대 GPU 메모리 크기 

  

※ 실행 명령 예시 

Tensorflow의 saved mode.pb을 SAVED_MODEL_DIR의 위치에 넣어준뒤,  
```
$ python3 tf-trt.py$ python3 tf-trt.py 
```
이후 OUTPUT_SAVED_MODEL_DIR의 위치에 tensorRT가 적용된 .pb파일이 생성된것을 확인할 수 있습니다. 

 
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
 

2. 실행속도 단축의 정도를 판단하기위한 테스트파일 test.py 
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
 

 

test.py의 출력은 다음과 같이 나옵니다. 

Nano : 5 fps -> 7 fps, NX : 7 fps -> 12 fps, AGX : 15 fps -> 30 fps 

[Nano] 

 

[AGX] 

 

 

 

※정확도(예측률) 비교 

명확한 input 

72% -> 84% 

정답에 대한 예측률이 더 좋아짐 

 

 

정답 예측률이 더 안좋아지는 경우도 있음 

90% -> 53% 

 

 

멀리있어 식별이 어려운경우 & 뒤쪽 신호등이 함께 보일때 

78%, 97%(오답 : 직진신호를 좌회전신호로 인식) 

-> 78%, 96%(마찬가지로 오답인식) 

 

화면 중앙에서 벗어나 가장자리에 있는 경우 인식 못함 

입력 이미지 더 멀리 디지스트쪽에 빨간불이 더 있으나 인식 못함 

: 인식범위 한계 = 500~600m (카메라 성능에 따라 달라질듯) 

 

 

약간 어두워지기 시작한 케이스 

 

추후 첨부 

 

 

 

※ 메모리 부족할시 

1. zram (메모리 압축 영역) 확장  

 	 

따라서 zram을 기본값 2gb에서 4gb로 확장하여 사용하도록 함  

첨부한 링크의 쉘 스크립트 파일을 사용하면 쉽게 수정 가능함  

 	 

https://github.com/JetsonHacksNano/resizeSwapMemory  

 	 

 	 

2. zram 해제  

  

zram의 크기를 너무 크게 설정할 경우 오히려 성능에 지장이 갈 수 있으므로, 더 많은 메모리가 필요한 경우 swap file을 이용하도록 함  

  

sudo systemctl disable nvzramconfig  

  

https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-transfer-learning.md  

 	 

 	 

3. swap file 설정  

 	 

Tensorflow는 gpu 메모리를 가상 장치의 형태로 미리 할당받아 사용함  

Jetson Nano는 4gb 메모리를 cpu, gpu가 함께 사용함  

TensorRT 모델 빌드 작업의 경우 cpu, gpu가 동시에 많은 양의 메모리를 사용함  

gpu에 메모리를 많이 할당한 경우 남아있는 적은 양의 메모리로 모든 cpu 작업을 처리하기 위해 스왑이 너무 자주 일어나 전체적인 작업 속도가 매우 느려짐  

따라서 zram이 사용하던 공간을 임시로 해제하고 swap file로 대체하여 스왑이 일어나는 빈도를 줄임  

첨부한 링크의 쉘 스크립트 파일을 사용하면 쉽게 수정 가능함  

 	 

https://github.com/JetsonHacksNano/installSwapfile  

 	 

 	 

4. GUI 비활성화  

  

sudo systemctl set-default multi-user.target  

sudo reboot  

 

 ---

※ 참고 사이트 

Jetpack 설치 : https://developer.nvidia.com/jetpack-sdk-463  

https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorflow  

Tensorflow 직접 설치 : https://forums.developer.nvidia.com/t/tensorflow-for-jetson-tx2/64596  

Tf-trt : https://www.tensorflow.org/api_docs/python/tf/experimental/tensorrt/Converter  

Frozen graph – saved model : https://stackoverflow.com/questions/44329185/convert-a-graph-proto-pb-pbtxt-to-a-savedmodel-for-use-in-tensorflow-serving-o/44329200#44329200  
