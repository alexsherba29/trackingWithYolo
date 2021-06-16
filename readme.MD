# Wiliot project

original model located in [here](https://drive.google.com/file/d/1HGbaF61x4N9mLbjWYYPNrnIyQG1WoLUb/view?usp=sharing)
tiny model located in [here](https://drive.google.com/file/d/1luUMPAj-MvTD-xRoZ3tunt9P8k-j2Jfy/view?usp=sharing)

## This container is for NVIDIA Jetson nano with deepstreamV5.1 platform 

### Description 
YOLO model that train to detect white rectangel. Work on deepdtream and with one, two and four 4vl2(USB) camera.

### Installation & Usage

```bash
$ sudo chmod -R 777 /opt/nvidia/deepstream/deepstream-5.1/sources/

$ cd /opt/nvidia/deepstream/deepstream-5.1/sources

$ git clone git clone https://github.com/dspip/willot_project.git -b tiny21FPS

$ cd /opt/nvidia/deepstream/deepstream-5.1/sources/willot_project

$ CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo

```
now you can move the alexProject dir where ever you want

```python

$ python3 wiliotTagTracker.py -c tag_tracker_configure_file.yaml

```

#### OR with DOCKER
``` 
$ xhost +

$ export CONN_HTTP="http://<ip address>:<port>"

$ docker pull 247555/tagtracking:latest

$ docker run -it --rm --network="host" --runtime nvidia --gpus all -e CONN_HTTP=$CONN_HTTP --device /dev/video0:/dev/video0 --device /dev/video1:/dev/video1 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix 247555/tagtracking
```
