FROM nvcr.io/nvidia/deepstream-l4t:5.1-21.02-samples

RUN apt update 
RUN apt install -y screen libgl1-mesa-glx
RUN apt install -y build-essential
RUN apt install -y python3-pip
RUN apt install -y python3-dev
RUN pip3 install --upgrade pip
RUN pip3 install requests && pip3 install gsutil
RUN apt-get install -y gstreamer-1.0 && apt-get install -y python3-gi

RUN wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
RUN pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

RUN git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
RUN mv deepstream_python_apps /opt/nvidia/deepstream/deepstream-5.1/sources/
RUN mkdir -p  /opt/nvidia/deepstream/deepstream-5.1/sources/deepstream_python_apps/apps/alexProj   ## Create working directory
WORKDIR /opt/nvidia/deepstream/deepstream-5.1/sources/deepstream_python_apps/apps/alexProj


COPY . .

ENV CUDA_VER=10.2
ENV CONN_HTTP="http://localhost:4321"
RUN cd nvdsinfer_custom_impl_Yolo && make 


CMD python3 ./deepstream_test_4.py --conn-http="$CONN_HTTP"
