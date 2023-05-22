ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

MAINTAINER yeyy

RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update
RUN apt-get install -y python3.8 ffmpeg python3-pip

ADD enhancer-b201.tar.gz /Enhancer-B201
RUN pip install -r /Enhancer-B201/requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN echo "export LC_ALL=\"C.UTF-8\"" >> ~/.bashrc

RUN python /Enhancer-B201/code/models/DCNv2/setup_indocker.py build develop

WORKDIR /Enhancer-B201
