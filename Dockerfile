FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

USER root

RUN apt-get update ##[edited]
RUN apt-get install git ffmpeg libsm6 libxext6 -y

RUN pip install --upgrade pip

RUN pip install jupyter==1.0.0
RUN pip install pandas==1.2.2
RUN pip install nltk==3.5
RUN pip install opencv-python==4.4.0.46
RUN pip install tpu-star==0.0.1-rc9
RUN pip install pre-commit==2.10.1
RUN pip install bezier==2020.5.19
RUN pip install augmixations==0.1.2
RUN pip install albumentations==0.1.12
RUN pip install neptune-client==0.5.1
RUN pip install psutil==5.8.0
RUN pip install tqdm==4.56.2
RUN pip install gdown==3.12.2

WORKDIR /home

ENV NVIDIA_VISIBLE_DEVICES 0

ENV PATH /usr/local/cuda-10.1/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda-10.1/lib:/usr/local/cuda-10.1/lib64:${LD_LIBRARY_PATH}

EXPOSE 8888

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--NotebookApp.password=''", "--NotebookApp.token=''", "--allow-root"]
