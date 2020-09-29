FROM wzj751127122/caffe:19.12-py3

#RUN apt update
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && apt-get update

RUN apt install -y ffmpeg

#RUN apt install -y cmake

RUN apt install -y python3.6-venv

RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com\
 && pip install scikit-build opencv-python sklearn -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com\
 && pip list

RUN pip -V
RUN python -V

#RUN python3 -m venv venv && source venv/bin/activate && python -V && pip -V \
# && pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com\
# && pip install scikit-build opencv-python sklearn -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com\
# && pip list

#CMD source venv/bin/activate && python V && pip -V

