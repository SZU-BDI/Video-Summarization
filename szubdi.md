# work jots

```
sudo docker build . -t szubdi/caffe:latest

#sh dk0.sh python main2.py ../test.mp4
#sh dk.sh python main2.py ../d/test.mp4
sh run.sh ../d/test.mp4

```

local env

```

#sudo sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && apt-get update
cat << EEE | sudo bash
sed -i 's/http:\/\/.*archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list 
apt update
# TODO https://medium.com/@anidh.singh/install-caffe-on-ubuntu-with-cuda-6d0da9e8f860
apt install -y python3-venv cmake caffe-cuda protobuf-compiler
apt install -y python3-opencv
apt install -y libopenblas-dev
apt install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
apt install -y libboost-all-dev libgflags-dev libgoogle-glog-dev liblmdb-dev python3-dev
EEE

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip -V
python -V

pip install protobuf -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
pip install scikit-build opencv-python sklearn -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
pip list


```
