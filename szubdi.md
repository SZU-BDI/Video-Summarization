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
apt install -y python3-venv cmake caffe-cuda
EEE

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip -V
python -V


pip install scikit-build opencv-python sklearn -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
pip list


```
