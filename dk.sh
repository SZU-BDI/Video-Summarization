	#wzj751127122/caffe:19.12-py3
CMD="sudo docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all -v $PWD/../d:/d -v $PWD:/w -w /w -ti szubdi/caffe"
echo $CMD
$CMD $*
