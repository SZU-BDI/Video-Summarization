CMD="docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD/../d:/d -v $PWD:/w -w /w -ti szubdi/caffe"
echo $CMD
$CMD $*
