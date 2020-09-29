	#wzj751127122/caffe:19.12-py3
sudo docker run \
	--gpus=all\
	-v $PWD/../d:/d -v $PWD:/w -w /w\
	-ti szubdi/caffe\
	$*
