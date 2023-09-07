sudo xhost +si:localuser:root
sudo docker run --gpus all -it \
		--name yolov7 \
		--network host \
		-e DISPLAY=$DISPLAY \
		-v /tmp/.X11-unix/:/tmp/.X11-unix \
		-v /tmp/argus_socket:/tmp/argus_socket \
		-v /etc/enctune.conf:/etc/enctune.conf \
		--device /dev/video2:/dev/video2 --device /dev/video3:/dev/video3  \
		-v $PWD/:/workspace/ nvcr.io/nvidia/pytorch:22.08-py3 bash
