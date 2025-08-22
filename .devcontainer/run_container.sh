#!/bin/bash

CONTAINER_NAME="fideslib_container"
TAG="latest"
IMAGE_NAME="fideslib"
USER=$(whoami)

# change DIR to match your local directory
DIR="/home/$USER/fides"


docker run --privileged --gpus all -itd --rm --group-add=video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --device /dev/kfd --device /dev/dri \
    --name $CONTAINER_NAME --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v $DIR:/app/fides/ \
    $IMAGE_NAME:$TAG && \

docker exec -it $CONTAINER_NAME bash

