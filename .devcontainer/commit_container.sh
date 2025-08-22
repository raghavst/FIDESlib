#!/bin/bash

CONTAINER_NAME="fideslib_container"
IMAGE_NAME="fideslib"
TAG="latest"

# Commit the container to the image
docker commit $CONTAINER_NAME $IMAGE_NAME:$TAG && \
docker stop $CONTAINER_NAME -t 1
