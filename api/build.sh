#! /bin/bash

docker build -t comnist .
docker tag comnist gregvi/comnist:latest

