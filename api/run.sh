#! /bin/bash

docker rm cn1 -f
docker run --name cn1 -p 5002:5002 -t comnist
