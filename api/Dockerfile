FROM ubuntu:latest

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN	apt-get update -y && \
	apt-get install -y python python-pip curl && \
	apt-get clean all

RUN	apt-get install supervisor -y

RUN	pip install --upgrade pip && \
	pip install Pillow && \
	pip install flask && \
	pip install flask_restful && \
	pip install numpy && \
	pip install requests && \
	pip install tensorflow && \
	pip install keras && \
	pip install requests && \
	pip install pandas && \
	pip install h5py

COPY *.py /app/
COPY weights/* /app/weights/
COPY assets/* /app/assets/
WORKDIR /app

# default command
CMD python word.py
