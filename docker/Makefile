help:
	@cat Makefile

DATA?="${HOME}/shared_data"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
TEST=tests/
SRC=$(shell dirname `pwd`)

build:
	docker build -t yordan --build-arg python_version=2.7.13 -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run -it --rm -v $(DATA):/data -p 8888:8888 --env KERAS_BACKEND=$(BACKEND) yordan bash

ipython: build
	$(DOCKER) run -it --rm -v $(SRC):/src -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) yordan ipython

notebook: build
	$(DOCKER) run -it --rm -v $(SRC):/src -v $(DATA):/data --net=host --env KERAS_BACKEND=$(BACKEND) yordan

test: build
	$(DOCKER) run -it --rm -v $(SRC):/src -v $(DATA):/data --env KERAS_BACKEND=$(BACKEND) yordan py.test $(TEST)

