# Docker-images

This set contains a basic version of a docker container for running ML using

* Python 2.7
* Tensorflow
* Keras
* Autograd
* PyMC3
* Chainer
* OpenCV
* scikit-learn
* Notebooks

It allows for the frameworks to run on the GPUs.

Based on the `keras` docker image, the starting of the docker containers are encapsulated with a makefile.

To gain terminal access to the image run

```
make bash
```

A shared folder will be created under `/home/${user}/shared_data` on the host machine, which is linked to `/data/` on the container. Any files stored there will be persistent across start/stop of the container.

Make sure that the host folder allows for other user to create/modify files in the folder and any folders in it.

