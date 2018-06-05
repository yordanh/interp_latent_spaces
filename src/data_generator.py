#!/usr/bin/env python
"""
title           :data_generator.py
description     :Loads the image dataset contained in train,unseen,ulabelled folders under data/.
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :05/2018
python_version  :2.7.14
==============================================================================
"""

import os
import cv2
import numpy as np

import chainer

from config_parser import ConfigParser

class DataGenerator(object):
    def __init__(self, label_mode=None):
        self.label_mode = label_mode

    def generate_dataset(self, ignore=[], args=None):
        if args.data == "mnist":
            # Load the MNIST dataset
            train, test = chainer.datasets.get_mnist()

            data_dimensions = [28,28]

            desired_classes_train = [0, 1, 2]
            desired_classes_test = [0, 1, 2]

            train = np.array([x[0] for x in train if x[1] in desired_classes_train])
            test_labels = np.array([x[1] for x in test if x[1] in desired_classes_test])
            test = np.array([x[0] for x in test if x[1] in desired_classes_test])

            if args.model == "conv":
                train = train.reshape(len(train), 28, 28, 1)
                test = test.reshape(len(test), 28, 28, 1)
                train = np.swapaxes(train, 1 ,3)
                test = np.swapaxes(test, 1 ,3)  
            else:
                train = train.reshape(len(train), 784)
                test = test.reshape(len(test), 784)        

        elif args.data == "sprites":

            data_dimensions = [100, 100, 3]

            folder_name_train = "data/dSprites/train/"
            folder_name_unseen = "data/dSprites/unseen/"
            folder_name_unlabelled = "data/dSprites/unlabelled/"
            data_split = 0.8
            image_size = 100

            config_parser = ConfigParser("config/config.json")
            labels = config_parser.parse_labels()
            groups = config_parser.parse_groups()

            train = []
            train_labels = []
            train_vectors = []
            train_masks = []
            test = []
            test_labels = []
            test_vectors = []
            test_masks = []
            unseen = []
            unseen_labels = []
            unseen_vectors = []

            folder_list_train = os.listdir(folder_name_train)

            # big, blue, flat are 3 folders with overlapping images
            if args.labels == "singular":
                for folder in folder_list_train:
                    image_list = os.listdir(folder_name+folder)
                    number_of_images = len(image_list)
                    train_n = int(data_split * number_of_images)
                    test_n = number_of_images - train_n
                    train_indecies = np.random.choice(range(number_of_images), train_n, replace=False)
                    test_indecies = filter(lambda x : x not in train_indecies, range(number_of_images))

                    print("Processing TRAINING folder {0}/{1} with {2} images".format(folder_list_train.index(folder), 
                                                                                      len(folder_list_train), 
                                                                                      len(image_list)))

                    if folder in folders_for_training:
                        for image_name in np.take(image_list, train_indecies, axis=0):
                                train.append(cv2.imread(folder_name+folder+"/"+image_name, 1))
                        train_labels += [folder] * test_n
                        train_vectors += list(np.tile(folders_for_training.index(folder), (train_n)))

                        for image_name in np.take(image_list, test_indecies, axis=0):
                            test.append(cv2.imread(folder_name+folder+"/"+image_name, 1))#
                        test_labels += [folder] * test_n
                        test_vectors += list(np.tile(folders_for_training.index(folder), (test_n)))

                if os.path.exists(folder_name_unseen):
                    folder_list_unseen = os.listdir(folder_name_unseen)
                    for folder in folder_list_unseen:
                        image_list = os.listdir(folder_name_unseen+folder)
                        number_of_images = len(image_list)
                        unseen_n = test_n
                        unseen_indecies = np.random.choice(range(number_of_images), unseen_n, replace=False)                    

                        print("Processing UNSEEN folder {0}/{1} with {2} images".format(folder_list_unseen.index(folder), 
                                                                                        len(folder_list_unseen), 
                                                                                        len(image_list)))

                        for image_name in np.take(image_list, unseen_indecies, axis=0):
                            unseen.append(cv2.imread(folder_name_unseen+folder+"/"+image_name, 1))
                        unseen_labels += [folder] * unseen_n
                        unseen_vectors += list(np.tile(folders_for_unseen.index(folder), (unseen_n)))
            
            # big_blue_flat is one folder with unique images
            elif args.labels == "composite":

                train_vectors = [[] for group in groups]
                test_vectors = [[] for group in groups]
                unseen_vectors = [[] for group in groups]


                for folder in folder_list_train:
                    image_list = os.listdir(folder_name_train+folder)
                    number_of_images = len(image_list)
                    train_n = int(data_split * number_of_images)
                    test_n = number_of_images - train_n
                    train_indecies = np.random.choice(range(number_of_images), train_n, replace=False)
                    test_indecies = filter(lambda x : x not in train_indecies, range(number_of_images))

                    print("Processing TRAINING folder {0}/{1} with {2} images".format(folder_list_train.index(folder), 
                                                                                      len(folder_list_train), 
                                                                                      len(image_list)))

                    for image_name in np.take(image_list, train_indecies, axis=0):
                            train.append(cv2.imread(folder_name_train+folder+"/"+image_name, 1))

                    train_labels += folder.split('_') * train_n
                    train_masks += [1] * train_n

                    for i, group in enumerate(groups):
                        label = filter(lambda x : x in groups[str(i)], folder.split('_'))
                        label = label[0]
                        train_vectors[i] += list(np.tile(groups[str(i)].index(label), (train_n)))

                    for image_name in np.take(image_list, test_indecies, axis=0):
                        test.append(cv2.imread(folder_name_train+folder+"/"+image_name, 1))

                    test_labels += folder.split('_') * test_n
                    test_masks += [1] * test_n

                    for i, group in enumerate(groups):
                        label = filter(lambda x : x in groups[str(i)], folder.split('_'))
                        label = label[0]
                        test_vectors[i] += list(np.tile(groups[str(i)].index(label), (test_n)))

                # unlabelled datapoints
                if os.path.exists(folder_name_unlabelled) and "unlabelled" not in ignore:
                    folder_list_unlabelled = os.listdir(folder_name_unlabelled)
                    for folder in folder_list_unlabelled:
                        image_list = os.listdir(folder_name_unlabelled + folder)
                        number_of_images = len(image_list)
                        train_n = int(data_split * number_of_images)
                        test_n = number_of_images - train_n
                        train_indecies = np.random.choice(range(number_of_images), train_n, replace=False)
                        test_indecies = filter(lambda x : x not in train_indecies, range(number_of_images))

                        print("Processing UNLABELLED folder {0}/{1} with {2} images".format(folder_list_unlabelled.index(folder), 
                                                                                            len(folder_list_unlabelled), 
                                                                                            len(image_list)))

                        for image_name in np.take(image_list, train_indecies, axis=0):
                                train.append(cv2.imread(folder_name_unlabelled+folder+"/"+image_name, 1))

                        train_labels += folder.split('_') * train_n
                        train_masks += [0] * train_n

                        for i, group in enumerate(groups):
                            label = 0
                            train_vectors[i] += list(np.tile(label, (train_n)))

                        for image_name in np.take(image_list, test_indecies, axis=0):
                            test.append(cv2.imread(folder_name_unlabelled+folder+"/"+image_name, 1))

                        test_labels += folder.split('_') * test_n
                        test_masks += [0] * test_n

                        for i, group in enumerate(groups):
                            label = 0
                            test_vectors[i] += list(np.tile(label, (test_n)))

                # unseen datapoints
                if os.path.exists(folder_name_unseen) and "unseen" not in ignore:
                    folder_list_unseen = os.listdir(folder_name_unseen)
                    for folder in folder_list_unseen:
                        image_list = os.listdir(folder_name_unseen+folder)
                        number_of_images = len(image_list)
                        unseen_n = test_n
                        unseen_indecies = np.random.choice(range(number_of_images), unseen_n, replace=False)                                        

                        print("Processing UNSEEN folder {0}/{1} with {2} images".format(folder_list_unseen.index(folder), 
                                                                                        len(folder_list_unseen), 
                                                                                        len(image_list)))

                        for image_name in np.take(image_list, unseen_indecies, axis=0):
                            unseen.append(cv2.imread(folder_name_unseen+folder+"/"+image_name, 1))#
                        
                        unseen_labels += folder.split('_') * unseen_n

                        for i, group in enumerate(groups):   
                            label = 0
                            unseen_vectors[i] += list(np.tile(label, (unseen_n)))

            # print("Train Vectors: {}".format(np.array(train_vectors)))
            # print("Train Labels: {}".format(np.array(train_labels)))
            # print("Train Masks: {}".format(np.array(train_masks)))

            # print("Test Vectors: {}".format(np.array(test_vectors)))
            # print("Test Labels: {}".format(np.array(test_labels)))
            # print("Test Masks: {}".format(np.array(test_masks)))

            # print("Unseen Vectors: {}".format(np.array(unseen_vectors)))
            # print("Unseen Labels: {}".format(np.array(unseen_labels)))

            train = np.array(train, dtype=np.float32) / 255.
            train_labels = np.array(train_labels)
            train_vectors = np.array(train_vectors)
            train_masks = np.array(train_masks)

            test = np.array(test, dtype=np.float32) / 255.
            test_labels = np.array(test_labels)
            test_vectors = np.array(test_vectors)
            test_masks = np.array(test_masks)
            
            unseen = np.array(unseen, dtype=np.float32) / 255.
            unseen_labels = np.array(unseen_labels)
            unseen_vectors = np.array(unseen_vectors)

            # adapt this if using `channels_first` image data format
            if args.model == "conv":
                train = np.reshape(train, (len(train), image_size, image_size, 3))
                test = np.reshape(test, (len(test), image_size, image_size, 3))
                unseen = np.reshape(unseen, (len(unseen), image_size, image_size, 3))
                train = np.swapaxes(train, 1 ,3)
                test = np.swapaxes(test, 1 ,3)
                unseen = np.swapaxes(unseen, 1 ,3)
            else:
                train = np.reshape(train, (len(train), image_size * image_size * 3))
                test = np.reshape(test, (len(test), image_size * image_size * 3))
                unseen = np.reshape(unseen, (len(unseen), image_size * image_size * 3))

        #augment the training, testing, unseen datapoints with their labels
        if args.labels == "composite":
            train_concat = zip(train, train_vectors[0], train_vectors[1], train_masks)
            test_concat = zip(test, test_vectors[0], test_vectors[1], test_masks)
            if len(unseen) != 0:
                unseen_concat = zip(unseen, unseen_vectors[0], unseen_vectors[1])
            else:
                unseen_concat = zip(unseen, unseen_vectors)    
        elif args.labels == "singular":
            train_concat = zip(train, train_vectors)
            test_concat = zip(test, test_vectors)
            unseen_concat = zip(unseen, unseen_vectors)

        # print("Train Concat: {}".format(np.array(train_concat[0])))
        # print("Test Concat: {}".format(np.array(test_concat[0])))

        result = []
        result.append(train)
        result.append(train_labels)
        result.append(train_concat)
        result.append(train_vectors)

        result.append(test)
        result.append(test_labels)
        result.append(test_concat)
        result.append(test_vectors)

        result.append(unseen)
        result.append(unseen_labels)
        result.append(unseen_concat)
        result.append(unseen_vectors)

        result.append(groups)

        return result