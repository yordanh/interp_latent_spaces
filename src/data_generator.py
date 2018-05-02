import os
import cv2
import numpy as np
import chainer
from config_parser import ConfigParser

class DataGenerator(object):
    def __init__(self, label_mode=None):
        self.label_mode = label_mode

    def generate_dataset(self, args):
        if args.data == "mnist":
            # Load the MNIST dataset
            # train, test = chainer.datasets.get_mnist(withlabel=False)
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

            folder_name = "data/dSprites/"
            data_split = 0.8
            image_size = 100

            config_parser = ConfigParser("config/config.json")
            labels = config_parser.parse_labels()
            groups = config_parser.parse_groups()
            folders_for_training = labels["train"]
            folders_for_unseen = labels["unseen"]

            train = []
            train_labels = []
            train_vectors = []
            test = []
            test_labels = []
            test_vectors = []
            unseen = []
            unseen_labels = []
            unseen_vectors = []

            folder_list = os.listdir(folder_name)

            print(folders_for_training)
            print(groups)
            print(folder_list)


            # big, blue, flat are 3 folders with overlapping images
            if args.labels == "singular":
                for folder in folder_list:
                    image_list = os.listdir(folder_name+folder)
                    number_of_images = len(image_list)
                    index = folder_list.index(folder)

                    print("Processing folder {0}/{1} with {2} images".format(folder_list.index(folder), 
                                                                             len(folder_list), 
                                                                             len(image_list)))

                    if folder in folders_for_training:
                        for image_name in image_list[0:int(data_split * number_of_images)]:
                                train.append(cv2.imread(folder_name+folder+"/"+image_name, 1))
                        train_labels += [folder] * (int(data_split * number_of_images))
                        n = int(data_split * number_of_images)
                        train_vectors += list(np.tile(folders_for_training.index(folder), (n)))

                        for image_name in image_list[int(data_split * number_of_images):]:
                            test.append(cv2.imread(folder_name+folder+"/"+image_name, 1))#
                        test_labels += [folder] * (number_of_images - int(data_split * number_of_images))
                        n = number_of_images - int(data_split * number_of_images)
                        test_vectors += list(np.tile(folders_for_training.index(folder), (n)))

                    if folder in folders_for_unseen:
                        for image_name in image_list[int(data_split * number_of_images):]:
                            unseen.append(cv2.imread(folder_name+folder+"/"+image_name, 1))#
                        unseen_labels += [folder] * (number_of_images - int(data_split * number_of_images))
                        n = number_of_images - int(data_split * number_of_images)
                        unseen_vectors += list(np.tile(folders_for_unseen.index(folder), (n)))
            
            # big_blue_flat is one folder with unique images
            elif args.labels == "composite":

                train_vectors = [[] for group in groups]
                test_vectors = [[] for group in groups]

                for folder in folder_list:
                    image_list = os.listdir(folder_name+folder)
                    number_of_images = len(image_list)
                    index = folder_list.index(folder)

                    print("Processing folder {0}/{1} with {2} images".format(folder_list.index(folder), 
                                                                             len(folder_list), 
                                                                             len(image_list)))

                    # if folder in folders_for_training:
                    for image_name in image_list[0:int(data_split * number_of_images)]:
                            train.append(cv2.imread(folder_name+folder+"/"+image_name, 1))

                    train_labels += folder.split('_') * (int(data_split * number_of_images))

                    n = int(data_split * number_of_images)
                    for i, group in enumerate(groups):
                        label = filter(lambda x : x in groups[str(i)], folder.split('_'))
                        label = label[0]
                        train_vectors[i] += list(np.tile(groups[str(i)].index(label), (n)))

                    for image_name in image_list[int(data_split * number_of_images):]:
                        test.append(cv2.imread(folder_name+folder+"/"+image_name, 1))

                    test_labels += folder.split('_') * (number_of_images - int(data_split * number_of_images))

                    n = number_of_images - int(data_split * number_of_images)
                    for i, group in enumerate(groups):
                        label = filter(lambda x : x in groups[str(i)], folder.split('_'))
                        label = label[0]
                        test_vectors[i] += list(np.tile(groups[str(i)].index(label), (n)))

                    # if folder in folders_for_unseen:
                    #     for image_name in image_list[int(data_split * number_of_images):]:
                    #         unseen.append(cv2.imread(folder_name+folder+"/"+image_name, 1))#
                    #     unseen_labels += [folder] * (number_of_images - int(data_split * number_of_images))
                    #     n = number_of_images - int(data_split * number_of_images)
                    #     unseen_vectors += list(np.tile(folders_for_unseen.index(folder), (n)))

            # print("Train Vectors: {}".format(np.array(train_vectors)))
            # print("Train Labels: {}".format(np.array(train_labels)))

            # print("Test Vectors: {}".format(np.array(test_vectors)))
            # print("Test Labels: {}".format(np.array(test_labels)))

            train = np.array(train)
            train_labels = np.array(train_labels)
            train_vectors = np.array(train_vectors)
            test = np.array(test)
            test_labels = np.array(test_labels)
            test_vectors = np.array(test_vectors)
            unseen = np.array(unseen)
            unseen_labels = np.array(unseen_labels)
            unseen_vectors = np.array(unseen_vectors)
            train = train.astype('float32') / 255.
            test = test.astype('float32') / 255.
            unseen = unseen.astype('float32') / 255.

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
            train_concat = zip(train, train_vectors[0], train_vectors[1])
            test_concat = zip(test, test_vectors[0], test_vectors[1])
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