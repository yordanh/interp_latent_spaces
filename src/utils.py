#!/usr/bin/env python
"""
title           :utils.py
description     :Utility functions to be used for result processing after the model training phase.
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :05/2018
python_version  :2.7.14
==============================================================================
"""

import os
import numpy as np
from scipy.stats import multivariate_normal
import cv2
from scipy.stats import norm
from math import sqrt
import itertools
import  copy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import shutil

import chainer
import chainer.functions as F

########################################
############ UTIL FUNCTIONS ############
########################################

# delete all result files from the output folder
def clear_last_results(folder_name=None):
    all_files = list(filter(lambda filename : '.' in filename, os.listdir(folder_name)))
    map(lambda x : os.remove(folder_name + x), all_files)

    leftover_folders = list(filter(lambda filename : filename != "models", os.listdir(folder_name)))
    map(lambda x : shutil.rmtree(folder_name + x), leftover_folders)

    os.mkdir(folder_name + "gifs")
    os.mkdir(folder_name + "gifs/manifold_gif")
    os.mkdir(folder_name + "gifs/scatter_gif")
    os.mkdir(folder_name + "scatter")
    os.mkdir(folder_name + "eval")


# for a given set of example images, calculate their reconstructions
def perform_reconstructions(model=None, train=None, test=None, unseen=None, no_images=None, name_suffix=None, args=None):
    train_ind = np.linspace(0, len(train) - 1, no_images, dtype=int)
    x = chainer.Variable(np.asarray(train[train_ind]))
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x1 = model(x)
    save_images(x.data, no_images, os.path.join(args.out, 'train_' + name_suffix), args=args)
    save_images(x1.data, no_images, os.path.join(args.out, 'train_' + name_suffix + "_rec"), args=args)

    # reconstruct testing examples
    test_ind = np.linspace(0, len(test) - 1, no_images, dtype=int)
    x = chainer.Variable(np.asarray(test[test_ind]))
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x1 = model(x)
    save_images(x.data, no_images, os.path.join(args.out, 'test_' + name_suffix), args=args)
    save_images(x1.data, no_images, os.path.join(args.out, 'test_' + name_suffix + "_rec"), args=args)

    # reconstruct unseen examples
    if len(unseen) != 0:
        unseen_ind = np.linspace(0, len(unseen) - 1, no_images, dtype=int)
        x = chainer.Variable(np.asarray(unseen[unseen_ind]))
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            x1 = model(x)
        save_images(x.data, no_images, os.path.join(args.out, 'unseen_' + name_suffix), args=args)
        save_images(x1.data, no_images, os.path.join(args.out, 'unseen_' + name_suffix + "_rec"), args=args)

    # draw images from randomly sampled z under a 'vanilla' normal distribution
    z = chainer.Variable(
        np.random.normal(0, 1, (no_images, args.dimz)).astype(np.float32))
    x = model.decode(z)
    save_images(x.data, no_images, os.path.join(args.out, 'sampled_' + name_suffix), args=args)


# plot and save loss and accuracy curves
def plot_loss_curves(stats=None, args=None):
    # overall train/validation losses
    plt.figure(figsize=(10, 10))
    plt.grid()
    colors = ['r', 'k', 'b', 'g', 'gold']
    for i, channel in enumerate(stats):
        if channel == "valid_label_loss" or channel == "valid_label_acc" or channel == "train_accs":
            continue
        plt.plot(range(args.epoch_labelled + args.epoch_unlabelled),stats[channel], marker='x', color=colors[i], label=channel)
    plt.xlabel("Epoch #", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=14)
    plt.savefig(os.path.join(args.out + "losses"), bbox_inches="tight")
    plt.close()

    # validation label loss
    plt.figure(figsize=(10, 10))
    plt.grid()
    plt.plot(range(args.epoch_labelled + args.epoch_unlabelled),stats['valid_label_loss'], marker='x', color='g', label='valid_label_loss')
    plt.xlabel("Epoch #", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=14)
    plt.savefig(os.path.join(args.out + "label_loss"), bbox_inches="tight")
    plt.close()

    # validation label accuracy
    plt.figure(figsize=(10, 10))
    plt.grid()
    plt.plot(range(args.epoch_labelled + args.epoch_unlabelled),stats['valid_label_acc'], marker='x', color='r', label='valid_label_acc')
    plt.xlabel("Epoch #", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=14)
    plt.savefig(os.path.join(args.out + "valid_label_acc"), bbox_inches="tight")
    plt.close()

    # training label accuracy
    plt.figure(figsize=(10, 10))
    plt.grid()
    plt.plot(range(args.epoch_labelled + args.epoch_unlabelled),stats['train_accs'], marker='x', color='b', label='train_accs')
    plt.xlabel("Epoch #", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=14)
    plt.savefig(os.path.join(args.out + "train_label_acc"), bbox_inches="tight")
    plt.close()


# calculate statistics for the predicted labels
def compare_labels(test=None, test_labels=None, model=None, args=None, cuttoff_thresh=1):

    mu, ln_var = model.encode(test)
    
    if args.labels == "composite":
        hat_labels_0, hat_labels_1 = model.predict_label(mu, ln_var, softmax=True)

        print("\nValidation Accuracy for group 0: {0}\n".format(F.accuracy(hat_labels_0, test_labels[0])))
        print("\nValidation Accuracy for group 1: {0}\n".format(F.accuracy(hat_labels_1, test_labels[1])))
    elif args.labels == "singular":
        hat_labels = model.predict_label(mu, ln_var, softmax=True)

        print("Validation Accuracy: {0}\n".format(F.accuracy(hat_labels, test_labels)))


# visualize the results
def save_images(x=None, no_images=None, filename=None, args=None):

    fig, ax = plt.subplots(int(sqrt(no_images)), int(sqrt(no_images)), figsize=(9, 9), dpi=100)
    for ai, xi in zip(ax.flatten(), x):
        if args.model == "conv":
            xi = np.swapaxes(xi, 0, 2)
        else:
            if args.data == "mnist":
                xi = xi.reshape(28, 28)
            else:
                xi = xi.reshape(100, 100, 3)
        
        if xi.shape[-1] == 1:
            xi = xi.reshape(xi.shape[:-1])

        ai.set_xticks([])
        ai.set_yticks([])
        image = ai.imshow(cv2.cvtColor(xi, cv2.COLOR_BGR2RGB))
    fig.savefig(filename)


# attach a color to each singular and composite class labels, both for their data points 
# and fitted overlayed distributions
def attach_colors(labels=None, composite=True):

    colors = ['c', 'b', 'g', 'y', 'k', 'orange', 'maroon', 'lime', 'salmon', 
              'crimson', 'gold', 'coral', 'navy', 'purple', 'olive', 'r', 'yellowgreen', 'brown']
    result = {"singular":{}, "composite":{}}

    counter = 0
    for label in sorted(set(labels)):
        if label in result["singular"]:
            continue
        else:
            result["singular"][label] = {}
            result["singular"][label]["data"] = colors[counter]
            result["singular"][label]["dist"] = colors[counter + 1]
            counter += 1

    result["singular"]["unknown"] = {}
    result["singular"]["unknown"]["dist"] = colors[-1]

    if composite:
        labels = labels.reshape(len(labels) / 2, 2)
        labels = np.array(["_".join(x) for x in labels])

        counter = 0
        for label in sorted(set(labels)):
            if label in result["composite"]:
                continue
            else:
                result["composite"][label] = {}
                result["composite"][label]["data"] = colors[counter]
                result["composite"][label]["dist"] = colors[counter + 1]
                counter += 1

    return result


# plot a set of input datapoitns to the latent space and fit a normal distribution over the projections
# show the contours for the overall data distribution
def plot_overall_distribution(data=None, labels=None, boundaries=None, colors=None, model=None, 
                              overlay=True, spread=False, filename=None):
    latent_all = None

    # scatter plot all the data points in the latent space
    plt.figure(figsize=(10, 10))
    concise_colors = list(set(labels))
    for label in sorted(set(labels)):
        indecies = [i for i, x in enumerate(labels) if x == label]
        filtered_data = chainer.Variable(data.take(indecies, axis=0))
        if spread:
            latent = model.get_latent(filtered_data)
        else:
            latent = model.get_latent_mu(filtered_data)
        latent = latent.data
        plt.scatter(latent[:, 0], latent[:, 1], c=colors[label]["data"], label=str(label), alpha=0.75)

        if latent_all is not None:
            latent_all = np.append(latent_all, latent, axis=0)
        else:
            latent_all = latent
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

    # plot bounding box for the visualised manifold
    # boundaries are [[min_x, min_y],[max_x, max_y]]
    plt.plot([boundaries[0,0], boundaries[1,0]], [boundaries[0,1], boundaries[0,1]], 'r') # top line
    plt.plot([boundaries[0,0], boundaries[1,0]], [boundaries[1,1], boundaries[1,1]], 'r') # bottom line
    plt.plot([boundaries[0,0], boundaries[0,0]], [boundaries[0,1], boundaries[1,1]], 'r') # left line
    plt.plot([boundaries[1,0], boundaries[1,0]], [boundaries[0,1], boundaries[1,1]], 'r') # right line
    # major axes
    plt.plot([boundaries[0,0], boundaries[1,0]], [0,0], 'k')
    plt.plot([0,0], [boundaries[0,1], boundaries[1,1]], 'k')

    plt.grid()
    plt.savefig(filename, bbox_inches="tight")
    
    if overlay:
        # fit and plot a distribution over all the latent projections
        delta = 0.025
        mean = np.mean(latent_all, axis=0)
        cov = np.cov(latent_all.T)
        x = np.arange(min(latent_all[:, 0]), max(latent_all[:, 0]), delta)
        y = np.arange(min(latent_all[:, 1]), max(latent_all[:, 1]), delta)
        X, Y = np.meshgrid(x, y)
        Z = multivariate_normal.pdf(np.array([zip(c,d) for c,d in zip(X,Y)]), mean=mean, cov=cov)
        plt.contour(X, Y, Z, colors='r')
        plt.title("mu[0]:{0}; mu[1]:{1}\ncov[0,0]:{2}; cov[1,1]:{3}\ncov[0,1]:{4}".format(round(mean[0],2), 
                  round(mean[1],2), round(cov[0,0],2), round(cov[1,1],2), round(cov[0,1],2)), fontweight="bold", fontsize=14)
        plt.savefig(filename + "_overlayed", bbox_inches="tight")
    plt.close()


# plot a set of input datapoitns to the latent space and fit normal distributions over the projections
# show the contours for the distribution for each label
def plot_separate_distributions(data=None, labels=None, groups=None, boundaries=None, 
                                colors=None, model=None, filename=None, overlay=True):
    latent_all = []

    # scatter plot all the data points in the latent space
    plt.figure(figsize=(10, 10))
    for label in set(labels):
        indecies = [i for i, x in enumerate(labels) if x == label]
        filtered_data = chainer.Variable(data.take(indecies, axis=0))
        latent = model.get_latent(filtered_data)
        latent = latent.data
        latent_all.append(latent)
        plt.scatter(latent[:, 0], latent[:, 1], c=colors[label]["data"], label=str(label), alpha=0.75)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

    # plot bounding box for the visualised manifold
    # boundaries are [[min_x, min_y],[max_x, max_y]]
    plt.plot([boundaries[0,0], boundaries[1,0]], [boundaries[0,1], boundaries[0,1]], 'r') # top line
    plt.plot([boundaries[0,0], boundaries[1,0]], [boundaries[1,1], boundaries[1,1]], 'r') # bottom line
    plt.plot([boundaries[0,0], boundaries[0,0]], [boundaries[0,1], boundaries[1,1]], 'r') # left line
    plt.plot([boundaries[1,0], boundaries[1,0]], [boundaries[0,1], boundaries[1,1]], 'r') # right line
    # major axes
    plt.plot([boundaries[0,0], boundaries[1,0]], [0,0], 'k')
    plt.plot([0,0], [boundaries[0,1], boundaries[1,1]], 'k')

    plt.grid()
    plt.savefig(filename, bbox_inches="tight")

    # fit and overlay distributions for each class/label
    if overlay:
        counter = 0
        for label in set(labels):
            latent = latent_all[counter]
            counter += 1

            delta = 0.025
            mean = np.mean(latent, axis=0)
            cov = np.cov(latent.T)
            x = np.arange(min(latent[:, 0]), max(latent[:, 0]), delta)
            y = np.arange(min(latent[:, 1]), max(latent[:, 1]), delta)
            X, Y = np.meshgrid(x, y)
            Z = multivariate_normal.pdf(np.array([zip(c,d) for c,d in zip(X,Y)]), mean=mean, cov=cov)
            plt.contour(X, Y, Z, colors=colors[label]["dist"])
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
        plt.savefig(filename + "_overlayed", bbox_inches="tight")
    plt.close()

    # plot the per-group disttibutions
    if groups is not None:
        for key in groups:
            plt.figure(figsize=(10, 10))
            labels_group = labels[int(key)::2]
            for label in set(labels_group):
                indecies = [i for i, x in enumerate(labels) if x == label]
                filtered_data = chainer.Variable(data.take(indecies, axis=0))
                latent = model.get_latent(filtered_data)
                latent = latent.data

                plt.scatter(latent[:, 0], latent[:, 1], c=colors[label]["data"], label=str(label), alpha=0.75)

                if overlay:
                    delta = 0.025
                    mean = np.mean(latent, axis=0)
                    cov = np.cov(latent.T)
                    x = np.arange(min(latent[:, 0]), max(latent[:, 0]), delta)
                    y = np.arange(min(latent[:, 1]), max(latent[:, 1]), delta)
                    X, Y = np.meshgrid(x, y)
                    Z = multivariate_normal.pdf(np.array([zip(c,d) for c,d in zip(X,Y)]), mean=mean, cov=cov)
                    plt.contour(X, Y, Z, colors=colors[label]["dist"])
            
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

            # plot bounding box for the visualised manifold
            # boundaries are [[min_x, min_y],[max_x, max_y]]
            plt.plot([boundaries[0,0], boundaries[1,0]], [boundaries[0,1], boundaries[0,1]], 'r') # top line
            plt.plot([boundaries[0,0], boundaries[1,0]], [boundaries[1,1], boundaries[1,1]], 'r') # bottom line
            plt.plot([boundaries[0,0], boundaries[0,0]], [boundaries[0,1], boundaries[1,1]], 'r') # left line
            plt.plot([boundaries[1,0], boundaries[1,0]], [boundaries[0,1], boundaries[1,1]], 'r') # right line
            # major axes
            plt.plot([boundaries[0,0], boundaries[1,0]], [0,0], 'k')
            plt.plot([0,0], [boundaries[0,1], boundaries[1,1]], 'k')

            plt.grid()
            plt.savefig(filename + "_group_" + key + '_overlayed', bbox_inches="tight")
            plt.close()

    # scatter datapoints and fit and overlay a distribution over each data label
    counter = 0
    for label in set(labels):
        plt.figure(figsize=(10, 10))
        latent = latent_all[counter]
        plt.scatter(latent[:, 0], latent[:, 1], c=colors[label]["data"], label=str(label), alpha=0.75)

        if overlay:
            delta = 0.025
            mean = np.mean(latent, axis=0)
            cov = np.cov(latent.T)
            x = np.arange(min(latent[:, 0]), max(latent[:, 0]), delta)
            y = np.arange(min(latent[:, 1]), max(latent[:, 1]), delta)
            X, Y = np.meshgrid(x, y)
            Z = multivariate_normal.pdf(np.array([zip(c,d) for c,d in zip(X,Y)]), mean=mean, cov=cov)
            plt.contour(X, Y, Z, colors=colors[label]["dist"])
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

        # plot bounding box for the visualised manifold
        # boundaries are [[min_x, min_y],[max_x, max_y]]
        plt.plot([boundaries[0,0], boundaries[1,0]], [boundaries[0,1], boundaries[0,1]], 'r') # top line
        plt.plot([boundaries[0,0], boundaries[1,0]], [boundaries[1,1], boundaries[1,1]], 'r') # bottom line
        plt.plot([boundaries[0,0], boundaries[0,0]], [boundaries[0,1], boundaries[1,1]], 'r') # left line
        plt.plot([boundaries[1,0], boundaries[1,0]], [boundaries[0,1], boundaries[1,1]], 'r') # right line
        # major axes
        plt.plot([boundaries[0,0], boundaries[1,0]], [0,0], 'k')
        plt.plot([0,0], [boundaries[0,1], boundaries[1,1]], 'k')

        plt.grid()
        plt.savefig(filename + "_overlayed" + "_" + str(counter), bbox_inches="tight")
        plt.close()
        counter += 1

# sample datapoints under the prior normal distribution and reconstruct
# samples_per_dimension has to be even
def plot_sampled_images(model=None, data=None, boundaries=None, samples_per_dimension=16, 
                        image_size=100, offset=10, image_channels=3, filename=None, figure_title=None):
        
        n_latent = model.n_latent

        dimensions_pairs = list(itertools.combinations(range(n_latent), 2))
        for pair in dimensions_pairs:

            rows = image_size * samples_per_dimension + offset * samples_per_dimension
            columns = image_size * samples_per_dimension + offset * samples_per_dimension
            figure = np.ones((rows, columns, image_channels))
            # major axes
            if image_channels == 1:
                line_pixel = [1]
            else:
                line_pixel = [0,0,1]
            quadrant_size = (samples_per_dimension / 2) * image_size + ((samples_per_dimension / 2) - 1) * offset
            figure[quadrant_size : quadrant_size + offset, :, :] = np.tile(line_pixel, (offset, (quadrant_size + offset) * 2, 1))
            figure[:, quadrant_size : quadrant_size + offset, :] = np.tile(line_pixel, ((quadrant_size + offset) * 2, offset, 1))
            
            # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
            # to produce values of the latent variables z, since the prior of the latent space is Gaussian
            # x and y are sptlit because of the way open cv has its axes
            grid_x = np.linspace(boundaries[1,0], boundaries[0,0], samples_per_dimension)
            grid_y = np.linspace(boundaries[0,1], boundaries[1,1], samples_per_dimension)

            for i, yi in enumerate(grid_x):
                for j, xi in enumerate(grid_y):
                    z_sample = np.zeros((1, n_latent))
                    z_sample[0, pair[0]] = xi
                    z_sample[0, pair[1]] = yi
                    z_sample = np.array([[z_sample]]).astype(np.float32)
                    x_decoded = model.decode(chainer.Variable(z_sample)).data
                    image_sample = x_decoded.reshape(x_decoded.shape[1:])
                    image_sample = np.swapaxes(image_sample, 0, 2)
                    image_sample = image_sample.reshape(100, 100, 3)

                    figure[i * image_size + i * offset: (i + 1) * image_size + i * offset,
                           j * image_size + j * offset: (j + 1) * image_size + j * offset,
                           :] = image_sample

            figure = np.array(figure*255, dtype=np.uint8)

            plt.figure(figsize=(15,15))
            image = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
            if figure_title:
                plt.title(figure_title, fontsize=20)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('Z' + str(pair[0]), fontsize=20)
            plt.ylabel('Z' + str(pair[1]), fontsize=20)
            plt.savefig(filename + '_Z' + str(pair[0]) + '_Z' + str(pair[1]), bbox_inches="tight")
            plt.close()

########################################
############# EVAL METRICS #############
########################################

def axes_alignment(data=None, labels=None, model=None, folder_name=None):

    labels = labels.tolist()
    for label in set(labels):
        indecies = [i for i, x in enumerate(labels) if x == label]
        filtered_data = chainer.Variable(data.take(indecies, axis=0))
        latent = model.get_latent_mu(filtered_data)
        latent = latent.data
        hinton_diagram(data=np.array([latent[:, i] for i in range(latent.shape[-1])]), label=label, folder_name=folder_name)

def hinton_diagram(data=None, label=None, folder_name=None):
        fig,ax = plt.subplots(1,1)
        data = np.array(data)
        principal_axes = np.identity(data.shape[0])

        ax.patch.set_facecolor('lightgray')
        ax.set_aspect('equal', 'box')
        
        # Customize minor tick labels
        ax.set_xticks([0,1,2,3],      minor=False)
        ax.set_xticklabels(['pc1','pc2','pc3','pc4'], minor=False, fontsize=14)

        ax.set_yticks([0,1,2,3],      minor=False)
        ax.set_yticklabels(['z1','z2','z3','z4'], minor=False, fontsize=14)
        
        scatter = np.cov(data)
        eig_val, eig_vec = np.linalg.eig(scatter)
        eig_vec = eig_vec.T
        
        pairs = list(itertools.product(eig_vec, principal_axes))
        cosines = np.array([abs(cosine(x=p[0], y=p[1])) for p in pairs]).reshape(scatter.shape)

        min_eig_value_x = eig_val.argmin()
        min_eig_value_y = cosines[min_eig_value_x].argmax()
        max_eig_value = eig_val.max()
        height, width = cosines.shape
        
        fmt = '.2f'
        for (x, y), c in np.ndenumerate(cosines):
            val = eig_val[x]
            if x == min_eig_value_x or y == min_eig_value_y:
                color = (1.0, 1.0, 1.0)
                text_color = (0.0, 0.0, 0.0)
            else:
                color = (0.0, 0.0, 0.0)
                text_color = (1.0, 1.0, 1.0)
            size = np.sqrt(c)
            rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                 facecolor=color, edgecolor=color)
            ax.add_patch(rect)
            ax.text(y, x, format(c, fmt),
                     horizontalalignment="center",
                     color=text_color,
                     fontsize=14, fontweight='bold')

        ax.autoscale_view()
        ax.invert_yaxis()
        ax.set_title(label, fontweight="bold", fontsize=14)
        plt.savefig(os.path.join(folder_name, label + '_Hinton.png'))
        plt.close()

    
def cosine(x=None,y=None):
    return np.dot(x,y) / float(np.linalg.norm(x) * np.linalg.norm(y))


def test_time_classification(data_test=None, data_all=None, labels=None, unseen_labels=None, groups=None, 
                             boundaries=None, model=None, colors=None, folder_name=None):

    classifiers = {}
    stds = {}
    predicted_labels = []
    for key in sorted(groups.keys()):
        stds[key] = []
        classifiers[key] = []
        
    for key in sorted(groups.keys()):
        for label in groups[key]:
            indecies = [i for i, x in enumerate(labels) if x == label]
            filtered_data = chainer.Variable(data_test.take(indecies, axis=0))
            latent = model.get_latent(filtered_data)
            latent = latent.data

            mean = np.mean(latent, axis=0)
            cov = np.cov(latent.T)
            classifiers[key].append({"label": label, "mean":mean[int(key)], "cov":cov[int(key),int(key)]})
        
        # sort the list by the value of the mean element
        classifiers[key] = sorted(classifiers[key], key=lambda k: k['mean']) 

    for key in sorted(classifiers.keys()):

        # intermediate unknown distributions
        classifier_tuples = zip(classifiers[key], classifiers[key][1:])

        # guarding unknown distributions
        lefmost = classifiers[key][0]
        rightmost = classifiers[key][-1]
        # boundaries are [[min_x, min_y],[max_x, max_y]]
        classifiers[key] += [{"label": "unknown", "mean": boundaries[0][int(key)], "cov": lefmost["cov"]}]
        classifiers[key] += [{"label": "unknown", "mean": boundaries[1][int(key)], "cov": rightmost["cov"]}]

        classifiers[key] += [{"label": "unknown", "mean": 0.5 * (cl1["mean"] + cl2["mean"]), "cov": 0.5 * (cl1["cov"] + \
                             cl2["cov"])} for (cl1, cl2) in classifier_tuples]


    all_latent = model.get_latent_mu(data_all)

    # Show the 1D Gaussians per Group
    range = np.arange(-10, 10, 0.001)
    all_labels = np.append(labels, unseen_labels, axis=0)
    for label in list(set(all_labels)):
        for key in sorted(classifiers.keys()):
            if label in groups[key] or label[0] == "u":
                indecies = [i for i, x in enumerate(all_labels) if x == label]
                filtered_data = chainer.Variable(np.repeat(data_all, 2, axis=0).take(indecies, axis=0))

                latent = model.get_latent_mu(filtered_data)

                plt.figure(figsize=(10,10))

                for cl in classifiers[key]:
                    color = colors["singular"][cl["label"]]["dist"]
                    plt.plot(range, norm.pdf(range, cl["mean"], cl["cov"]), color=color, label=cl["label"])
                    plt.plot([cl["mean"], cl["mean"]], [0, norm.pdf([cl["mean"]], cl["mean"], cl["cov"])], color='r', linestyle="--")
                    plt.xlim(boundaries[0][int(key)] - 2, boundaries[1][int(key)] + 2)
                x = latent[:, int(key)]
                y = np.zeros((1, len(latent)))
                plt.scatter(x.data, y, alpha=0.75, marker='o', label=label + "_data")
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)
                plt.grid()
                plt.savefig(os.path.join(folder_name, label + "_group_" + key + "_testime_distrobutions"), bbox_inches="tight")
                plt.close()

    for key in sorted(classifiers.keys()):
        points = all_latent[:, int(key)]
        stds[key] = [[{"label": c["label"], "value": abs(c["mean"] - point.data) / c["cov"]} for c in classifiers[key]] for point in points]
        stds[key] = map(lambda point_stds : sorted(point_stds, key=lambda k: k["value"])[0]["label"], stds[key])

    predicted_labels = np.array([stds[key] for key in sorted(stds.keys())])

    return predicted_labels


def label_analysis(labels=None, predictions=None, groups=None, model=None, folder_name=None):
    
    true_labels = []
    n_groups = len(groups)
    groups = copy.deepcopy(groups)
    for i in range(n_groups):
        groups[str(i)].append("unknown")
        true_labels.append(labels[i::n_groups])

    # at this point both true_labels and predictions are strings
    true_sets = [sorted(list(set(group))) for group in true_labels]
    pred_sets = [sorted(groups[group_key]) for group_key in sorted(groups.keys())]

    cms = []
    for i in range(len(predictions)):
        pred_per_group = predictions[i]
        true_per_group = true_labels[i]
        pred_set = pred_sets[i]
        true_set = true_sets[i]

        cm = np.zeros((len(true_set), len(pred_set)))

        for i in range(len(true_per_group)):
            label_t = true_per_group[i]
            label_p = pred_per_group[i]
            x = true_set.index(label_t)
            y = pred_set.index(label_p)
            cm[x,y] += 1

        cms.append(cm)

    cms = np.array(cms)

    plot_confusion_matrix(cms=cms, group_classes=zip(true_sets, pred_sets),
                          title="Confusion Matrix Singular",
                          folder_name=folder_name)


def plot_confusion_matrix(cms=None, group_classes=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          folder_name=None):

    fig, subfiures = plt.subplots(nrows=1, ncols=2, figsize=(20,10))
    for i, subfig in enumerate(subfiures):

        cm = cms[i]
        (true, pred) = group_classes[i]

        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        cax = subfig.imshow(cm, interpolation='nearest', cmap=cmap)
        subfig.set_title(i, fontweight="bold", fontsize=14)
        fig.colorbar(cax, ax=subfig)
        
        subfig.set_xticks(range(len(pred)), minor=False)
        subfig.set_xticklabels(pred, minor=False, fontsize=14)
        subfig.set_yticks(range(len(true)), minor=False)
        subfig.set_yticklabels(true, minor=False, fontsize=14)

        fmt = '.2f'
        thresh = cm.max() / 2.
        for x, y in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

            size = 1.0
            rect = plt.Rectangle([y - size / 2, x - size / 2], size, size,
                                     facecolor=(0,0,0,0))
            subfig.add_patch(rect)
            subfig.text(y, x, format(cm_norm[x, y], fmt),
                     horizontalalignment="center",
                     color="white" if cm[x, y] > thresh else "black",
                     fontsize=14, fontweight='bold')

        subfig.autoscale_view()
        subfig.set_ylabel('True label', fontsize=12)
        subfig.set_xlabel('Predicted label', fontsize=12)
    fig.tight_layout()
    plt.savefig(os.path.join(folder_name, title + "_confusion_matrices" + '.png'))
    plt.close()