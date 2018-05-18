#!/usr/bin/env pytho
import os

import chainer
import chainer.functions as F
import numpy as np
from scipy.stats import multivariate_normal
import cv2
from scipy.stats import norm

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

########################################
############ UTIL FUNCTIONS ############
########################################

# delete all result files from the output folder
def clear_last_results(folder_name):
    all_images = list(filter(lambda filename : '.png' in filename, os.listdir(folder_name)))
    map(lambda x : os.remove(folder_name + x), all_images)


# for a given set of example images, calculate their reconstructions
def perform_reconstructions(model, train, test, unseen, args):
    train_ind = np.linspace(0, len(train) - 1, 9, dtype=int)
    x = chainer.Variable(np.asarray(train[train_ind]))
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x1 = model(x)
    save_images(x.data, os.path.join(args.out, 'train'), args=args)
    save_images(x1.data, os.path.join(args.out, 'train_reconstructed'), args=args)

    # reconstruct testing examples
    test_ind = np.linspace(0, len(test) - 1, 9, dtype=int)
    x = chainer.Variable(np.asarray(test[test_ind]))
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x1 = model(x)
    save_images(x.data, os.path.join(args.out, 'test'), args=args)
    save_images(x1.data, os.path.join(args.out, 'test_reconstructed'), args=args)

    # reconstruct unseen examples
    if len(unseen) != 0:
        unseen_ind = np.linspace(0, len(unseen) - 1, 9, dtype=int)
        x = chainer.Variable(np.asarray(unseen[unseen_ind]))
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            x1 = model(x)
        save_images(x.data, os.path.join(args.out, 'unseen'), args=args)
        save_images(x1.data, os.path.join(args.out, 'unseen_reconstructed'), args=args)

    # draw images from randomly sampled z under a 'vanilla' normal distribution
    z = chainer.Variable(
        np.random.normal(0, 1, (9, args.dimz)).astype(np.float32))
    x = model.decode(z)
    save_images(x.data, os.path.join(args.out, 'sampled'), args=args)


# plot and save loss and accuracy curves
def plot_loss_curves(stats, args):
    # overall train/validation losses
    plt.figure(figsize=(6, 6))
    plt.grid()
    colors = ['r', 'k', 'b', 'g', 'gold']
    for i, channel in enumerate(stats):
        if channel == "valid_label_loss" or channel == "valid_label_acc" or channel == "train_accs":
            continue
        plt.plot(range(args.epoch),stats[channel], marker='x', color=colors[i], label=channel)
    
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(args.out + "losses"))

    # validation label loss
    plt.figure(figsize=(6, 6))
    plt.grid()
    plt.plot(range(args.epoch),stats['valid_label_loss'], marker='x', color='g', label='valid_label_loss')
    
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(args.out + "label_loss"))

    # validation label accuracy
    plt.figure(figsize=(6, 6))
    plt.grid()
    plt.plot(range(args.epoch),stats['valid_label_acc'], marker='x', color='r', label='valid_label_acc')
    
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(args.out + "label_acc"))

    # training label accuracy
    plt.figure(figsize=(6, 6))
    plt.grid()
    plt.plot(range(args.epoch),stats['train_accs'], marker='x', color='b', label='train_accs')
    
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(args.out + "train_label_acc"))

    plt.close()


# calculate statistics for the predicted labels
def compare_labels(test, test_labels, model, args, cuttoff_thresh=1):

    mu, ln_var = model.encode(test)
    
    if args.labels == "composite":
        hat_labels_0, hat_labels_1 = model.predict_label(mu, ln_var, softmax=True)

        print("\nValidation Accuracy for group 0: {0}\n".format(F.accuracy(hat_labels_0, test_labels[0])))
        print("\nValidation Accuracy for group 1: {0}\n".format(F.accuracy(hat_labels_1, test_labels[1])))
    elif args.labels == "singular":
        hat_labels = model.predict_label(mu, ln_var, softmax=True)

        print("Validation Accuracy: {0}\n".format(F.accuracy(hat_labels, test_labels)))


# visualize the results
def save_images(x, filename, args):

    fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
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

        image = ai.imshow(cv2.cvtColor(xi, cv2.COLOR_BGR2RGB))
    fig.savefig(filename)


# attach a color to each singular and composite class labels, both for their data points 
# and fitted overlayed distributions
def attach_colors(labels, composite=True):

    colors = ['c', 'b', 'g', 'y', 'k', 'orange', 'maroon', 'lime', 'salmon', 'crimson', 'gold', 'coral']
    result = {"singular":{}, "composite":{}}

    counter = 0
    for label in set(labels):
        if label in result["singular"]:
            continue
        else:
            result["singular"][label] = {}
            result["singular"][label]["data"] = colors[counter]
            result["singular"][label]["dist"] = colors[counter + 1]
            counter += 1

    if composite:
        labels = labels.reshape(len(labels) / 2, 2)
        labels = np.array(["_".join(x) for x in labels])

        counter = 0
        for label in set(labels):
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
def plot_overall_distribution(data, labels, boundaries, colors, model, name, args):
    latent_all = None

    # scatter plot all the data points in the latent space
    plt.figure(figsize=(6, 6))
    concise_colors = list(set(labels))
    labels = labels.tolist()
    for label in set(labels):
        indecies = [i for i, x in enumerate(labels) if x == label]
        filtered_data = chainer.Variable(data.take(indecies, axis=0))
        latent = model.get_latent(filtered_data)
        latent = latent.data
        plt.scatter(latent[:, 0], latent[:, 1], c=colors[label]["data"], label=str(label))

        if latent_all is not None:
            latent_all = np.append(latent_all, latent, axis=0)
        else:
            latent_all = latent
    plt.legend(loc='upper right')

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
    plt.savefig(os.path.join(args.out, name))
    
    # fit and plot a distribution over all the latent projections
    delta = 0.025
    mean = np.mean(latent_all, axis=0)
    cov = np.cov(latent_all.T)
    x = np.arange(min(latent_all[:, 0]), max(latent_all[:, 0]), delta)
    y = np.arange(min(latent_all[:, 1]), max(latent_all[:, 1]), delta)
    X, Y = np.meshgrid(x, y)
    Z = multivariate_normal.pdf(np.array([zip(c,d) for c,d in zip(X,Y)]), mean=mean, cov=cov)
    plt.contour(X, Y, Z, colors='r')
    plt.title("mu[0]:{0}; mu[1]:{1}\ncov[0,0]:{2}; cov[1,1]:{3}\ncov[0,1]:{4}".format(round(mean[0],2), round(mean[1],2), round(cov[0,0],2), round(cov[1,1],2), round(cov[0,1],2)))
    plt.savefig(os.path.join(args.out, name + "_overlayed"))
    plt.close()


# plot a set of input datapoitns to the latent space and fit normal distributions over the projections
# show the contours for the distribution for each label
def plot_separate_distributions(data, labels, boundaries, colors, model, name, args):
    latent_all = []

    # scatter plot all the data points in the latent space
    plt.figure(figsize=(6, 6))
    labels = labels.tolist()
    for label in set(labels):
        indecies = [i for i, x in enumerate(labels) if x == label]
        filtered_data = chainer.Variable(data.take(indecies, axis=0))
        latent = model.get_latent(filtered_data)
        latent = latent.data
        latent_all.append(latent)
        plt.scatter(latent[:, 0], latent[:, 1], c=colors[label]["data"], label=str(label))
    plt.legend(loc='upper right')

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
    plt.savefig(os.path.join(args.out, name))

    # fit and overlay distributions for each class/label
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
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(args.out, name + "_overlayed"))
    plt.close()

    # scatter datapoints and fit and overlay a distribution over each data label
    counter = 0
    for label in set(labels):
        plt.figure(figsize=(6, 6))
        latent = latent_all[counter]
        plt.scatter(latent[:, 0], latent[:, 1], c=colors[label]["data"], label=str(label))

        delta = 0.025
        mean = np.mean(latent, axis=0)
        cov = np.cov(latent.T)
        x = np.arange(min(latent[:, 0]), max(latent[:, 0]), delta)
        y = np.arange(min(latent[:, 1]), max(latent[:, 1]), delta)
        X, Y = np.meshgrid(x, y)
        Z = multivariate_normal.pdf(np.array([zip(c,d) for c,d in zip(X,Y)]), mean=mean, cov=cov)
        plt.contour(X, Y, Z, colors=colors[label]["dist"])
        plt.legend(loc='upper right')

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
        plt.savefig(os.path.join(args.out, name + "_overlayed" + "_" + str(counter)))
        plt.close()
        counter += 1

# sample datapoints under the prior normal distribution and reconstruct
# samples_per_dimension has to be even
def plot_sampled_images(model, data, boundaries=None, samples_per_dimension=16, image_size=100, offset=10, image_channels=3, args=None):

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
        grid_x = np.linspace(boundaries[0,0], boundaries[1,0], samples_per_dimension)
        grid_y = np.linspace(boundaries[0,1], boundaries[1,1], samples_per_dimension)

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]]).astype(np.float32)
                x_decoded = model.decode(chainer.Variable(z_sample)).data
                image_sample = x_decoded.reshape(x_decoded.shape[1:])

                # if the network is conv it already outputs reshaped images; only has to swap the channels to be at the back
                if args.model == "conv":
                    image_sample = np.swapaxes(image_sample, 0, 2)
                else:
                    if args.data == "sprites":
                        image_sample = image_sample.reshape(100, 100, 3)
                    else:
                        image_sample = image_sample.reshape(28, 28, 1)

                figure[i * image_size + i * offset: (i + 1) * image_size + i * offset,
                       j * image_size + j * offset: (j + 1) * image_size + j * offset,
                       :] = image_sample

        figure = (figure*255)
        cv2.imwrite("result/latent_samples.png", figure)