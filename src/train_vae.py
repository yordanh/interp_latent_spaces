#!/usr/bin/env python
"""
title           :train_vae.py
description     :Contains the main trainign loop and test time evaluation of the model.
author          :Yordan Hristov <yordan.hristov@ed.ac.uk
date            :05/2018
python_version  :2.7.14
==============================================================================
"""

import argparse
import os
import cv2
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import subprocess
import shutil

import chainer
from chainer import training
from chainer.training import extensions
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu
import chainer.functions as F
from chainer import serializers


import net
import data_generator
from config_parser import ConfigParser
from utils import *


def main():
    parser = argparse.ArgumentParser(description='Chainer example: VAE')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result/',
                        help='Directory to output the result')
    parser.add_argument('--epoch_labelled', '-e', default=100, type=int,
                        help='Number of epochs to learn only with labelled data')
    parser.add_argument('--epoch_unlabelled', '-u', default=100, type=int,
                    help='Number of epochs to learn with labelled and unlabelled data')
    parser.add_argument('--dimz', '-z', default=2, type=int,
                        help='Dimention of encoded vector')
    parser.add_argument('--batchsize', '-batch', type=int, default=128,
                        help='Learning minibatch size')
    parser.add_argument('--data', '-d', default='sprites',
                        help='Name of the dataset to be used for experiments')
    parser.add_argument('--model', '-m', default='conv',
                        help='Convolutional or linear model')
    parser.add_argument('--beta', '-b', default=100,
                        help='Beta coefficient for the KL loss')
    parser.add_argument('--gamma', '-g', default=100000,
                        help='Gamma coefficient for the classification loss')
    parser.add_argument('--labels', '-l', default="composite", 
                        help='Determined how to treat the labels for the different images')
    parser.add_argument('--freq', '-f', default=10, 
                    help='Frequency at which snapshots of the model are saved.')
    parser.add_argument('--mode', default="weakly", 
                    help='Mode of training - weakly supervised or unsupervised')
    args = parser.parse_args()

    print('\n###############################################')
    print('# GPU: \t\t\t{}'.format(args.gpu))
    print('# dim z: \t\t{}'.format(args.dimz))
    print('# Minibatch-size: \t{}'.format(args.batchsize))
    print('# Epochs Labelled: \t{}'.format(args.epoch_labelled))
    print('# Epochs Unabelled: \t{}'.format(args.epoch_unlabelled))
    print('# Dataset: \t\t{}'.format(args.data))
    print('# Model Architecture: \t{}'.format(args.model))
    print('# Beta: \t\t{}'.format(args.beta))
    print('# Gamma: \t\t{}'.format(args.gamma))
    print('# Frequency: \t\t{}'.format(args.freq))
    print('# Trainign model: \t{}'.format(args.model))
    print('# Out Folder: \t\t{}'.format(args.out))
    print('###############################################\n')

    stats = {'train_loss': [], 'train_accs': [], 'valid_loss': [], 'valid_rec_loss': [], 'valid_label_loss': [],\
         'valid_label_acc': [], 'valid_kl': []}

    models_folder = os.path.join(args.out, "models")
    manifold_gif = os.path.join(args.out, "gifs/manifold_gif")
    scatter_gif = os.path.join(args.out, "gifs/scatter_gif")
    scatter_folder = os.path.join(args.out, "scatter")
    eval_folder = os.path.join(args.out, "eval")
    shutil.rmtree(os.path.join(args.out, "models"))
    os.mkdir(os.path.join(args.out, "models"))

    if args.mode == "unsupervised":
        ignore = []
    else:
        ignore = ["unlabelled"]

    generator = data_generator.DataGenerator()
    train, train_labels, train_concat, train_vectors, test, test_labels, test_concat, test_vectors, unseen,\
    unseen_labels, unseen_concat, unseen_vectors, groups = generator.generate_dataset(ignore=ignore, args=args)
    
    data_dimensions = train.shape
    print('\n###############################################')
    print("DATA_LOADED")
    print("# Training: \t\t{0}".format(train.shape))
    print("# Training labels: \t{0}".format(set(train_labels)))
    print("# Training labels: \t{0}".format(train_labels.shape))
    print("# Training vectors: \t{0}".format(train_vectors.shape))
    print("# Testing: \t\t{0}".format(test.shape))
    print("# Testing labels: \t{0}".format(set(test_labels)))
    print("# Testing labels: \t{0}".format(test_labels.shape))
    print("# Testing vectors: \t{0}".format(test_vectors.shape))
    print("# Unseen: \t\t{0}".format(unseen.shape))
    print("# Unseen labels: \t{0}".format(set(unseen_labels)))
    print('###############################################\n')

    train_iter = chainer.iterators.SerialIterator(train_concat, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_concat, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Prepare VAE model, defined in net.py
    if args.model == "conv":
        if args.data == "sprites":
            model = net.Conv_VAE(train.shape[1], n_latent=args.dimz, groups=groups, beta=args.beta, gamma=args.gamma)
        else:
            model = net.Conv_VAE_MNIST(train.shape[1], args.dimz, beta=args.beta)
    else:
        model = net.VAE(train.shape[1], args.dimz, 500)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    lf = model.get_loss_func()
    
    stats, model, optimizer, epochs_so_far = training_loop(model=model, optimizer=optimizer, stats=stats, 
                                                           epochs=args.epoch_labelled, train_iter=train_iter, 
                                                           test_iter=test_iter, lf=lf, models_folder=models_folder, 
                                                           args=args)

    print("Save Model\n")
    serializers.save_npz(os.path.join(models_folder, 'final.model'), model)

    print("Save Optimizer\n")
    serializers.save_npz(os.path.join(models_folder, 'final.state'), optimizer)

    print("Clear Images from Last experiment\n")
    clear_last_results(folder_name=args.out)

    if args.mode == "weakly":
        model.to_cpu()  

        config_parser = ConfigParser("config/config.json")
        groups = config_parser.parse_groups()

        # calculate manifold boundaries
        latent = model.get_latent(test).data
        mean = np.mean(latent, axis=0)
        cov = np.cov(latent.T)
        no_std = 2
        # boundaries are [[min_x, min_y],[max_x, max_y]]
        boundaries = np.array([mean - no_std*cov.diagonal(), mean + no_std*cov.diagonal()])

        # assign colors to each label for plotting purposes
        all_labels = np.append(test_labels, unseen_labels, axis=0)
        colors = attach_colors(labels=all_labels)  

        # visualise the learnt data manifold in the latent space
        print("Plot Reconstructed images sampeld from a standart Normal\n")
        data = np.repeat(np.append(test, unseen, axis=0), 2, axis=0)
        figure_title = "Manifold Visualisation"
        plot_sampled_images(model=model, data=data, boundaries=boundaries, image_size=data_dimensions[-1], 
                            image_channels=data_dimensions[1], filename=os.path.join(args.out, "manifold"), 
                            figure_title=figure_title)

        print("Performing Reconstructions\n")
        perform_reconstructions(model=model, train=train, test=test, unseen=unseen, no_images=25, name_suffix="supervised", args=args)

        os.mkdir(os.path.join(scatter_folder, "supervised"))
        print("Plot Latent Testing Distribution for Singular Labels\n")
        data = np.repeat(test, 2, axis=0)
        plot_labels = test_labels
        plot_separate_distributions(data=data, labels=plot_labels, groups=groups, boundaries=boundaries, colors=colors["singular"], 
                                    model=model, filename=os.path.join(scatter_folder, "supervised", "singular_separate"))
        plot_overall_distribution(data=data, labels=plot_labels, boundaries=boundaries, colors=colors["singular"], 
                                  model=model, filename=os.path.join(scatter_folder, "supervised", "singular_together"))



        generator = data_generator.DataGenerator()
        train, train_labels, train_concat, train_vectors, test, test_labels, test_concat, test_vectors, unseen,\
        unseen_labels, unseen_concat, unseen_vectors, groups = generator.generate_dataset(args=args)
        
        data_dimensions = train.shape
        print('\n###############################################')
        print("DATA_LOADED")
        print("# Training: \t\t{0}".format(train.shape))
        print("# Training labels: \t{0}".format(set(train_labels)))
        print("# Training labels: \t{0}".format(train_labels.shape))
        print("# Training vectors: \t{0}".format(train_vectors.shape))
        print("# Testing: \t\t{0}".format(test.shape))
        print("# Testing labels: \t{0}".format(set(test_labels)))
        print("# Testing labels: \t{0}".format(test_labels.shape))
        print("# Testing vectors: \t{0}".format(test_vectors.shape))
        print("# Unseen: \t\t{0}".format(unseen.shape))
        print("# Unseen labels: \t{0}".format(set(unseen_labels)))
        print('###############################################\n')

        train_iter = chainer.iterators.SerialIterator(train_concat, args.batchsize)
        test_iter = chainer.iterators.SerialIterator(test_concat, args.batchsize,
                                                     repeat=False, shuffle=False)

        model = net.Conv_VAE(train.shape[1], n_latent=args.dimz, groups=groups, beta=args.beta, gamma=args.gamma)
        serializers.load_npz("result/models/final.model", model)
        model.gamma = 10000
        # model.beta = 1

        if args.gpu >= 0:
            # Make a specified GPU current
            chainer.cuda.get_device_from_id(args.gpu).use()
            model.to_gpu()

        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)

        lf = model.get_loss_func()
        stats, model, optimizer, _ = training_loop(model=model, optimizer=optimizer, stats=stats, epochs=args.epoch_unlabelled, 
                                                train_iter=train_iter, test_iter=test_iter, lf=lf, models_folder=models_folder,
                                                epochs_so_far=epochs_so_far, args=args)

########################################
########### RESULTS ANALYSIS ###########
########################################

    model.to_cpu()  

    config_parser = ConfigParser("config/config.json")
    groups = config_parser.parse_groups()

    # calculate manifold boundaries
    latent = model.get_latent(test).data
    mean = np.mean(latent, axis=0)
    cov = np.cov(latent.T)
    # boundaries are [[min_x, min_y],[max_x, max_y]]
    boundaries = np.array([mean - no_std*cov.diagonal(), mean + no_std*cov.diagonal()])

    # assign colors to each label for plotting purposes
    all_labels = np.append(test_labels, unseen_labels, axis=0)
    colors = attach_colors(labels=all_labels)

    # visualise the learnt data manifold in the latent space
    print("Plot Reconstructed images sampeld from a standart Normal\n")
    data = np.repeat(np.append(test, unseen, axis=0), 2, axis=0)
    figure_title = "Manifold Visualisation"
    plot_sampled_images(model=model, data=data, boundaries=boundaries, image_size=data_dimensions[-1], 
                        image_channels=data_dimensions[1], filename=os.path.join(args.out, "manifold_1"), 
                        figure_title=figure_title)

    print("Test time Classification\n")
    tmp_labels = test_time_classification(data_test=np.repeat(test, 2, axis=0), data_all=np.append(test, unseen, axis=0), 
                                          labels=test_labels, unseen_labels=unseen_labels, groups=groups, 
                                          boundaries=boundaries, model=model, colors=colors, folder_name=eval_folder)

    print("Label Analisys\n")
    true_labels = np.append(test_labels, unseen_labels, axis=0)
    label_analysis(labels=true_labels, predictions=tmp_labels, groups=groups, model=model, folder_name=eval_folder)

    print("Saving the loss plots\n")
    plot_loss_curves(stats=stats, args=args)

    print("Evaluate Axes Alignment\n")
    data = np.repeat(np.append(test, unseen, axis=0), 2, axis=0)
    plot_labels = np.append(test_labels, unseen_labels, axis=0)
    axes_alignment(data=data, labels=plot_labels, model=model, folder_name=eval_folder)

    print("Performing Reconstructions\n")
    perform_reconstructions(model=model, train=train, test=test, unseen=unseen, no_images=25, name_suffix="weakly_supervised", args=args)

    print("Plot Latent Testing Distribution for Singular Labels\n")
    data = np.repeat(test, 2, axis=0)
    plot_labels = test_labels
    plot_separate_distributions(data=data, labels=plot_labels, groups=groups, boundaries=boundaries, colors=colors["singular"], 
                                model=model, filename=os.path.join(scatter_folder, "singular_separate"))
    plot_overall_distribution(data=data, labels=plot_labels, boundaries=boundaries, colors=colors["singular"], 
                              model=model, filename=os.path.join(scatter_folder, "singular_together"))

    print("Plot Latent Testing Distribution for Singular Labels + Unseen Distribution\n")
    data = np.repeat(np.append(test, unseen, axis=0), 2, axis=0)
    plot_labels = np.append(test_labels, unseen_labels, axis=0)
    plot_separate_distributions(data=data, labels=plot_labels, boundaries=boundaries, colors=colors["singular"], 
                                model=model, filename=os.path.join(scatter_folder, "singular_separate_unseen"))
    plot_overall_distribution(data=data, labels=plot_labels, boundaries=boundaries, colors=colors["singular"], 
                              model=model, filename=os.path.join(scatter_folder, "singular_together_unseen"))

    if args.labels == "composite":
        print("Plot Latent Testing Distribution for Composite Labels\n")
        # compose the composite labels
        data = test
        test_labels_tmp = test_labels.reshape(len(test_labels) / 2, 2)
        plot_labels = np.array(["_".join(x) for x in test_labels_tmp])
        plot_separate_distributions(data=data, labels=plot_labels, boundaries=boundaries, colors=colors["composite"], 
                                    model=model, filename=os.path.join(scatter_folder, "composite_separate"))
        plot_overall_distribution(data=data, labels=plot_labels, boundaries=boundaries, colors=colors["composite"], 
                                  model=model, filename=os.path.join(scatter_folder, "composite_together"))

        print("Plot Latent Testing Distribution for Composite Labels + Unseen Distribution\n")
        data = np.append(test, unseen, axis=0)
        test_labels_tmp = np.append(test_labels, unseen_labels, axis=0)
        test_labels_tmp = test_labels_tmp.reshape(len(test_labels_tmp) / 2, 2)
        plot_labels = np.array(["_".join(x) for x in test_labels_tmp])
        plot_separate_distributions(data=data, labels=plot_labels, boundaries=boundaries, colors=colors["composite"], 
                                    model=model, filename=os.path.join(scatter_folder, "composite_separate_unseen"))
        plot_overall_distribution(data=data, labels=plot_labels, boundaries=boundaries, colors=colors["composite"], 
                                  model=model, filename=os.path.join(scatter_folder, "composite_together_unseen"))

    print("Generating data for retrospective model evaluation\n")
    for model_name in list(filter(lambda name : "final" not in name, os.listdir(models_folder))):
        serializers.load_npz(os.path.join(models_folder, model_name), model)
        filename = model_name.replace(".model", "")

        figure_title = "Manifold Visualisation for epoch {0}".format(filename)
        data = np.repeat(np.append(test, unseen, axis=0), 2, axis=0)
        plot_sampled_images(model=model, data=data, boundaries=boundaries, image_size=data_dimensions[-1], 
                            image_channels=data_dimensions[1], filename=os.path.join(manifold_gif, filename),
                            figure_title = figure_title)


        data = np.repeat(np.append(test, unseen, axis=0), 2, axis=0)
        plot_labels = np.append(test_labels, unseen_labels, axis=0)
        for key in groups:
            if not os.path.exists(os.path.join(scatter_gif, key)):
                os.mkdir(os.path.join(scatter_gif, key))
            plot_group_distribution(data=data, labels=plot_labels, boundaries=boundaries, colors=colors["singular"], 
                                    model=model, group_id=key, filename=os.path.join(scatter_gif, key, filename))

    print("Making the Latent Manifold GIF\n")
    samples = [x.split('_')[0] for x in os.listdir(manifold_gif)]
    rests = ['_'.join(x.split('_')[1:]) for x in os.listdir(manifold_gif)]
    samples.sort(key=int)
    samples = [os.path.join(manifold_gif, x + "_" + rest) for (x, rest) in zip(samples, rests)]
    result_name = os.path.join(manifold_gif, "samples_animation.gif")
    subprocess.call(["convert", "-loop", "5", "-delay",  "50"] + samples + [result_name])

    for key in groups:
        print("Making the Composite Label Distribution GIF for group" + key + "\n")
        folder_name = os.path.join(scatter_gif, key)
        distr = [x.replace(".png", "") for x in os.listdir(folder_name)]
        distr.sort(key=int)
        distr = [os.path.join(folder_name, x + ".png") for x in distr]
        result_name = os.path.join(folder_name, "distr_animation.gif")
        subprocess.call(["convert", "-loop", "5", "-delay",  "50"] + distr + [result_name])

def training_loop(model=None, optimizer=None, stats=None, epochs=None, train_iter=None, test_iter=None, lf=None, 
                  models_folder=None, epochs_so_far=0, args=None):

    train_losses = []
    train_accs = []

    while train_iter.epoch < epochs:

        # ------------ One epoch of the training loop ------------
        # ---------- One iteration of the training loop ----------
        train_batch = train_iter.next()

        image_train = concat_examples(train_batch, 0)

        # Calculate the loss with softmax_cross_entropy
        train_loss, train_rec_loss, train_label_loss, acc, _ = model.get_loss_func()(image_train)
        train_losses.append(train_loss.array)
        train_accs.append(acc.array)
        # Calculate the gradients in the network
        model.cleargrads()
        train_loss.backward()

        # Update all the trainable paremters
        optimizer.update()


        if train_iter.epoch % int(args.freq) == 0:
            serializers.save_npz(os.path.join(models_folder ,str(train_iter.epoch + epochs_so_far) + '.model'), model)

        # --------------------- iteration until here --------------------- 

        if train_iter.is_new_epoch:

            test_losses = []
            test_accs = []
            test_rec_losses = []
            test_label_losses = []
            test_kl = []
            while True:

                test_batch = test_iter.next()

                image_test = concat_examples(test_batch, 0)

                loss, rec_loss, label_loss, label_acc, kl = model.get_loss_func()(image_test)
                test_losses.append(loss.array)
                test_rec_losses.append(rec_loss.array)
                test_label_losses.append(label_loss.array)
                test_accs.append(label_acc.array)
                test_kl.append(kl.array)

                if test_iter.is_new_epoch:
                    test_iter.epoch = 0
                    test_iter.current_position = 0
                    test_iter.is_new_epoch = False
                    test_iter._pushed_position = None
                    break

            stats['train_loss'].append(np.mean(to_cpu(train_losses)))
            stats['train_accs'].append(np.mean(to_cpu(train_accs)))
            stats['valid_loss'].append(np.mean(to_cpu(test_losses)))
            stats['valid_rec_loss'].append(np.mean(to_cpu(test_rec_losses)))
            stats['valid_label_loss'].append(np.mean(to_cpu(test_label_losses)))
            stats['valid_label_acc'].append(np.mean(to_cpu(test_accs)))
            stats['valid_kl'].append(np.mean(to_cpu(test_kl)))
    
            print(("Epoch: {0} \t T_Loss: {1} \t V_Loss: {2} \t V_Rec_Loss: {3} \t V_Label_Loss: {4} \t " + \
                  "V_KL: {6} \t T_Acc: {7} \t V_Acc: {5}").format(train_iter.epoch, 
                                                                round(stats['train_loss'][-1], 2),
                                                                round(stats['valid_loss'][-1], 2),
                                                                round(stats['valid_rec_loss'][-1], 2),
                                                                round(stats['valid_label_loss'][-1], 2),
                                                                round(stats['valid_label_acc'][-1], 2),
                                                                round(stats['valid_kl'][-1], 2),
                                                                round(stats['train_accs'][-1], 2)))
            train_losses = []
            train_accs = []
        # --------------------- epoch until here --------------------- 

    return stats, model, optimizer, epochs


if __name__ == '__main__':
    main()
