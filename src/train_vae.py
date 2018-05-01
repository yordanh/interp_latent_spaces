#!/usr/bin/env python
"""Chainer example: train a VAE on MNIST
"""
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions
import numpy as np
from scipy.stats import multivariate_normal
import cv2
from scipy.stats import norm
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu
import chainer.functions as F




import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


import net
import data_generator


def main():
    parser = argparse.ArgumentParser(description='Chainer example: VAE')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result/',
                        help='Directory to output the result')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--dimz', '-z', default=20, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--batchsize', '-batch', type=int, default=128,
                        help='learning minibatch size')
    parser.add_argument('--data', '-d', default='sprites',
                        help='Name of the dataset to be used for experiments')
    parser.add_argument('--model', '-m', default='conv',
                        help='Convolutional or linear model')
    # parser.add_argument('--mode', '-mo', default='verbose',
    #                     help='Training mode')
    parser.add_argument('--beta', '-b', default=1,
                        help='Beta coefficient for the loss')
    args = parser.parse_args()

    print('\n###############################################')
    print('# GPU: \t\t\t{}'.format(args.gpu))
    print('# dim z: \t\t{}'.format(args.dimz))
    print('# Minibatch-size: \t{}'.format(args.batchsize))
    print('# epoch: \t\t{}'.format(args.epoch))
    print('# Dataset: \t\t{}'.format(args.data))
    print('# Model Architecture: \t{}'.format(args.model))
    print('# Beta: \t\t{}'.format(args.beta))
    # print('# Training Mode: \t{}'.format(args.mode))
    print('# Out Folder: \t\t{}'.format(args.out))
    print('###############################################\n')

    generator = data_generator.DataGenerator()
    train, train_labels, train_concat, train_vectors, test, test_labels, test_concat, test_vectors, unseen, unseen_labels, unseen_concat, unseen_vectors = generator.generate_dataset(args)
    data_dimensions = train.shape
    print('\n###############################################')
    print("DATA_LOADED")
    print("# Training: \t\t{0}".format(train.shape))
    print("# Training labels: \t{0}".format(set(train_labels)))
    print("# Testing: \t\t{0}".format(test.shape))
    print("# Testing labels: \t{0}".format(set(test_labels)))
    print("# Unseen: \t\t{0}".format(unseen.shape))
    print("# Unseen labels: \t{0}".format(set(unseen_labels)))
    print('###############################################\n')

    stats = {'train_loss': [], 'train_accs': [], 'valid_loss': [], 'valid_rec_loss': [], 'valid_label_loss': [], 'valid_label_acc': [], 'valid_kl': []}

    train_iter = chainer.iterators.SerialIterator(train_concat, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_concat, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Prepare VAE model, defined in net.py
    if args.model == "conv":
        if args.data == "sprites":
            model = net.Conv_VAE(train.shape[1], n_latent=args.dimz, n_labels=len(set(train_labels)), beta=args.beta)
        else:
            model = net.Conv_VAE_MNIST(train.shape[1], args.dimz, beta=args.beta)
    else:
        model = net.VAE(train.shape[1], args.dimz, 500)


    compare_labels(test, np.array(test_vectors), model)    


    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Setup an optimizer
    # optimizer = chainer.optimizers.RMSprop(lr=0.001, alpha=0.9, eps=1e-7)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # if args.mode == "verbose":
    #     # Set up an updater. StandardUpdater can explicitly specify a loss function
    #     # used in the training with 'loss_func' option
    #     updater = training.StandardUpdater(
    #         train_iter, optimizer,
    #         device=args.gpu, loss_func=model.get_loss_func())

    #     trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    #     trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu,
    #                                         eval_func=model.get_loss_func(k=10)))

    #     trainer.extend(extensions.LogReport())
    #     trainer.extend(extensions.PrintReport(
    #         ['epoch', 'main/loss', 'validation/main/loss',
    #          'main/rec_loss', 'validation/main/rec_loss', 'elapsed_time']))
    #     trainer.extend(extensions.ProgressBar())

    #     # Save two plot images to the result dir
    #     if extensions.PlotReport.available():
    #         trainer.extend(
    #             extensions.PlotReport(['main/loss', 'validation/main/loss'],
    #                                   'epoch', file_name='loss.png'))

    #         trainer.extend(
    #             extensions.PlotReport(['main/rec_loss', 'validation/main/rec_loss'],
    #                                   'epoch', file_name='rec_loss.png'))
    #         trainer.extend(
    #             extensions.PlotReport(['main/label_loss', 'validation/main/label_loss'],
    #                                   'epoch', file_name='label_loss.png'))

    #     # Run the training loop
    #     trainer.run()

    # elif args.mode == "fast":
    train_losses = []
    train_accs = []
    lf = model.get_loss_func()
    while train_iter.epoch < args.epoch:

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
    
            print("Epoch: {0} \t T_Loss: {1} \t V_Loss: {2} \t V_Rec_Loss: {3} \t V_Label_Loss: {4} \t V_KL: {6} \t T_Acc: {7} \t V_Acc: {5}".format(train_iter.epoch, 
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


########################################
########### RESULTS ANALYSIS ###########
########################################

    model.to_cpu()  

    # remove all images from previous experiments
    print("Clean Images from Last experiment\n")
    clean_last_results(args.out)

    print("Saving the loss plots\n")
    plot_loss_curves(stats, args)

    print("Predict labels\n")
    compare_labels(test, np.array(test_vectors), model)    

    print("Performing Reconstructions\n")
    # reconstruct training examples
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

    # visualise distributions in the latent space
    print("Plot Latent Testing Distribution\n")
    test_labels = np.append(test_labels, unseen_labels, axis=0)
    plot_separate_distributions(test, test_labels, skip_labels=unseen_labels, model=model, name="a_separate", args=args)
    test = np.append(test, unseen, axis=0)
    # all test datapoints
    plot_overall_distribution(test, test_labels, skip_labels=unseen_labels, model=model, name="b_together", args=args)

    print("Plot Latent Testing + Unseen Distribution\n")
    # all test datapoints + unseen datapoints
    plot_separate_distributions(test, test_labels, skip_labels=[], model=model, name="a_separate_unseen", args=args)

    print("Plot Reconstructed images sampeld from a standart Normal\n")
    # visualise the learnt data manifold in the latent space
    plot_sampled_images(model, image_size=data_dimensions[-1], image_channels=data_dimensions[1], args=args)

########################################
############ UTIL FUNCTIONS ############
########################################

def clean_last_results(folder_name):
    all_images = list(filter(lambda filename : '.png' in filename, os.listdir(folder_name)))
    map(lambda x : os.remove(folder_name + x), all_images)


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


def compare_labels(test, test_labels, model, cuttoff_thresh=1):
    mu, _ = model.encode(test)
    hat_labels = model.predict_label(mu, softmax=True)

    result = []
    counters = {}

    for label in test_labels:
        counters[label] = 0

    for i, label in enumerate(test_labels):
        hat_label = hat_labels.data[i]
        if counters[label] > cuttoff_thresh:
            continue
        else:
            result.append((hat_label, label))
            counters[label] += 1

    print(np.array(result))

    print("\nValidation Accuracy: {0}\n".format(F.accuracy(hat_labels, test_labels)))


# Visualize the results
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
    # plt.colorbar(image)
    fig.savefig(filename)


# plot a set of input datapoitns to the latent space and fit a normal distribution over the projections
def plot_overall_distribution(test, test_labels, skip_labels, model, name, args):
    latent_all = None

    colors = ['c', 'b', 'g', 'y', 'k', 'orange', 'maroon', 'lime', 'salmon', 'crimson', 'gold', 'coral']

    # scatter plot all the data points in the latent space
    plt.figure(figsize=(6, 6))
    concise_colors = list(set(test_labels))
    test_labels = test_labels.tolist()
    for label in set(test_labels):
        # keep the color order between plot_overall_distribution and plot_separate_distributions 
        if label in skip_labels:
            continue
        indecies = [i for i, x in enumerate(test_labels) if x == label]
        filtered_test = chainer.Variable(test.take(indecies, axis=0))
        latent = model.get_latent(filtered_test)
        latent = latent.data
        plt.scatter(latent[:, 0], latent[:, 1], c=colors[concise_colors.index(label)], label=str(label))

        if latent_all is not None:
            latent_all = np.append(latent_all, latent, axis=0)
        else:
            latent_all = latent

    plt.legend(loc='upper right')

    plt.plot([-5, 5], [0, 0], 'r')
    plt.plot([0, 0], [-5, 5], 'r')
    plt.savefig(os.path.join(args.out, name))
    
    # fit a distribution over all the latent projections
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
def plot_separate_distributions(test, test_labels, skip_labels, model, name, args):
    latent_all = []

    colors = ['c', 'b', 'g', 'y', 'k', 'orange']
    colors_dist = ['maroon', 'lime', 'salmon', 'crimson', 'gold', 'coral']
    # fit a distribution over the latent projections for each label
    plt.figure(figsize=(6, 6))
    concise_colors = list(set(test_labels))
    test_labels = test_labels.tolist()
    for label in set(test_labels):
        if label in skip_labels:
            continue
        indecies = [i for i, x in enumerate(test_labels) if x == label]
        filtered_test = chainer.Variable(test.take(indecies, axis=0))
        latent = model.get_latent(filtered_test)
        latent = latent.data
        latent_all.append(latent)
        plt.scatter(latent[:, 0], latent[:, 1], c=colors[concise_colors.index(label)], label=str(label))
    plt.legend(loc='upper right')

    plt.plot([-5, 5], [0, 0], 'r')
    plt.plot([0, 0], [-5, 5], 'r')
    plt.savefig(os.path.join(args.out, name))

    counter = 0
    for label in set(test_labels):
        if label in skip_labels:
            continue
        # indecies = [i for i, x in enumerate(test_labels) if x == label]
        # filtered_test = chainer.Variable(test.take(indecies, axis=0))
        # latent = model.get_latent(filtered_test)
        # latent = latent.data
        latent = latent_all[counter]
        counter += 1

        delta = 0.025
        mean = np.mean(latent, axis=0)
        cov = np.cov(latent.T)
        x = np.arange(min(latent[:, 0]), max(latent[:, 0]), delta)
        y = np.arange(min(latent[:, 1]), max(latent[:, 1]), delta)
        X, Y = np.meshgrid(x, y)
        Z = multivariate_normal.pdf(np.array([zip(c,d) for c,d in zip(X,Y)]), mean=mean, cov=cov)
        plt.contour(X, Y, Z, colors=colors_dist[concise_colors.index(label)])
    plt.legend(loc='upper right')

    plt.plot([-5, 5], [0, 0], 'r')
    plt.plot([0, 0], [-5, 5], 'r')
    plt.savefig(os.path.join(args.out, name + "_overlayed"))

    plt.close()





    # for better inspection
    # concise_colors = list(set(test_labels))
    # test_labels = test_labels.tolist()
    counter = 0
    for label in set(test_labels):
        plt.figure(figsize=(6, 6))
        if label in skip_labels:
            continue
        # indecies = [i for i, x in enumerate(test_labels) if x == label]
        # filtered_test = chainer.Variable(test.take(indecies, axis=0))
        # latent = model.get_latent(filtered_test)
        # latent = latent.data
        latent = latent_all[counter]
        plt.scatter(latent[:, 0], latent[:, 1], c=colors[concise_colors.index(label)], label=str(label))

        delta = 0.025
        mean = np.mean(latent, axis=0)
        cov = np.cov(latent.T)
        x = np.arange(min(latent[:, 0]), max(latent[:, 0]), delta)
        y = np.arange(min(latent[:, 1]), max(latent[:, 1]), delta)
        X, Y = np.meshgrid(x, y)
        Z = multivariate_normal.pdf(np.array([zip(c,d) for c,d in zip(X,Y)]), mean=mean, cov=cov)
        plt.contour(X, Y, Z, colors=colors_dist[concise_colors.index(label)])

        plt.legend(loc='upper right')

        plt.plot([-5, 5], [0, 0], 'r')
        plt.plot([0, 0], [-5, 5], 'r')
        plt.savefig(os.path.join(args.out, name + "_overlayed" + "_" + str(counter)))
        plt.close()
        counter += 1

# sample datapoints under the prior normal distribution and reconstruct
def plot_sampled_images(model, samples_per_dimension=10, image_size=100, offset=10, image_channels=3, args=None):

        figure = np.ones((image_size * samples_per_dimension + offset * samples_per_dimension, image_size * samples_per_dimension + offset * samples_per_dimension, image_channels))
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
        grid_x = norm.ppf(np.linspace(0.95, 0.05, samples_per_dimension))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, samples_per_dimension))

        # print("SAMPLES:\n")

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]]).astype(np.float32)
                x_decoded = model.decode(chainer.Variable(z_sample)).data
                image_sample = x_decoded.reshape(x_decoded.shape[1:])

                # print("Min: {0} \t Max: {1} \t Mean: {2}".format(np.min(image_sample), np.max(image_sample), np.mean(image_sample)))


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

if __name__ == '__main__':
    main()
