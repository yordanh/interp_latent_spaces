import six

import chainer
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L
from chainer.backends.cuda import get_device_from_array
import numpy
import cupy


class VAE(chainer.Chain):
    """Variational AutoEncoder"""

    def __init__(self, n_in, n_latent, n_h):
        super(VAE, self).__init__()
        with self.init_scope():
            # encoder
            self.le1 = L.Linear(n_in, n_h)
            self.le2_mu = L.Linear(n_h, n_latent)
            self.le2_ln_var = L.Linear(n_h, n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_h)
            self.ld2 = L.Linear(n_h, n_in)

    def __call__(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.tanh(self.le1(x))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.tanh(self.ld1(z))
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2


    def get_latent(self, x):
        mu, ln_var = self.encode(x)
        return F.gaussian(mu, ln_var)
        # return mu


    def get_loss_func(self, C=1.0, k=1):
        """Get loss function of VAE.

        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.

        Args:
            C (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """
        def lf(x):

            images, labels = zip(*x)
            images = list(images)

            mu, ln_var = self.encode(images)
            batchsize = len(mu.data)
            # reconstruction loss
            rec_loss = 0
            for l in six.moves.range(k):
                z = F.gaussian(mu, ln_var)
                rec_loss += F.bernoulli_nll(images, self.decode(z, sigmoid=False)) \
                    / (k * batchsize)
            self.rec_loss = rec_loss
            self.loss = self.rec_loss + \
                C * gaussian_kl_divergence(mu, ln_var) / batchsize
            chainer.report(
                {'rec_loss': rec_loss, 'loss': self.loss}, observer=self)
            return self.loss
        return lf

class Conv_VAE(chainer.Chain):
    """Convolutional Variational AutoEncoder"""

    def __init__(self, in_channels, n_latent, groups, beta=100, gamma=100000):
        super(Conv_VAE, self).__init__()
        with self.init_scope():

            self.in_channels = in_channels
            self.beta = beta
            self.gamma = gamma
            self.groups = groups

            # encoder
            self.encoder_conv_0 = L.Convolution2D(in_channels, 32, ksize=3, pad=1) # (100, 100)
            # max pool ksize=2 (50, 50)
            self.encoder_conv_1 = L.Convolution2D(32, 16, ksize=3, pad=1) # (50, 50)
            # max pool ksize=2 (25,25)
            self.encoder_conv_2 = L.Convolution2D(16, 8, ksize=4) # (22, 22)
            # max pool ksize=2 (11,11)
            self.encoder_conv_3 = L.Convolution2D(8, 8, ksize=4) # (8, 8)
            # reshape from (8, 8, 8) to (1,512)
            self.encoder_dense_0 = L.Linear(512, 8)

            self.encoder_mu = L.Linear(8, n_latent)
            self.encoder_ln_var = L.Linear(8, n_latent)

            # label predictors taking only the mean value into account
            self.label_predictor_0 = L.Linear(1, len(self.groups["0"]))
            self.label_predictor_1 = L.Linear(1, len(self.groups["1"]))

            # decoder
            self.decoder_dense_0 = L.Linear(n_latent, 8)
            self.decoder_dense_1 = L.Linear(8, 512)
            # reshape from (1, 512) to (8, 8, 8)
            self.decoder_conv_0 = L.Convolution2D(8, 8, ksize=3, pad=1) # (8, 8)
            # unpool ksize=2 (16, 16)
            self.decoder_conv_1 = L.Convolution2D(8, 8, ksize=3) # (14, 14)
            # unpool ksize=2 (28, 28)
            self.decoder_conv_2 = L.Convolution2D(8, 16, ksize=4) # (25, 25)
            # unpool ksize=2 (50, 50)
            self.decoder_conv_3 = L.Convolution2D(16, 32, ksize=3, pad=1) # (50, 50)
            # unpool ksize=2 (100, 100)
            self.decoder_output_img = L.Convolution2D(32, in_channels, ksize=3, pad=1) # (100, 100)


    def __call__(self, x):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0])

    def encode(self, x):
        conv_0_encoded = F.relu(self.encoder_conv_0(x)) # (100, 100)
        pool_0_encoded = F.max_pooling_2d(conv_0_encoded, ksize=2) # (50, 50)
        conv_1_encoded = F.relu(self.encoder_conv_1(pool_0_encoded)) # (50, 50)
        pool_1_encoded = F.max_pooling_2d(conv_1_encoded, ksize=2) # (25, 25)
        conv_2_encoded = F.relu(self.encoder_conv_2(pool_1_encoded)) # (22, 22)
        pool_2_encoded = F.max_pooling_2d(conv_2_encoded, ksize=2) # (11, 11)
        conv_3_encoded = F.relu(self.encoder_conv_3(pool_2_encoded)) # (8, 8)
        reshaped_encoded = F.reshape(conv_3_encoded, (len(conv_3_encoded), 1, 512)) # (1, 512)
        dense_0_encoded = self.encoder_dense_0(reshaped_encoded) # (1, 8)
        mu = self.encoder_mu(dense_0_encoded) # (1, 2)
        ln_var = self.encoder_ln_var(dense_0_encoded)  # (1, 2) log(sigma**2)

        return mu, ln_var
        

    def decode(self, z, sigmoid=True):
        dense_0_decoded = F.relu(self.decoder_dense_0(z)) # (1, 8)
        dense_1_decoded = F.relu(self.decoder_dense_1(dense_0_decoded)) # (1, 512)
        reshaped_decoded = F.reshape(dense_1_decoded, (len(dense_1_decoded), 8, 8, 8))# (8, 8)
        deconv_0_decoded = F.relu(self.decoder_conv_0(reshaped_decoded)) # (8, 8)
        up_0_decoded = F.unpooling_2d(deconv_0_decoded, ksize=2, cover_all=False) # (16, 16)
        deconv_1_decoded = F.relu(self.decoder_conv_1(up_0_decoded)) # (14, 14)
        up_1_decoded = F.unpooling_2d(deconv_1_decoded, ksize=2, cover_all=False) # (28, 28)
        deconv_2_decoded = F.relu(self.decoder_conv_2(up_1_decoded)) # (25, 25)
        up_2_decoded = F.unpooling_2d(deconv_2_decoded, ksize=2, cover_all=False) # (50, 50)
        deconv_3_decoded = F.relu(self.decoder_conv_3(up_2_decoded)) # (50, 50)
        up_3_decoded = F.unpooling_2d(deconv_3_decoded, ksize=2, cover_all=False) # (100, 100)
        out_img = self.decoder_output_img(up_3_decoded) # (100, 100)

        # need the check because the bernoulli_nll has a sigmoid in it
        if sigmoid:
            return F.sigmoid(out_img)
        else:
            return out_img
    
    def predict_label(self, mu, ln_var, softmax=True):
        mu_0 = mu[:,0, None]
        mu_1 = mu[:,1, None]

        ln_var_0 = ln_var[:,0, None]
        ln_var_1 = ln_var[:,1, None]

        latent_0 = F.concat((mu_0, ln_var_0), axis=1)
        latent_1 = F.concat((mu_1, ln_var_1), axis=1)
        # latent = mu

        one_hot_vector_0 = self.label_predictor_0(mu_0)
        one_hot_vector_1 = self.label_predictor_1(mu_1)

        # need the check because the bernoulli_nll has a sigmoid in it
        if softmax:
            return F.softmax(one_hot_vector_0), F.softmax(one_hot_vector_1)
        else:
            return one_hot_vector_0, one_hot_vector_1

    def get_latent(self, x):
        mu, ln_var = self.encode(x)
        return F.gaussian(mu, ln_var)
        # return mu


    def get_loss_func(self, k=1):
        """Get loss function of VAE.

        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.

        Args:
            self.beta (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
        """
        def lf(x):

            in_img = x[0]
            in_labels_0 = x[1]
            in_labels_1 = x[2]

            mu, ln_var = self.encode(in_img)
            batchsize = len(mu.data)
            # reconstruction loss
            rec_loss = 0
            label_loss = 0
            label_acc = 0
            for l in six.moves.range(k):
                z = F.gaussian(mu, ln_var)

                out_img = self.decode(z, sigmoid=False)
                rec_loss += F.bernoulli_nll(in_img, out_img) / (k * batchsize)

                out_labels_0, out_labels_1 = self.predict_label(mu, ln_var, softmax=False)
                label_acc += F.accuracy(out_labels_0, in_labels_0)
                label_acc += F.accuracy(out_labels_1, in_labels_1)
                label_loss += F.softmax_cross_entropy(out_labels_0, in_labels_0) / (k * batchsize)
                label_loss += F.softmax_cross_entropy(out_labels_1, in_labels_1) / (k * batchsize)

            self.rec_loss = rec_loss
            self.label_loss = self.gamma * label_loss
            self.label_acc = label_acc

            kl = gaussian_kl_divergence(mu, ln_var) / (batchsize)
            self.kl = self.beta * kl
            
            self.loss = self.rec_loss + self.label_loss + self.kl
            
            return self.loss, self.rec_loss, self.label_loss, self.label_acc, self.kl
        
        return lf