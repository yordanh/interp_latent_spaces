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

    def __init__(self, in_channels, n_latent, n_labels, beta=1):
        super(Conv_VAE, self).__init__()
        with self.init_scope():

            self.in_channels = in_channels
            self.beta = beta
            self.n_labels = n_labels

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

            # label predictor
            self.label_predictor = L.Linear(n_latent, n_labels)

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
    
    def predict_label(self, mu, softmax=True):
        # print(mu)
        # print(mu.data[:,0])
        one_hot_vector = self.label_predictor(mu)

        # need the check because the bernoulli_nll has a sigmoid in it
        if softmax:
            return F.softmax(one_hot_vector)
        else:
            return one_hot_vector

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
            in_labels = x[1]

            mu, ln_var = self.encode(in_img)
            batchsize = len(mu.data)
            # reconstruction loss
            rec_loss = 0
            label_loss = 0
            label_acc = 0
            for l in six.moves.range(k):
                z = F.gaussian(mu, ln_var)

                out_img = self.decode(z, sigmoid=False)
                out_labels = self.predict_label(mu, softmax=False)
                label_acc = F.accuracy(out_labels, in_labels)

                rec_loss += F.bernoulli_nll(in_img, out_img) / (k * batchsize)
                label_loss += F.softmax_cross_entropy(out_labels, in_labels) / (k * batchsize)
                # rec_loss += F.mean_squared_error(in_img, out_img) / (k * batchsize)
                # rec_loss += F.gaussian_nll(in_img, out_img, self.xp.zeros_like(out_img) - 10) / (k * batchsize)
            self.rec_loss = rec_loss
            self.label_loss = 100000 * label_loss
            self.label_acc = label_acc

            # print("Label loss: {}".format(label_loss.data))
            
            kl = gaussian_kl_divergence(mu, ln_var) / (batchsize)
            self.kl = self.beta * kl
            # print("Mean: {0} \t Ln_var: {1} \t KL: {2}".format(mu, ln_var, kl))
            # self.loss = self.rec_loss + 10000000 * self.label_loss + self.beta * kl / batchsize
            self.loss = self.rec_loss + self.label_loss + self.kl
            # chainer.report(
            #     {'rec_loss': rec_loss,'label_loss':self.label_loss , 'loss': self.loss}, observer=self)
            return self.loss, self.rec_loss, self.label_loss, self.label_acc, self.kl
        return lf


# class Conv_VAE_MNIST(chainer.Chain):
#     def __init__(self, in_channels, n_latent, beta=1):
#         super(Conv_VAE_MNIST, self).__init__()
#         with self.init_scope():

#             self.in_channels = in_channels
#             self.beta = beta

#             # encoder
#             self.encoder_conv_0 = L.Convolution2D(in_channels, 32, ksize=3, pad=1) # (28, 28)
#             # max pool ksize=2 (14, 14)
#             self.encoder_conv_1 = L.Convolution2D(32, 16, ksize=3, pad=1) # (14, 14)
#             # max pool ksize=2 (7, 7)
#             self.encoder_conv_2 = L.Convolution2D(16, 8, ksize=4) # (4, 4)
#             self.encoder_dense_0 = L.Linear(128, 8)

#             self.encoder_mu = L.Linear(8, n_latent)
#             self.encoder_ln_var = L.Linear(8, n_latent)

#             # decoder
#             self.decoder_dense_0 = L.Linear(n_latent, 8)
#             self.decoder_dense_1 = L.Linear(8, 128)
#             # reshape from (1, 128) to (8, 4, 4)
#             self.decoder_conv_0 = L.Convolution2D(8, 8, ksize=3, pad=1) # (4, 4)
#             # # unpool ksize=2 (8, 8)
#             self.decoder_conv_1 = L.Convolution2D(8, 16, ksize=2) # (7, 7)
#             # unpool ksize=2 (14, 14)
#             self.decoder_conv_2 = L.Convolution2D(16, 32, ksize=3, pad=1) # (14, 14)
#             # unpool ksize=2 (28, 28)
#             self.decoder_conv_3 = L.Convolution2D(32, in_channels, ksize=3, pad=1) # (28, 28)

#     def __call__(self, x):
#         """AutoEncoder"""
#         return self.decode(self.encode(x)[0])

#     def encode(self, x):
#         conv_0_encoded = F.relu(self.encoder_conv_0(x)) # (28, 28)
#         pool_0_encoded = F.max_pooling_2d(conv_0_encoded, ksize=2) # (14, 14)
#         conv_1_encoded = F.relu(self.encoder_conv_1(pool_0_encoded)) # (14, 14)
#         pool_1_encoded = F.max_pooling_2d(conv_1_encoded, ksize=2) # (7, 7)
#         conv_2_encoded = F.relu(self.encoder_conv_2(pool_1_encoded)) # (4, 4)
#         reshaped_encoded = F.reshape(conv_2_encoded, (len(conv_2_encoded), 1, 128)) # (1, 128)
#         dense_0_encoded = self.encoder_dense_0(reshaped_encoded) # (1, 8)
#         mu = self.encoder_mu(dense_0_encoded) # (1, 2)
#         ln_var = self.encoder_ln_var(dense_0_encoded)  # (1, 2) log(sigma**2)

#         return mu, ln_var
        

#     def decode(self, z, sigmoid=True):
#         dense_0_decoded = F.relu(self.decoder_dense_0(z)) # (1, 8)
#         dense_1_decoded = F.relu(self.decoder_dense_1(dense_0_decoded)) # (1, 128)
#         reshaped_decoded = F.reshape(dense_1_decoded, (len(dense_1_decoded), 8, 4, 4))# (4, 4)
#         deconv_0_decoded = F.relu(self.decoder_conv_0(reshaped_decoded)) # (4, 4)
#         up_0_decoded = F.unpooling_2d(deconv_0_decoded, ksize=2, cover_all=False) # (8, 8)
#         deconv_1_decoded = F.relu(self.decoder_conv_1(up_0_decoded)) # (7, 7)
#         up_1_decoded = F.unpooling_2d(deconv_1_decoded, ksize=2, cover_all=False) # (14, 14)
#         deconv_2_decoded = F.relu(self.decoder_conv_2(up_1_decoded)) # (14, 14)
#         up_2_decoded = F.unpooling_2d(deconv_2_decoded, ksize=2, cover_all=False) # (28, 28)
#         out_img = self.decoder_conv_3(up_2_decoded) # (28, 28)

#         # need the check because the bernoulli_nll has a sigmoid in it
#         if sigmoid:
#             return F.sigmoid(out_img)
#         else:
#             return out_img  

#     def get_latent(self, x):
#         mu, ln_var = self.encode(x)
#         return F.gaussian(mu, ln_var)
#         # return mu


#     def get_loss_func(self, k=1):
#         """Get loss function of VAE.

#         The loss value is equal to ELBO (Evidence Lower Bound)
#         multiplied by -1.

#         Args:
#             self.beta (int): Usually this is 1.0. Can be changed to control the
#                 second term of ELBO bound, which works as regularization.
#             k (int): Number of Monte Carlo samples used in encoded vector.
#         """
#         def lf(x):

#             mu, ln_var = self.encode(x)
#             batchsize = len(mu.data)
#             # reconstruction loss
#             rec_loss = 0
#             for l in six.moves.range(k):
#                 z = F.gaussian(mu, ln_var)
#                 out_img = self.decode(z, sigmoid=False)
#                 rec_loss += F.bernoulli_nll(x, out_img) / (k * batchsize)
#                 # rec_loss += F.mean_squared_error(x, out_img) / (k * batchsize)
#                 # rec_loss += F.gaussian_nll(x, out_img, self.xp.zeros_like(out_img) - 10) / (k * batchsize)
#             self.rec_loss = rec_loss
#             self.loss = self.rec_loss + self.beta * gaussian_kl_divergence(mu, ln_var) / batchsize
#             chainer.report(
#                 {'rec_loss': rec_loss, 'loss': self.loss}, observer=self)
#             return self.loss
#         return lf