# Style transfer network from "A Learned Representation for Artistic Style"
# (Dumoulin, Shlens, Kudlur, 2016), ported from Magenta's image_stylization
# model (Apache 2.0) to plain TensorFlow 2.
#
# Weights are read straight out of Magenta's original TF1 checkpoints, so
# neither magenta nor tf_slim is needed at runtime.
"""Multi-style pastiche generator."""

import numpy as np
import tensorflow as tf

_EPSILON = 1e-5


class Stylizer:
    """Loads one pastiche-generator checkpoint and stylizes images with it."""

    def __init__(self, checkpoint_path):
        reader = tf.train.load_checkpoint(checkpoint_path)
        # The published checkpoints contain some all-NaN filters in
        # contract/conv1. Their InstanceNorm gamma is exactly 0 for every style,
        # so those channels are meant to output a constant beta; zeroing the
        # weights gives exactly that without NaNs leaking through the network.
        self._vars = {
            name: tf.constant(np.nan_to_num(reader.get_tensor(name), nan=0.0))
            for name in reader.get_variable_to_shape_map()
        }
        self.num_styles = int(
            self._vars['transformer/contract/conv1/InstanceNorm/beta'].shape[0])
        self._transform = tf.function(
            self._forward,
            input_signature=[
                tf.TensorSpec([1, None, None, 3], tf.float32),
                tf.TensorSpec([self.num_styles], tf.float32),
            ])

    def stylize(self, image, weights):
        """Stylizes an image.

        Args:
          image: float32 array of shape [height, width, 3] with values in [0, 1].
          weights: dict mapping style index to weight. {3: 1.0} applies style 3;
            {3: 0.5, 7: 0.5} blends two styles.

        Returns:
          float32 array of shape [height, width, 3] with values in [0, 1].
        """
        mixture = np.zeros([self.num_styles], dtype=np.float32)
        for index, weight in weights.items():
            if not 0 <= index < self.num_styles:
                raise ValueError('style index %d out of range [0, %d)' %
                                 (index, self.num_styles))
            mixture[index] = weight
        height, width = image.shape[:2]
        out = self._transform(image[None].astype(np.float32), mixture)
        # The two stride-2 layers round up, so the output can be a few pixels
        # larger than the input.
        return out[0, :height, :width].numpy()

    def _forward(self, image, mixture):
        h = self._conv(image, 'contract/conv1', 9, 1, mixture)
        h = self._conv(h, 'contract/conv2', 3, 2, mixture)
        h = self._conv(h, 'contract/conv3', 3, 2, mixture)
        for i in range(1, 6):
            h = self._residual(h, 'residual/residual%d' % i, mixture)
        h = self._upsample(h, 'expand/conv1', mixture)
        h = self._upsample(h, 'expand/conv2', mixture)
        return self._conv(h, 'expand/conv3/conv', 9, 1, mixture,
                          activation=tf.nn.sigmoid)

    def _conv(self, x, scope, kernel_size, stride, mixture,
              activation=tf.nn.relu):
        """Reflect-padded convolution + weighted instance norm + activation."""
        prefix = 'transformer/' + scope + '/'
        pad = kernel_size // 2
        x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
        x = tf.nn.conv2d(x, self._vars[prefix + 'weights'], stride, 'VALID')

        beta = tf.tensordot(mixture, self._vars[prefix + 'InstanceNorm/beta'], 1)
        gamma = tf.tensordot(mixture, self._vars[prefix + 'InstanceNorm/gamma'], 1)
        mean, variance = tf.nn.moments(x, [1, 2], keepdims=True)
        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, _EPSILON)
        return activation(x) if activation else x

    def _residual(self, x, scope, mixture):
        h = self._conv(x, scope + '/conv1', 3, 1, mixture)
        h = self._conv(h, scope + '/conv2', 3, 1, mixture, activation=None)
        return x + h

    def _upsample(self, x, scope, mixture):
        """Nearest-neighbour 2x upsampling followed by a convolution."""
        shape = tf.shape(x)
        x = tf.image.resize(x, [2 * shape[1], 2 * shape[2]], method='nearest')
        return self._conv(x, scope + '/conv', 3, 1, mixture)
