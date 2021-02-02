"""
bnn_model.py

This module implements a Bayesian neural network in tensorflow probability.
It reuses code from

    @misc{Hafner2018,
    	title={Noise Contrastive Priors for Functional Uncertainty},
    	author={Danijar Hafner and Dustin Tran and Timothy Lillicrap and Alex Irpan and James Davidson},
    	year={2018, accessed 28.09.2020},
    	eprint={1807.09289},
    	archivePrefix={arXiv},
    	url={https://github.com/brain-research/ncp}
    }

The _network_ function defines the network layers and is called by define_graph.
The _define_graph_ function initializes the corresponding tensors in a graph.

The defined graph can be accessed and run in a tf.Session via BNN.graph
"""

from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from classes.attrdict import AttrDict
import numpy as np

from .base_model import BaseModel

# defines the network architecture and makes it Bayes by defining prior, posterior
# and a DenseReparameterization which implements the inference via stochastic
# forward passes such that an tf.optimizer can be used for training
def network(inputs, config):
    # classical dense connected layers
    hidden = inputs
    for i, size in enumerate(config.layer_sizes):
        hidden = tf.layers.dense(hidden, size, tf.nn.leaky_relu)

    # define the posterior as an independent normal with given std
    init_std = np.log(np.exp(config.weight_std) - 1).astype(np.float32)
    kernel_posterior = tfd.Independent(
        tfd.Normal(
            tf.get_variable(
                "kernel_mean",
                (hidden.shape[-1].value, 1),
                tf.float32,
                tf.random_normal_initializer(0, config.weight_std),
            ),
            tf.nn.softplus(
                tf.get_variable(
                    "kernel_std",
                    (hidden.shape[-1].value, 1),
                    tf.float32,
                    tf.constant_initializer(init_std),
                )
            ),
        ),
        2,
    )
    # prior is a normal as well and has the same shape as the posterior
    kernel_prior = tfd.Independent(
        tfd.Normal(
            tf.zeros_like(kernel_posterior.mean()),
            tf.zeros_like(kernel_posterior.mean()) + tf.nn.softplus(init_std),
        ),
        2,
    )
    # for the bias, the posterior is simply a constant
    bias_prior = None
    bias_posterior = tfd.Deterministic(
        tf.get_variable("bias_mean", (1,), tf.float32, tf.constant_initializer(0.0))
    )
    # add the KL divergence of prior and posterior to tf.GraphKeys.REGULARIZATION_LOSSES
    # to add them later in the loss computation
    tf.add_to_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES,
        tfd.kl_divergence(kernel_posterior, kernel_prior),
    )
    # make the network probabilistic
    mean = tfp.layers.DenseReparameterization(
        1,
        kernel_prior_fn=lambda *args, **kwargs: kernel_prior,
        kernel_posterior_fn=lambda *args, **kwargs: kernel_posterior,
        bias_prior_fn=lambda *args, **kwargs: bias_prior,
        bias_posterior_fn=lambda *args, **kwargs: bias_posterior,
    )(hidden)
    # define output
    mean_dist = tfd.Normal(
        tf.matmul(hidden, kernel_posterior.mean()) + bias_posterior.mean(),
        tf.sqrt(tf.matmul(hidden ** 2, kernel_posterior.variance())),
    )
    std = tf.layers.dense(hidden, 1, tf.nn.softplus) + 1e-6
    data_dist = tfd.Normal(mean, std)
    return data_dist, mean_dist


# defines the computational graph in tensorflow
# contains the tensors for the network model, as well as the optimizer
def define_graph(config):
    network_tpl = tf.make_template("network", network, config=config)
    inputs = tf.placeholder(tf.float32, [None, config.num_inputs], name="inputs")
    targets = tf.placeholder(tf.float32, [None, 1], name="targets")
    num_samples = tf.placeholder(tf.int32, [], name="num_samples")
    num_rand_samples = tf.placeholder(tf.int32, [])
    batch_size = tf.shape(inputs)[0]

    data_dist, mean_dist = network_tpl(inputs)
    assert len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    divergence = sum(
        [
            tf.reduce_sum(tensor)
            for tensor in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        ]
    )
    num_batches = tf.cast(num_samples, float) / tf.cast(batch_size, float)
    losses = [
        config.divergence_scale * divergence / num_batches,
        -data_dist.log_prob(targets),
    ]
    loss = sum(tf.reduce_sum(loss) for loss in losses) / tf.cast(batch_size, float)
    optimizer = tf.train.AdamOptimizer(config.learning_rate)
    gradients, variables = zip(
        *optimizer.compute_gradients(loss, colocate_gradients_with_ops=True)
    )
    if config.clip_gradient:
        gradients, _ = tf.clip_by_global_norm(gradients, config.clip_gradient)
    optimize = optimizer.apply_gradients(zip(gradients, variables))
    reset_optimizer = tf.variables_initializer(optimizer.variables())

    data_mean = mean_dist.mean()
    data_mean_sample = mean_dist.sample(num_rand_samples)
    data_noise = data_dist.stddev()
    data_uncertainty = mean_dist.stddev()

    return AttrDict(locals())


class BNN(BaseModel):
    def __init__(self, net_config, train_schedule):
        super().__init__(net_config)
        self.graph = define_graph(net_config)
        self.ts = train_schedule
        self.has_uncertainty = True

    def sample_y(self, X_data, num_rand_samples=1):
        feed_dict = {
            self.graph.inputs: X_data,
            self.graph.num_rand_samples: num_rand_samples,
        }
        y_data = self.session.run(self.graph.data_mean_sample, feed_dict=feed_dict)
        y_data = y_data.reshape(num_rand_samples, len(X_data))
        return y_data
