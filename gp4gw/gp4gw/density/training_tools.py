# GPFlow training utils
import gpflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from gpflow.ci_utils import ci_niter
from gpflow import set_trainable
from typing import Tuple, Optional


def mean_squared_error(y, y_pred):
    """
    Function to calculate MSE
    """
    return np.mean((y - y_pred) ** 2)

class HeteroskedasticGaussian(gpflow.likelihoods.Likelihood):
    """
    Class for heteroskedastic errors on the likelihood prediction.
    See https://gpflow.readthedocs.io/en/develop/notebooks/advanced/varying_noise.html
    """
    def __init__(self, **kwargs):
        # this likelihood expects a single latent function F, and two columns in the data matrix Y:
        super().__init__(latent_dim=1, observation_dim=2, **kwargs)

    def _log_prob(self, F, Y):
        # log_prob is used by the quadrature fallback of variational_expectations and predict_log_density.
        # Because variational_expectations is implemented analytically below, this is not actually needed,
        # but is included for pedagogical purposes.
        # Note that currently relying on the quadrature would fail due to https://github.com/GPflow/GPflow/issues/966
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        return gpflow.logdensities.gaussian(Y, F, NoiseVar)

    def _variational_expectations(self, Fmu, Fvar, Y):
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        Fmu, Fvar = Fmu[:, 0], Fvar[:, 0]
        return (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(NoiseVar)
            - 0.5 * (tf.math.square(Y - Fmu) + Fvar) / NoiseVar
        )

    # The following two methods are abstract in the base class.
    # They need to be implemented even if not used.

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError
        
        
def train_exact_heteroskedastic(
    model: gpflow.models.VGP,
    optimizer: tf.optimizers = tf.optimizers.Adam(learning_rate=0.1),
    natgrad_opt: gpflow.optimizers = gpflow.optimizers.NaturalGradient(gamma=1.0),
    epochs: int=100,
    logging_epoch_freq: int=10
):
    """
    Training loop for heteroskedastic GP
    """

    set_trainable(model.q_mu, False)
    set_trainable(model.q_sqrt, False)
    set_trainable(model.mean_function, False)
    
    loss = list()
    for epoch in range(ci_niter(epochs)):
        epoch_id = epoch + 1
        natgrad_opt.minimize(model.training_loss, [(model.q_mu, model.q_sqrt)])
        optimizer.minimize(model.training_loss, model.trainable_variables)
        loss.append(model.training_loss())
        if epoch_id % logging_epoch_freq == 0:
            tf.print(
                f"Epoch {epoch_id}: LOSS (train) {model.training_loss()}"
            )
    plt.plot(range(epochs), loss)
    plt.xlabel('Epoch', fontsize=25)
    plt.ylabel('Loss', fontsize=25)
    plt.tight_layout()

def optimization_exact(model: gpflow.models.GPR,
                     optimizer: tf.optimizers = tf.optimizers.Adam(learning_rate=0.01)):
    """
    Optimisation step for Sparse or Exact GP (no heteroskedastic errors)
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.trainable_variables)
        loss = model.training_loss()
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train_SGPR(
    model: gpflow.models.SGPR,
    epochs: int,
    optimizer: tf.optimizers = tf.optimizers.Adam(learning_rate=0.1),
    logging_epoch_freq: int = 10,
    epoch_var: Optional[tf.Variable] = None,
):
    """
    Training loop for Sparse GP
    """
    set_trainable(model.mean_function, False)
    tf_optimization_step = tf.function(optimization_exact)
    
    loss = list()
    for epoch in range(epochs):
        tf_optimization_step(model)
        if epoch_var is not None:
            epoch_var.assign(epoch + 1)

        epoch_id = epoch + 1
        loss.append(model.training_loss())
        if epoch_id % logging_epoch_freq == 0:
            tf.print(
                f"Epoch {epoch_id}: LOSS (train) {model.training_loss()}"
            )
    plt.plot(range(epochs), loss)
    plt.xlabel('Epoch', fontsize=25)
    plt.ylabel('Loss', fontsize=25)
    plt.tight_layout()

def checkpointing_train_SGPR(
    model: gpflow.models.SGPR,
    X: tf.Tensor,
    Y: tf.Tensor,
    epochs: int,
    manager: tf.train.CheckpointManager,
    optimizer: tf.optimizers = tf.optimizers.Adam(learning_rate=0.1),
    logging_epoch_freq: int = 10,
    epoch_var: Optional[tf.Variable] = None,
    exp_tag: str = 'test',
):
    """
    Training loop for Sparse GP with checkpointing
    """
    set_trainable(model.mean_function, False)
    tf_optimization_step = tf.function(optimization_exact)
    
    loss = list()
    for epoch in range(epochs):
        tf_optimization_step(model)
        if epoch_var is not None:
            epoch_var.assign(epoch + 1)

        epoch_id = epoch + 1
        loss.append(model.training_loss())
        if epoch_id % logging_epoch_freq == 0:
            ckpt_path = manager.save()
            tf.print(
                f"Epoch {epoch_id}: LOSS (train) {model.training_loss()}, saved at {ckpt_path}"
            )
            tf.print(f"MSE: {mean_squared_error(Y, model.predict_y(X)[0])}")
    plt.plot(range(epochs), loss)
    plt.xlabel('Epoch', fontsize=25)
    plt.ylabel('Loss', fontsize=25)
    plt.tight_layout()
