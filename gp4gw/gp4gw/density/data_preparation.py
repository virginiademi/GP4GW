import numpy as np
import arviz as az
import tensorflow as tf
from uncertainties import unumpy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import namedtuple
import itertools


def centroid_calc(d):
    """
    Function calculates centroid of bin.
    """
    return ((d[1:] - d[:-1]) / 2) + d[0:-1]

def create_2d_data_set(bins):
    """
    Create 2D grid of points
    """
    dim = 2
    centroid_array = np.array([centroid_calc(bins[i]) for i in range(dim)]).T
    print("Shape of binning:", centroid_array.shape)

    x = []
    for p1, p2 in itertools.product(centroid_array[:, 0], centroid_array[:, 1]):
        x.extend([(p1, p2)])
    x = np.array(x)
    print("Histogram points:", x.shape)
    return x

def create_4d_data_set(bins):
    """
    Create 4D grid of points
    """
    dim = 4
    centroid_array = np.array([centroid_calc(bins[i]) for i in range(dim)]).T
    print("Shape of binning:", centroid_array.shape)

    x = []
    for p1, p2, p3, p4 in itertools.product(
        centroid_array[:, 0],
        centroid_array[:, 1],
        centroid_array[:, 2],
        centroid_array[:, 3],
    ):
        x.extend([(p1, p2, p3, p4)])
    x = np.array(x)
    print("Histogram points:", x.shape)
    return x

def scale_data(x, y):
    """
    Scale x-data according to be in range [0,1].
    Scale y-data such that the mean is zero.
    """
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    assert x.dtype == y.dtype
    dtype = x.dtype
    N, Q = x.shape
    y_scaler = StandardScaler()
    x_scaler = MinMaxScaler()
    y_scaler.fit(y.reshape([-1, 1]))
    x_scaler.fit(x)
    
    y_scaled = y_scaler.transform(y.reshape([-1, 1])).astype(dtype).reshape([-1])
    x_scaled = x_scaler.transform(x).astype(dtype)
    print('Shape of x-data: N=%.f, Q=%.f' % (N, Q))
    return x_scaled, y_scaled, x_scaler, y_scaler

def calc_hist_errors(samples, bin_array, y_scaler):
    """
    Function to calculate errors from histogram density estimate.
    The error is given by Poisson noise ~ np.sqrt(N_counts).
    Returns errors both in "ral" space and "scaled" spacee.
    """
    if len(samples.shape) > 1:
        # multidimensional histogram
        raw_counts, bins = np.histogramdd(samples, density=False, bins=bin_array)
        vol = 1
        # Todo: this can be vectorized
        for i in range(len(bin_array)):
            vol = vol * np.diff(bin_array[i])[0]
    else:
        raw_counts, bins = np.histogram(samples, density=False, bins=bin_array)

        vol = np.diff(bin_array)

    error = np.sqrt(raw_counts)
    count_with_error = unumpy.uarray(raw_counts, error)
    dens_with_error = count_with_error / vol / count_with_error.sum()
    dens_with_error = dens_with_error.flatten()
                                          
    # Scale errors with y-scaling
    normed_dens_with_error = dens_with_error - y_scaler.mean_
    normed_dens_with_error = normed_dens_with_error / y_scaler.scale_

    dens_std_unscaled = unumpy.std_devs(dens_with_error)
    dens_std = unumpy.std_devs(normed_dens_with_error)
    return dens_std, dens_std_unscaled

def convert_to_arviz_data(tf_samples, parameters_scaler, param_names):
    """
    Returns 'InferenceData' posterior object
    :param tf_samples: mcmc or hmc samples from tf.run_chain 
    :param parameters_scaler: scaler for samples
    :param param_names: list of parameter names of samples
    :return: arviz posterior sample object
    """
    new_samples = np.swapaxes(tf_samples, 0, 1)
    samples = []
    for i in range(len(new_samples)):
        chain = parameters_scaler.inverse_transform(new_samples[i])
        samples.append(chain)
    new_samples = np.array(samples)
    dims = {key: new_samples.T[i].T for i, key in enumerate(param_names)}
    data = az.from_dict(dims)
    return data

def calc_mean_and_ci(samples, quantiles=(0.16, 0.84), fmt=".2f"):
    """
    Function to calculate mean and confidence interval
    Copied from https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/core/result.py#L767
    """
    summary = namedtuple("summary", ["median", "lower", "upper", "string"])
    quants_to_compute = np.array([quantiles[0], 0.5, quantiles[1]])
    quants = np.percentile(samples, quants_to_compute * 100)
    summary.median = quants[1]
    summary.plus = quants[2] - summary.median
    summary.minus = summary.median - quants[0]
    fmt = "{{0:{0}}}".format(fmt).format
    string_template = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    summary.string = string_template.format(
        fmt(summary.median), fmt(summary.minus), fmt(summary.plus)
    )

    return summary