"""
This module defines the distance functions.

.. note::

    Features are assumed to be matrices with dimension (sequence, channel).

"""

# pip install dtw-python
from dtw import dtw

import numpy as np


def dtw_dist(features1, features2):
    """
    calculates the normalized DTW distance assumeing that features1 and
    features2 are matrices that share the same number of channels and have the
    dimensions (sequence, channels).

    """
    dist = dtw(features1, features2, distance_only=True).normalizedDistance
    return dist


def avg_dist(features1, features2):
    """
    calculates the RMSE distance between the average vector of features1 and
    features2. It assumes that features1 and features2 are matrices that share
    the same number of channels and have the dimensions (sequence, channels).

    """
    avg1 = features1.mean(axis=0)
    avg2 = features2.mean(axis=0)
    dist = ((avg1 - avg2) ** 2).sum() ** 0.5  # rmse
    return dist


def channel_normalize(features, *, rescale_only=False, means=None, stds=None, backoff=1e-9):
    """
    calculats the channel normalization by substracting the mean and deviding
    by the variance (if rescale_only=False) otherwise only devides by the
    standard deviation.

    .. warning:

        This potentially destroys all relevant information in the features.
        Espeacially, any averaging afterward, will lead to an expected value of
        0 and is therefore meaningless.

    Assumes features has dimensions (sequence, channels).

    """
    if means is None:
        print("WARNING: Channel normalisation potentially removes all relevant information.")
        means = np.mean(features, axis=0)
    if stds is None:
        stds = np.std(features, axis=0)
    if any(stds <= 10 * backoff):
        print("WARNING: Some std are smaller than 10 * backoff parameter in"
              " channel_normalize")
    if rescale_only:
        normalized_features = features / (stds + backoff)
    else:
        normalized_features = (features - means) / (stds + backoff)
    return normalized_features

