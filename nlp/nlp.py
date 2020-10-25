import numpy as np


def tf_idf(i, j):
    """Calculates tf-idf score for each word in the
    FIRST corpus passed."""
    tf = i / (i + j)

    i_count = (i != 0).astype(int)
    j_count = (j != 0).astype(int)
    idf = np.log((i_count + j_count) / i_count)

    return tf * idf


def logodds_dirichlet(i, j, prior=None):
    """Calculate log-odds, uninformative Dirichlet prior
    (Monroe et al. 2008, sec. 3.3.4). The higher a word's
    value, the more "representative" it is of corpus i.
    A values of zero means no preference between corpora."""

    # Add one to each observation, per Laplace's Rule of
    # Succession, so we don't end up dividing by zero.
    i += 1
    j += 1

    if prior is None:
        prior = i + j

    # Compute word frequency totals
    n_i = np.sum(i)
    n_j = np.sum(j)
    n_prior = np.sum(prior)  # α_0

    # Compute the prior log odds
    delta = np.log((i + prior) / (n_i + n_prior - i - prior)) - \
            np.log((j + prior) / (n_j + n_prior - j - prior))

    # Compute the standard deviation
    sigma = np.sqrt(1/(i + prior) + 1/(j + prior))

    # Convert to z-score
    return delta / sigma
