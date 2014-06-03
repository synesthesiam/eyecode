import scipy.stats
import numpy as np

def wilcox_test(pop_1, pop_2=None, mu=None):
    """Wilcoxon signed rank test from R (unpaired, not exact).

    Returns
    -------
        p : float
            p-value from Wilcoxon test.
    """
    import pandas.rpy.common as rcom
    wilcox_test_fun = rcom.r["wilcox.test"]

    pop_1 = rcom.robj.FloatVector(pop_1)
    test_result = None
    if pop_2 is None:
        test_result = wilcox_test_fun(pop_1, mu=mu)
    else:
        pop_2 = rcom.robj.FloatVector(pop_2)
        test_result = wilcox_test_fun(pop_1, pop_2)

    return test_result[0][0], test_result[2][0]

def mcnemar_test(results_matrix):
    """Mcnemar chi-squared test from R.

    Returns
    -------
        c, p : tuple
            chi-sq stat and p-value from Mcnemar test.
    """
    import pandas.rpy.common as rcom
    mcnemar_test_fun = rcom.r["mcnemar.test"]

    results_matrix = rcom.convert_to_r_matrix(results_matrix)
    test_result = mcnemar_test_fun(results_matrix)
    return test_result[0][0], test_result[2][0]

def chisq_test(results_matrix):
    """Chi-square test from R.

    Returns
    -------
        c, p : tuple
            chi-sq stat and p-value from test.
    """
    import pandas.rpy.common as rcom
    test_fun = rcom.r["chisq.test"]

    results_matrix = rcom.convert_to_r_matrix(results_matrix)
    test_result = test_fun(results_matrix)
    return test_result[0][0], test_result[2][0]

def rank_biserial(u, n1, n2):
    """Computes the rank biserial correlation for a Mann-Whitney U test.

    Parameters
    ----------
    u : float
        Mann-Whitney U statistic

    n1 : int
        Sample size of group 1

    n2 : int
        Sample size of group 2

    Returns
    -------
    r : float
        Rank biserial correlation. Range in [-1, 1] where 0 means no
        relationship.
    """
    r = 1 - ((2 * u) / float(n1 * n2)) 
    return r

def correlation_method(method):
    if isinstance(method, str):
        if method == "pearson":
            method = scipy.stats.pearsonr
        elif method == "spearman":
            method = scipy.stats.spearmanr
        else:
            raise ValueError("method must be pearson or spearman")
    return method

#def correlation_matrix(data, columns, method="spearman"):
    #assert len(columns) > 0
    #method = correlation_method(method)
    #matrix = np.zeros(shape=(len(columns), len(columns)))

    #for i, c1 in enumerate(columns):
        #for j, c2 in enumerate(columns):
            #if i < j:
                #result, p = method(data[c1], data[c2])
                #matrix[i, j] = result
                #matrix[j, i] = result
            #elif i == j:
                #matrix[i, j] = 1.0

    #return matrix

def permute_correlation_matrix(data, samples=1000, method="spearman",
        interval=(2.5, 97.5), threshold=0.2, permute_corrs=None):
    num_cols = data.shape[1]
    method = correlation_method(method)
    values = data.values.copy()
    orig_corr = method(values)[0]

    permute_shape = (num_cols, num_cols, samples)
    if permute_corrs is not None:
        assert permute_corrs.shape == permute_shape
    else:
        permute_corrs = np.zeros(shape=permute_shape)

    for i in range(samples):
        # Shuffle all columns independently
        for j in range(num_cols):
            np.random.shuffle(values[:, j])

        # Compute correlation on shuffled array
        permute_corrs[:, :, i] = method(values)[0]

    lower = np.percentile(permute_corrs, interval[0], axis=2)
    upper = np.percentile(permute_corrs, interval[1], axis=2)
    sig_corr = ((orig_corr < lower) | (orig_corr > upper)) \
            & (np.abs(orig_corr) >= threshold)

    return orig_corr, sig_corr
