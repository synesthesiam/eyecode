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
