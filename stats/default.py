def wilcox_test(pop_1, pop_2=None, mu=None):
    """Wilcoxon signed rank test from R.

    Returns
    -------
        p : float
            p-value from Wilcoxon test.
    """
    import pandas.rpy.common as rcom
    wilcox_test_fun = rcom.r["wilcox.test"]

    pop_1 = rcom.robj.FloatVector(pop_1)
    if pop_2 is None:
        return wilcox_test_fun(pop_1, mu=mu)[2][0]
    else:
        pop_2 = rcom.robj.FloatVector(pop_2)
        return wilcox_test_fun(pop_1, pop_2)[2][0]
