from default import wilcox_test, mcnemar_test, chisq_test
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
def to_int(x):
    return x.astype(int)
