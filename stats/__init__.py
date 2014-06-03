from default import wilcox_test, mcnemar_test, chisq_test, rank_biserial, \
        permute_correlation_matrix
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
def to_int(x):
    return x.astype(int)
