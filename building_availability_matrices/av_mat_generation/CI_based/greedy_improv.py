import heapq
import numpy as np
import copy

class GreedyProblem(object):
    """
    Compute-efficient version of ``greedy.GreedyProblem`` (same interface).

    The greedy approximation iteratively adds the element v that maximizes
    the benefit cost ratio among all elements still affordable with the
    remaining budget.
    The current problem does not just have cardinality constraints, it has
    Knapsack constraints (nonnegative modular constraint).
    Reference: KRAUSE, Andreas et GOLOVIN, Daniel. Submodular function maximization.
    Tractability, 2014, vol. 3, no 71-104, p. 3.

    Why it is faster
    ----------------
    The objective is separable by client (row):

        obj(A) = sum_i ( sum_j c[i, j] * A[i, j] ) ** alpha_f ,   c = one_m_GHG_w >= 0

    so the marginal ratio of element (i, j) depends only on its own row sum
    s_i = sum_j c[i, j] * A[i, j]:

        ratio(i, j) = ( (s_i + c[i, j]) ** alpha_f - s_i ** alpha_f ) / GHG[i, j]

    Adding an element to row i only changes the ratios of the other elements
    of row i. Maintaining s_i incrementally makes each ratio O(1), and a
    lazily-evaluated max-heap avoids re-scoring every element each step.

    For alpha_f <= 1 the row term is concave, the objective is submodular, and
    marginal ratios are non-increasing in s_i, so a ratio computed at a smaller
    s_i is an upper bound of the current one. This is exactly the condition that
    makes lazy-greedy (Minoux) evaluation exact: the first *fresh* element popped
    from the heap is the true global maximiser.

    Complexity: ~O(M log M) typical (M = n_clients * n_rounds), versus the
    O(K * M^2) of the reference implementation (K = number of picks).
    """
    def __init__(self, GHG_matrix, alpha_f, w):
        self.GHG_mat = GHG_matrix.to_numpy()
        self.one_m_GHG_w = (np.max(self.GHG_mat) - self.GHG_mat)@np.diag(w)
        # self.one_m_GHG_w = (1 - self.GHG_mat/np.max(self.GHG_mat))@np.diag(w) # test
        # self.one_m_GHG_w = (np.diag(np.max(self.GHG_mat, axis=1))@np.ones((n_clients, n_rounds)) - self.GHG_mat)@np.diag(w) # test
        self.GHG_w = self.GHG_mat@np.diag(w) # test
        self.alpha_f = alpha_f
        self.n_clients = GHG_matrix.shape[0]
        self.n_rounds = GHG_matrix.shape[1]
        self.initialize_A()

    def initialize_A(self):
        self.A = np.zeros(self.GHG_mat.shape)

    def obj_func(self, A):
        # remove the minus here if you want to maximize instead of minimizing (argmin -> argmax):
        return np.sum(np.power(np.sum(np.multiply(self.one_m_GHG_w, A), axis=1), self.alpha_f))
        # return np.sum(np.power(np.sum(np.multiply(self.GHG_w, A), axis=1), self.alpha_f)) # test
        # return np.sum(np.power(np.sum(np.multiply(self.GHG_w, np.ones(A.shape) - A), axis=1), self.alpha_f)) # test

        # return -np.sum(np.sum(np.multiply(self.one_m_GHG_w, A), axis=1))

    def diff(self, i, j):
        A_new = copy.copy(self.A)
        A_new[i, j] = 1
        # return self.obj_func(A_new) - self.obj_func(self.A)
        return (self.obj_func(A_new) - self.obj_func(self.A))/self.GHG_mat[i, j] # test
        # return self.obj_func(A_new) # something else we can optimize

    def update_A(self, i_star, j_star):
        self.A[i_star, j_star] = 1

    def greedy_optimization(self, carbon_budget):
        """
        Lazy-greedy (Minoux) benefit/cost maximisation.

        Reproduces the selections and the stopping rule of the reference
        implementation (stop the first time the best-ratio element does not fit
        the remaining budget), with ties broken by smallest flat index.
        """
        self.initialize_A()
        c = self.one_m_GHG_w          # marginal benefit weights, >= 0
        g = self.GHG_mat              # costs (denominator of the ratio)
        a = self.alpha_f
        r = self.n_rounds

        row_sum = np.zeros(self.n_clients)
        row_ver = np.zeros(self.n_clients, dtype=np.int64)

        def ratio(i, j, s):
            # marginal gain of adding (i, j) when the current row sum of i is s
            return ((s + c[i, j]) ** a - s ** a) / g[i, j]

        # heap entries: (-ratio, flat_index, row_version_when_computed)
        # smallest flat_index breaks ties -> matches the original argmax.
        flat_ratio0 = (np.power(c, a) / g).ravel()          # ratios at s_i = 0
        heap = [(-flat_ratio0[k], k, 0) for k in range(flat_ratio0.size)]
        heapq.heapify(heap)

        selected = np.zeros(self.n_clients * r, dtype=bool)
        G = carbon_budget

        while heap:
            neg_val, flat, ver = heapq.heappop(heap)
            if selected[flat]:
                continue
            i = flat // r
            j = flat - i * r
            if ver != row_ver[i]:
                # stale: recompute against the current row sum and reinsert
                heapq.heappush(heap, (-ratio(i, j, row_sum[i]), flat, row_ver[i]))
                continue

            # `flat` is the current best affordable-agnostic element
            if g[i, j] > G:           # does not fit -> stop (original break rule)
                break

            G -= g[i, j]
            self.update_A(i, j)
            selected[flat] = True
            row_sum[i] += c[i, j]
            row_ver[i] += 1           # lazily invalidates the rest of row i
