import numpy as np
from abc import ABC, abstractmethod 
from scipy.stats import multivariate_normal
# def Forward(O, A, B, pi):
#     T = len(O)
#     N = A.shape[0]
#     alpha = np.zeros((N, T))
#     alpha[:, 0] = pi * B[:, O[0]]
#     for t in range(1, T):
#         Prob_Ot = B[:, O[t]]
#         alpha[:, t] = (A.T @ alpha[:, t-1]) * Prob_Ot
#     prob = np.sum(alpha[:,T-1])
#     return prob, alpha

# def Viterbi(O, A, B, pi):
#     T = len(O)
#     N = A.shape[0]
#     delta = np.zeros((N, T))
#     phi = np.zeros((N, T))
#     delta[:, 0] = pi * B[:, O[0]]
#     for t in range(1, T):
#         Prob_Ot = B[:, O[t]]
#         diag_matrix = np.diag(delta[:, t-1]) 
#         prob_matrix = A.T @ diag_matrix
#         delta[:, t] = np.max(prob_matrix, axis=1) * Prob_Ot
#         phi[:, t] = np.argmax(prob_matrix, axis=1)
#     path = np.zeros(T, dtype=np.int32)
#     state_prob_max = np.argmax(delta[:, T-1])
#     path[T-1] = state_prob_max
#     for t in range(T-2, -1, -1):
#         path[t] = phi[path[t+1], t+1]
#     return path, delta, phi

# def Backward(O, A, B):
#     T = len(O)
#     N = A.shape[0]
#     beta = np.zeros((N,T))
#     beta[:, T-1] = 1
#     for t in range(T-2, -1, -1):
#         beta[:, t] = A @ (beta[:, t+1]*B[:, O[t+1]])
#     return beta

# def Forward_log(O, A, B, pi):
#     T = len(O)
#     N = A.shape[0]
#     logA = np.log(A)
#     logB = np.log(B)
#     logpi = np.log(pi)
#     log_alpha = np.full((N, T), -np.inf)
#     log_alpha[:, 0] = logpi + logB[:, O[0]]
#     for t in range(1, T):
#         for j in range(N):
#             tmp = log_alpha[:, t-1] + logA[:, j]
#             m = np.max(tmp)
#             log_sum = m + np.log(np.sum(np.exp(tmp - m)))
#             log_alpha[j, t] = log_sum + logB[j, O[t]]
#     last = log_alpha[:, T-1]
#     m = np.max(last)
#     log_prob = m + np.log(np.sum(np.exp(last - m)))
#     return log_prob, log_alpha

# def Viterbi_log(O, A, B, pi):
#     T = len(O)
#     N = A.shape[0]
#     logA = np.log(A)
#     logB = np.log(B)
#     logpi = np.log(pi)
#     delta = np.full((N, T), -np.inf)
#     phi = np.zeros((N, T), dtype=np.int32)
#     delta[:, 0] = logpi + logB[:, O[0]]
#     for t in range(1, T):
#         for j in range(N):
#             scores = delta[:, t-1] + logA[:, j]
#             phi[j, t] = np.argmax(scores)
#             delta[j, t] = np.max(scores) + logB[j, O[t]]
#     path = np.zeros(T, dtype=np.int32)
#     path[T-1] = np.argmax(delta[:, T-1])
#     for t in range(T-2, -1, -1):
#         path[t] = phi[path[t+1], t+1]
#     log_path_prob = np.max(delta[:, T-1])
#     return path, delta, phi, log_path_prob


# def Backward_log(O, A, B):
#     T = len(O)
#     N = A.shape[0]
#     logA = np.log(A)
#     logB = np.log(B)
#     beta = np.full((N, T), -np.inf)
#     beta[:, -1] = 0.0
#     for t in range(T-2, -1, -1):
#         for i in range(N):
#             tmp = logA[i, :] + logB[:, O[t+1]] + beta[:, t+1]
#             m = np.max(tmp)
#             beta[i, t] = m + np.log(np.sum(np.exp(tmp - m)))
#     return beta

        # self.A = np.random.rand(N, N)
        # self.A = np.log(self.A/self.A.sum(axis=1, keepdims=True))
        # self.pi = np.random.rand(N)
        # self.pi = np.log(self.pi/self.pi.sum())
def logsumexp(x, axis=None):
    m = np.max(x, axis=axis, keepdims=True)
    s = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(s, axis=axis)

class BaseHMM(ABC):
    def __init__(self, A: np.ndarray, pi: np.ndarray):
        self.N = A.shape[0]
        self.A = A
        self.pi = pi
        self.logA = np.log(A)
        self.logpi = np.log(pi)

    @abstractmethod
    def log_B(self, o_t) -> np.ndarray:
        raise NotImplementedError

    def forward(self, O):
        T = len(O)
        log_alpha = np.full((self.N, T), -np.inf)
        log_alpha[:, 0] = self.logpi + self.log_B(O[0])
        for t in range(1, T):
            e = self.log_B(O[t])
            tmp = log_alpha[:, t-1][:, None] + self.logA
            log_alpha[:, t] = logsumexp(tmp, axis=0) + e
        log_prob = logsumexp(log_alpha[:, -1])
        return log_prob, log_alpha
    
    def backward(self, O):
        T = len(O)
        logBeta = np.full((self.N, T), -np.inf)
        logBeta[:, T-1] = 0.0
        for t in range(T-2, -1, -1):
            e_next = self.log_B(O[t+1])
            tmp = self.logA + e_next[None, :] + logBeta[:, t+1][None, :]
            logBeta[:, t] = logsumexp(tmp, axis=1)
        return logBeta
    
    def viterbi(self, O):
        T = len(O)
        delta = np.full((self.N, T), -np.inf)
        phi = np.zeros((self.N, T), dtype=np.int32)
        delta[:, 0] = self.logpi + self.log_B(O[0])
        for t in range(1, T):
            e = self.log_B(O[t])
            scores = delta[:, t-1][:, None] + self.logA
            phi[:, t] = np.argmax(scores, axis=0)
            delta[:, t] = np.max(scores, axis=0) + e
        path = np.zeros(T, dtype=np.int32)
        path[T-1] = np.argmax(delta[:, T-1])
        for t in range(T-2, -1, -1):
            path[t] = phi[path[t+1], t+1]
        log_path_prob = np.max(delta[:, -1])
        return path, delta, phi, log_path_prob

    @abstractmethod
    def fit(self, data, n_loop=50, bound_learning = 1e-4): 
        raise NotImplementedError

class discreteHMM(BaseHMM):
    def __init__(self, A: np.ndarray, B: np.ndarray, pi: np.ndarray):
        super().__init__(A, pi)
        self.B = B
        self.logB = np.log(B)

    def log_B(self, o_t) -> np.ndarray:
        return self.logB[:, o_t] 
    
    def fit(self, data, n_loop=50, bound_learning = 1e-4):        
        M = self.B.shape[1]
        last_log_likelihood = -np.inf
        for i in range(n_loop):
            pi_numerator = np.zeros(self.N)
            A_numerator = np.zeros((self.N, self.N))
            A_denominator = np.zeros((self.N, 1)) # Tổng gamma (trừ T)
            B_numerator = np.zeros((self.N, M))
            B_denominator = np.zeros((self.N, 1)) # Tổng gamma
            
            total_log_likelihood = 0

            # --- E-Step:
            for O in data:
                T = len(O)
                
                # 1. Tính alpha, beta và log-likelihood của chuỗi
                log_prob, log_alpha = self.forward(O) # log_alpha (N, T)
                logBeta = self.backward(O)          # logBeta (N, T)
                total_log_likelihood += log_prob

                # 2. Tính gamma: log_gamma[i, t] = P(X_t=i | O, lambda)
                log_gamma = log_alpha + logBeta - log_prob
                gamma = np.exp(log_gamma) # (N, T)

                # 3. Tính xi: log_xi[i, j, t] = P(X_t=i, X_{t+1}=j | O, lambda)
                log_xi = np.zeros((self.N, self.N, T - 1))
                for t in range(T - 1):
                    b_next = self.log_B(O[t+1]) # (N,)
                    # tmp[i, j] = log(alpha_t(i)) + log(a_ij) + log(b_j(O_{t+1})) + log(beta_{t+1}(j))
                    tmp = log_alpha[:, t][:, None] + self.logA + b_next[None, :] + logBeta[:, t+1][None, :]
                    log_xi[:, :, t] = tmp - log_prob # Chuẩn hóa
                
                xi = np.exp(log_xi) # (N, N, T-1)

                pi_numerator += gamma[:, 0]
                A_numerator += np.sum(xi, axis=2) # Tổng theo thời gian
                A_denominator += np.sum(gamma[:, :-1], axis=1, keepdims=True)
                B_denominator += np.sum(gamma, axis=1, keepdims=True)
                O_matrix = np.zeros((T, M))
                O_matrix[np.arange(T), O] = 1.0 # O_matrix[t, O[t]] = 1
                B_numerator += gamma @ O_matrix

            # --- M-Step:
            
            # Cẩn thận: Dùng np.nan_to_num để xử lý phép chia 0/0 (nếu có)
            self.pi = pi_numerator / len(data)
            self.A = np.nan_to_num(A_numerator / A_denominator)
            self.B = np.nan_to_num(B_numerator / B_denominator)
            
            # Chuẩn hóa
            self.pi /= np.sum(self.pi)
            # Chuẩn hóa A
            row_sums_A = np.sum(self.A, axis=1, keepdims=True)
            row_sums_A[row_sums_A == 0] = 1.0 # Tránh chia cho 0
            self.A = self.A / row_sums_A
            # Chuẩn hóa B
            row_sums_B = np.sum(self.B, axis=1, keepdims=True)
            row_sums_B[row_sums_B == 0] = 1.0 # Tránh chia cho 0
            self.B = self.B / row_sums_B
            # Cập nhật lại log (thêm 1e-10 để tránh log(0))
            self.logpi = np.log(self.pi + 1e-10)
            self.logA = np.log(self.A + 1e-10)
            self.logB = np.log(self.B + 1e-10)

            if abs(total_log_likelihood - last_log_likelihood) < bound_learning:
                break
            last_log_likelihood = total_log_likelihood
        return self
class continueHMM(BaseHMM):
    def __init__(self, A: np.ndarray, means: np.ndarray, covariances: np.ndarray, pi: np.ndarray):
        super().__init__(A, pi)
        self.means = means
        self.covariances = covariances
        self.D = means.shape[1]

        assert self.pi.shape == (self.N,), "Kích thước 'pi' không hợp lệ"
        assert self.A.shape == (self.N, self.N), "Kích thước 'A' không hợp lệ"
        assert self.means.shape == (self.N, self.D), "Kích thước 'means' không hợp lệ"
        assert self.covariances.shape == (self.N, self.D, self.D), "Kích thước 'covariances' không hợp lệ"
        
        self.logpi = np.log(self.pi + 1e-10)
        self.logA = np.log(self.A + 1e-10)
        self._precompute_emission_params()
    
    def _precompute_emission_params(self):
        """Tiền tính precision matrix và log normalizing constant cho tốc độ."""
        self._precisions = np.zeros_like(self.covariances)
        self._log_norm_consts = np.zeros(self.N)
        eye = np.eye(self.D)
        
        for i in range(self.N):
            # Ổn định covariance
            cov = self.covariances[i] + 1e-6 * eye
            # Đảm bảo đối xứng
            cov = 0.5 * (cov + cov.T)
            
            # Dùng Cholesky để tính logdet và precision hiệu quả
            try:
                L = np.linalg.cholesky(cov)
                logdet = 2.0 * np.sum(np.log(np.diag(L)))
                self._log_norm_consts[i] = -0.5 * (self.D * np.log(2.0 * np.pi) + logdet)
                
                # precision = inv(cov) = inv(L^T) @ inv(L)
                Linv = np.linalg.inv(L)
                self._precisions[i] = Linv.T @ Linv
            except np.linalg.LinAlgError:
                # Fallback nếu không PD
                self._precisions[i] = np.linalg.pinv(cov)
                sign, logdet = np.linalg.slogdet(cov)
                self._log_norm_consts[i] = -0.5 * (self.D * np.log(2.0 * np.pi) + logdet)

    def log_B(self, O_t) -> np.ndarray:
        """Tính log b_i(O_t) cho tất cả state i. Tối ưu bằng vector operations."""
        # O_t: (D,) -> diff: (N, D)
        diff = O_t[None, :] - self.means
        # Mahalanobis distance: quad[i] = diff[i]^T @ precision[i] @ diff[i]
        quad = np.einsum('nd,ndk,nk->n', diff, self._precisions, diff)
        return self._log_norm_consts - 0.5 * quad

    def log_B_sequence(self, O: np.ndarray) -> np.ndarray:
        """Tính log b_i(O_t) cho toàn bộ chuỗi. Trả về (N, T)."""
        T = O.shape[0]
        # diff: (N, T, D) = O[None,:,:] - means[:,None,:]
        diff = O[None, :, :] - self.means[:, None, :]
        # quad: (N, T)
        quad = np.einsum('ntd,ndk,ntk->nt', diff, self._precisions, diff)
        return self._log_norm_consts[:, None] - 0.5 * quad

    def forward(self, O):
        """Tối ưu: tính log_B cho toàn bộ chuỗi một lần."""
        T = O.shape[0]
        log_b_all = self.log_B_sequence(O)  # (N, T)
        
        log_alpha = np.full((self.N, T), -np.inf)
        log_alpha[:, 0] = self.logpi + log_b_all[:, 0]
        
        for t in range(1, T):
            tmp = log_alpha[:, t-1][:, None] + self.logA
            log_alpha[:, t] = logsumexp(tmp, axis=0) + log_b_all[:, t]
        
        log_prob = logsumexp(log_alpha[:, -1])
        return log_prob, log_alpha, log_b_all  # Trả về thêm log_b_all để tái sử dụng
    
    def backward(self, O, log_b_all=None):
        """Tối ưu: nhận log_b_all từ forward để tránh tính lại."""
        T = O.shape[0]
        if log_b_all is None:
            log_b_all = self.log_B_sequence(O)
        
        logBeta = np.full((self.N, T), -np.inf)
        logBeta[:, T-1] = 0.0
        
        for t in range(T-2, -1, -1):
            tmp = self.logA + log_b_all[:, t+1][None, :] + logBeta[:, t+1][None, :]
            logBeta[:, t] = logsumexp(tmp, axis=1)
        
        return logBeta

    def fit(self, data, n_loop=50, bound_learning=1e-4):
        last_log_likelihood = -np.inf
        
        for i_loop in range(n_loop):
            # Accumulators
            pi_numerator = np.zeros(self.N)
            A_numerator = np.zeros((self.N, self.N))
            A_denominator = np.zeros((self.N, 1))
            gamma_sum = np.zeros((self.N, 1))
            means_numerator = np.zeros((self.N, self.D))
            cov_numerator = np.zeros((self.N, self.D, self.D))
            total_log_likelihood = 0

            for O in data:
                T = O.shape[0]
                if T == 0:
                    continue
                
                # 1. Forward & backward (tái sử dụng log_b_all)
                log_prob, log_alpha, log_b_all = self.forward(O)
                logBeta = self.backward(O, log_b_all)
                total_log_likelihood += log_prob

                # 2. Gamma
                log_gamma = log_alpha + logBeta - log_prob
                gamma = np.exp(log_gamma)  # (N, T)

                # 3. Xi 
                if T > 1:
                    # log_xi: (N, N, T-1)
                    # tmp[i,j,t] = log_alpha[i,t] + logA[i,j] + log_b[j,t+1] + logBeta[j,t+1]
                    tmp = (log_alpha[:, :-1].reshape(self.N, 1, T-1)      # (N, 1, T-1)
                           + self.logA[:, :, None]                         # (N, N, 1)
                           + log_b_all[None, :, 1:]                        # (1, N, T-1)
                           + logBeta[None, :, 1:])                         # (1, N, T-1)
                    log_xi = tmp - log_prob
                    xi = np.exp(log_xi)  # (N, N, T-1)
                    
                    A_numerator += xi.sum(axis=2)
                    A_denominator += gamma[:, :-1].sum(axis=1, keepdims=True)

                # 4. Accumulate
                pi_numerator += gamma[:, 0]
                gamma_sum += gamma.sum(axis=1, keepdims=True)
                means_numerator += gamma @ O  # (N,T) @ (T,D)

                # Covariance - tối ưu bằng einsum
                # cov_numerator[i] += sum_t gamma[i,t] * (O[t] @ O[t].T)
                # = gamma[i,:] @ (O @ O.T) nhưng phải cẩn thận chiều
                # Cách hiệu quả: dùng einsum
                cov_numerator += np.einsum('nt,td,tk->ndk', gamma, O, O)

            # M-step
            self.pi = pi_numerator / len(data)
            self.pi /= self.pi.sum()
            
            self.A = np.nan_to_num(A_numerator / (A_denominator + 1e-12))
            self.A /= self.A.sum(axis=1, keepdims=True)
            
            self.means = np.nan_to_num(means_numerator / (gamma_sum + 1e-12))
            
            S2_term = np.nan_to_num(cov_numerator / (gamma_sum[:, :, None] + 1e-12))
            mean_outers = self.means[:, :, None] @ self.means[:, None, :]
            self.covariances = S2_term - mean_outers
            
            # Đảm bảo đối xứng và PD
            self.covariances = 0.5 * (self.covariances + self.covariances.transpose(0, 2, 1))
            self.covariances += 1e-6 * np.eye(self.D)[None, :, :]
            
            # Update
            self.logpi = np.log(self.pi + 1e-10)
            self.logA = np.log(self.A + 1e-10)
            self._precompute_emission_params()

            if abs(total_log_likelihood - last_log_likelihood) < bound_learning:
                print(f"Converged at iteration {i_loop+1}")
                break
            last_log_likelihood = total_log_likelihood
            
        return self