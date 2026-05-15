"""
MIMO-RLS-LQR Adaptive Governor for Multi-Cluster DVFS
Author: Stanislav Usychenko
"""

import numpy as np
from scipy.linalg import solve_discrete_are
from typing import Tuple, Optional, List


class MIMO_RLS_LQR_Governor:
    """
    Core class implementing MIMO-RLS online identification + LQR optimal control.
    """
    def __init__(self,
                 n_states: int = 6,
                 n_inputs: int = 2,
                 lambda_forget: float = 0.98,
                 q_diag: Optional[List[float]] = None,
                 r_diag: Optional[List[float]] = None):
        
        self.nx = n_states
        self.nu = n_inputs
        self.lambda_ = lambda_forget
        
        # MIMO-RLS parameters
        self.theta = np.zeros((self.nx, self.nx + self.nu))  # [A | B]
        self.P = np.eye(self.nx + self.nu) * 1000.0           # Covariance matrix
        
        # LQR cost matrices
        if q_diag is None:
            q_diag = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]   # State penalties
        if r_diag is None:
            r_diag = [0.1, 0.1]                        # Input penalties
            
        self.Q = np.diag(q_diag)
        self.R = np.diag(r_diag)
        
        self.K: Optional[np.ndarray] = None  # Feedback gain matrix

    def update_rls(self, x: np.ndarray, x_next: np.ndarray, u: np.ndarray) -> float:
        """One step MIMO Recursive Least Squares update"""
        phi = np.hstack([x, u]).reshape(-1, 1)   # Regressor [x; u]
        y = x_next.reshape(-1, 1)                 # Target
        
        y_hat = self.theta @ phi
        error = y - y_hat
        
        # RLS update with forgetting factor
        denom = self.lambda_ + phi.T @ self.P @ phi
        L = (self.P @ phi) / denom
        
        self.theta += error @ L.T
        self.P = (self.P - L @ phi.T @ self.P) / self.lambda_
        
        return float(np.mean(np.abs(error)))

    def compute_lqr_gain(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute optimal LQR gain K"""
        try:
            P = solve_discrete_are(A, B, self.Q, self.R)
            K = np.linalg.solve(self.R + B.T @ P @ B, B.T @ P @ A)
            return K
        except np.linalg.LinAlgError:
            # Fallback stable gain if Riccati fails
            return 0.01 * np.ones((self.nu, self.nx))

    def get_control(self, x: np.ndarray) -> np.ndarray:
        """Compute control action u = -Kx with saturation"""
        if self.K is None:
            # Initial stabilizing control
            return -0.05 * np.clip(x[:self.nu], -1.0, 1.0)
        
        u = -self.K @ x
        return np.clip(u, -0.4, 0.4)   # Limit frequency change

    def extract_ab(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract current A and B matrix estimates"""
        A = self.theta[:, :self.nx]
        B = self.theta[:, self.nx:]
        return A, B