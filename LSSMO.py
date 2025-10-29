import numpy as np


def LSSMO(Q: np.ndarray, eps: float, c: float, v: np.ndarray) -> np.ndarray:
    # Least squares SMO solver
    
    no_row, m = Q.shape
    x = np.zeros(no_row)  # Initialize vector x
    F = np.zeros(no_row)  # Vector of differentials
    D = np.zeros(no_row)  # Delta values
    
    # Initialize F and D vectors
    for i in range(no_row):
        Fi = -x.T @ Q[:, i] - c * v[i]
        Di = Fi * Fi / (2 * Q[i, i])
        F[i] = Fi
        D[i] = Di
    
    normF = np.linalg.norm(F)
    
    # Main optimization loop
    # Norm of F should be zero or close to zero (less than the defined tolerance)
    iter_count = 0
    max_iter = 500
    
    while normF > eps * no_row and iter_count < max_iter:
        # Find the index with maximum D value
        max_idx = np.argmax(D)
        
        # Update x
        t = F[max_idx] / Q[max_idx, max_idx]
        x[max_idx] = x[max_idx] + t
        
        # Recalculate F and D for all components
        for i in range(no_row):
            Fi = -x.T @ Q[:, i] - c * v[i]
            Di = Fi * Fi / (2 * Q[i, i])
            F[i] = Fi
            D[i] = Di
        
        normF = np.linalg.norm(F)
        iter_count += 1
    
    bestx = x
    return bestx


if __name__ == "__main__":
    # Demo removed
    pass