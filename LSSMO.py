import numpy as np


def LSSMO(Q: np.ndarray, eps: float, c: float, v: np.ndarray) -> np.ndarray:
    """
    Least Squares Sequential Minimal Optimization solver
    
    Parameters:
    -----------
    Q : np.ndarray
        Positive definite matrix
    eps : float
        Termination condition (tolerance)
    c : float
        Weight to be tuned (from cross validation)
    v : np.ndarray
        Vector
    
    Returns:
    --------
    bestx : np.ndarray
        Solution vector
    
    Example:
    --------
    Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    eps = 0.0001
    c = 2
    v = np.array([0, 1, 1])
    bestx = LSSMO(Q, eps, c, v)
    """
    
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


# Example usage
if __name__ == "__main__":
    # Test the LSSMO function
    Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    eps = 0.0001
    c = 2
    v = np.array([0, 1, 1], dtype=float)
    
    result = LSSMO(Q, eps, c, v)
    print("Solution vector:")
    print(result)