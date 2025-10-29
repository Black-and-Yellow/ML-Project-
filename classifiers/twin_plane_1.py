# Twin SVM: hyperplane 1
import numpy as np
from numpy import linalg

try:
    from cvxopt import solvers, matrix
    CVXOPT_AVAILABLE = True
except ImportError:
    CVXOPT_AVAILABLE = False
    print("cvxopt not installed; install pip install cvxopt")


def twin_plane_1(class_A, class_B, C1, epsilon=1e-5, regularization=1e-8):
    # Compute first Twin SVM hyperplane
    if not CVXOPT_AVAILABLE:
        raise ImportError("cvxopt is required. Install with: pip install cvxopt")
    
    # Compute B^T B with regularization for numerical stability
    BtB = np.dot(class_B.T, class_B)
    BtB = BtB + regularization * np.identity(BtB.shape[0])
    
    # Solve B^T B * A^T using linear system solver
    BtB_inv_At = linalg.solve(BtB, class_A.T)
    
    # Compute A * (B^T B)^-1 * A^T
    At_BtB_inv_At = np.dot(class_A, BtB_inv_At)
    
    # Make symmetric for numerical stability
    At_BtB_inv_At = (At_BtB_inv_At + At_BtB_inv_At.T) / 2
    
    m1 = class_A.shape[0]
    e1 = -np.ones((m1, 1))
    
    # Disable cvxopt output
    solvers.options['show_progress'] = False
    
    # Box constraints: 0 <= alpha <= C1
    lower_bound = np.zeros((m1, 1))
    upper_bound = C1 * np.ones((m1, 1))
    
    # Constraint matrix: alpha <= upper_bound and -alpha <= -lower_bound
    G = np.vstack((np.identity(m1), -np.identity(m1)))
    h = np.vstack((upper_bound, -lower_bound))
    
    # Solve QP: min 0.5 * alpha^T * P * alpha + q^T * alpha
    # subject to G * alpha <= h
    solution = solvers.qp(
        matrix(At_BtB_inv_At, tc='d'),
        matrix(e1, tc='d'),
        matrix(G, tc='d'),
        matrix(h, tc='d')
    )
    
    alpha = np.array(solution['x'])
    
    # Compute hyperplane parameters
    z = -np.dot(BtB_inv_At, alpha)
    w1 = z[:-1].flatten()
    b1 = float(z[-1])
    
    return [w1, b1]
