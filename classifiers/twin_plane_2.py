# Twin SVM: hyperplane 2
import numpy as np
from numpy import linalg

try:
    from cvxopt import solvers, matrix
    CVXOPT_AVAILABLE = True
except ImportError:
    CVXOPT_AVAILABLE = False
    print("cvxopt not installed; install pip install cvxopt")


def twin_plane_2(class_A, class_B, C2, epsilon=1e-5, regularization=1e-8):
    # Compute second Twin SVM hyperplane
    if not CVXOPT_AVAILABLE:
        raise ImportError("cvxopt is required. Install with: pip install cvxopt")
    
    # Compute A^T A with regularization for numerical stability
    AtA = np.dot(class_A.T, class_A)
    AtA = AtA + regularization * np.identity(AtA.shape[0])
    
    # Solve A^T A * B^T using linear system solver
    AtA_inv_Bt = linalg.solve(AtA, class_B.T)
    
    # Compute B * (A^T A)^-1 * B^T
    Bt_AtA_inv_Bt = np.dot(class_B, AtA_inv_Bt)
    
    # Make symmetric for numerical stability
    Bt_AtA_inv_Bt = (Bt_AtA_inv_Bt + Bt_AtA_inv_Bt.T) / 2
    
    m2 = class_B.shape[0]
    e2 = -np.ones((m2, 1))
    
    # Disable cvxopt output
    solvers.options['show_progress'] = False
    
    # Box constraints: 0 <= beta <= C2
    lower_bound = np.zeros((m2, 1))
    upper_bound = C2 * np.ones((m2, 1))
    
    # Constraint matrix: beta <= upper_bound and -beta <= -lower_bound
    G = np.vstack((np.identity(m2), -np.identity(m2)))
    h = np.vstack((upper_bound, -lower_bound))
    
    # Solve QP: min 0.5 * beta^T * P * beta + q^T * beta
    # subject to G * beta <= h
    solution = solvers.qp(
        matrix(Bt_AtA_inv_Bt, tc='d'),
        matrix(e2, tc='d'),
        matrix(G, tc='d'),
        matrix(h, tc='d')
    )
    
    beta = np.array(solution['x'])
    
    # Compute hyperplane parameters
    z = -np.dot(AtA_inv_Bt, beta)
    w2 = z[:-1].flatten()
    b2 = float(z[-1])
    
    return [w2, b2]
