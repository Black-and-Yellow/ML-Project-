import numpy as np
import time
from typing import Dict, Tuple, Any
from LSSMO import LSSMO


def Large_Scale_NonL_LSTWSVM(A: np.ndarray, A_test: np.ndarray, FunPara: Dict) -> Tuple[float, np.ndarray, np.ndarray, float, Dict]:
    """
    Large Scale Non-Linear Least Squares Twin Support Vector Machine
    
    Parameters:
    -----------
    A : np.ndarray
        Training data with labels in last column
    A_test : np.ndarray
        Test data with labels in last column
    FunPara : dict
        Dictionary containing parameters:
        - c0: fuzzy membership parameter
        - ir: imbalance ratio
        - c1: regularization parameter for model 1
        - c3: regularization parameter (c3=c4)
        - kerfPara: kernel parameters dictionary
        - eps_val: (optional) tolerance value
    
    Returns:
    --------
    acc : float
        Classification accuracy (percentage)
    obsX : np.ndarray
        True labels from test data
    Predict_Y : np.ndarray
        Predicted labels
    time : float
        Training time
    output_struct : dict
        Dictionary containing function name
    """
    
    start_time = time.time()
    
    # Extract parameters
    c0 = FunPara['c0']
    ir = FunPara['ir']
    
    # Apply fuzzy membership (assuming nufuzz2 is defined elsewhere)
    C = nufuzz2(A, c0, ir)
    del A
    
    # Separate membership values and data
    Amem = C[C[:, -2] == 1, -1]
    Bmem = C[C[:, -2] != 1, -1]
    C = C[:, :-1]

    # Print training class distribution (how many classes and counts per class)
    try:
        labels, counts = np.unique(C[:, -1], return_counts=True)
        print(f"Training classes found: {len(labels)}")
        for lab, cnt in zip(labels, counts):
            # labels are expected to be 1 or -1; cast to int for neat printing
            print(f"  Class {int(lab)}: {cnt} samples")
    except Exception:
        # If something unexpected happens, don't break the pipeline
        print("Could not determine training class distribution")
    
    # Separate positive and negative class samples
    A = C[C[:, -1] == 1, :-1]
    B = C[C[:, -1] != 1, :-1]
    del C
    
    # Model-1 parameters
    c1 = FunPara['c1']
    c2 = c1  # c1 = c2
    c3 = FunPara['c3']  # c3 = c4
    c4 = c3
    kerfPara = FunPara['kerfPara']
    eps = 1e-4  # For NDC eps=1e-2 and for small scale eps=1e-4
    
    p = A.shape[0]
    q = B.shape[0]
    
    # Construct vectors
    v1 = np.concatenate([np.zeros(p), np.ones(q)])
    v2 = np.concatenate([np.zeros(q), np.ones(p)])
    
    # Model-1: Construct Q matrix
    Inv_S22 = 1.0 / Bmem
    Inv_S2 = np.diag(Inv_S22)
    IQ = Inv_S2.T @ Inv_S2
    del Inv_S22, Inv_S2, Bmem
    
    # Construct kernel matrices
    AA = kernelfun(A, kerfPara, A) + c3 * np.eye(p)
    AB = kernelfun(A, kerfPara, B)
    BA = kernelfun(B, kerfPara, A)
    BB = kernelfun(B, kerfPara, B) + (c3 / c1) * IQ
    
    # Construct Q matrix
    Q_top = np.hstack([AA, AB])
    Q_bottom = np.hstack([BA, BB])
    Q = np.vstack([Q_top, Q_bottom]) + np.ones((p + q, p + q))
    
    # Solve for x using LSSMO
    x = LSSMO(Q, eps, c3, v1)
    
    del IQ, Q
    
    # Extract alpha and beta
    alpha = x[:p]
    beta = x[p:]
    
    # Model-2: Construct H matrix
    Inv_S11 = 1.0 / Amem
    Inv_S1 = np.diag(Inv_S11)
    IP = Inv_S1.T @ Inv_S1
    del Inv_S11, Inv_S1, Amem
    
    # Construct kernel matrices for model 2
    BB2 = kernelfun(B, kerfPara, B) + c4 * np.eye(q)
    BA2 = kernelfun(B, kerfPara, A)
    AB2 = kernelfun(A, kerfPara, B)
    AA2 = kernelfun(A, kerfPara, A) + (c4 / c2) * IP
    
    # Construct H matrix
    H_top = np.hstack([BB2, BA2])
    H_bottom = np.hstack([AB2, AA2])
    H = np.vstack([H_top, H_bottom]) + np.ones((p + q, p + q))
    
    # Solve for y using LSSMO
    y = LSSMO(H, eps, c4, v2)
    
    elapsed_time = time.time() - start_time
    
    del IP, H
    
    # Extract lambda and gamma
    lambda_vec = y[:q]
    gamma = y[q:]
    
    # Testing phase
    TestX = A_test
    P1 = TestX[:, :-1]
    obsX = TestX[:, -1]
    del TestX

    # Print test class distribution (how many classes and counts per class)
    try:
        test_labels, test_counts = np.unique(obsX, return_counts=True)
        print(f"Test classes found: {len(test_labels)}")
        for lab, cnt in zip(test_labels, test_counts):
            print(f"  Class {int(lab)}: {cnt} samples")
    except Exception:
        print("Could not determine test class distribution")
    
    e1 = np.ones(q)
    e2 = np.ones(p)
    
    # Calculate bias terms
    b1 = (e2.T @ alpha + e1.T @ beta) / c3
    b2 = -(e1.T @ lambda_vec + e2.T @ gamma) / c4
    
    # Calculate decision functions
    y1 = (kernelfun(P1, kerfPara, A) @ alpha + kernelfun(P1, kerfPara, B) @ beta) / c3 + b1
    y2 = -(kernelfun(P1, kerfPara, B) @ lambda_vec + kernelfun(P1, kerfPara, A) @ gamma) / c4 + b2
    
    # Predict classes
    Predict_Y = np.zeros(len(y1))
    for i in range(len(y1)):
        if min(abs(y1[i]), abs(y2[i])) == abs(y1[i]):
            Predict_Y[i] = 1
        else:
            Predict_Y[i] = -1
    
    # Calculate accuracy
    acc = np.sum(obsX == Predict_Y) / len(Predict_Y) * 100
    
    output_struct = {'function_name': 'Large_Scale_NonL_LSTWSVM'}
    
    return acc, obsX, Predict_Y, elapsed_time, output_struct


def kernelfun(X: np.ndarray, kerfPara: Dict, Y: np.ndarray) -> np.ndarray:
    """
    Kernel function matching MATLAB kernelfun.m implementation
    
    Parameters:
    -----------
    X : np.ndarray
        First data matrix (Xtrain in MATLAB)
    kerfPara : dict
        Kernel parameters: {'type': 'rbf'/'lin'/'poly', 'pars': [param1, param2, ...]}
    Y : np.ndarray
        Second data matrix (Xt in MATLAB)
    
    Returns:
    --------
    omega : np.ndarray
        Kernel matrix
    """
    kernel_type = kerfPara['type']
    kernel_pars = kerfPara['pars']
    
    nb_data = X.shape[0]
    
    if kernel_type == 'rbf':
        # RBF kernel: exp(-||x-y||^2 / (2*sigma^2))
        # kernel_pars[0] is sigma^2
        omega = -2 * X @ Y.T
        XXh = np.sum(X**2, axis=1).reshape(-1, 1) @ np.ones((1, Y.shape[0]))
        Yh = np.sum(Y**2, axis=1).reshape(-1, 1) @ np.ones((1, nb_data))
        omega = omega + XXh + Yh.T
        omega = np.exp(-omega / (2 * kernel_pars[0]))
        
    elif kernel_type == 'lin':
        # Linear kernel
        omega = X @ Y.T
        
    elif kernel_type == 'poly':
        # Polynomial kernel: (X*Y' + kernel_pars[0])^kernel_pars[1]
        omega = (X @ Y.T + kernel_pars[0]) ** kernel_pars[1]
        
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    return omega


def nufuzz2(A: np.ndarray, c0: float, ir: float) -> np.ndarray:
    """
    Fuzzy membership function - matches MATLAB nufuzz2.m implementation
    
    Parameters:
    -----------
    A : np.ndarray
        Input data with labels in last column
    c0 : float
        Fuzzy parameter (controls membership spread)
    ir : float
        Imbalance ratio
    
    Returns:
    --------
    finA : np.ndarray
        Data with class indicator in second-to-last column and 
        membership values in the last column
    """
    # Separate positive and negative classes
    A_pos = A[A[:, -1] == 1, :-1]
    B_neg = A[A[:, -1] != 1, :-1]
    
    x, y = A_pos.shape
    x1, y1 = B_neg.shape
    
    # Redefine C to keep order consistent with A and B
    C = np.vstack([
        np.column_stack([A_pos, np.ones(x)]),
        np.column_stack([B_neg, -np.ones(x1)])
    ])
    
    no_input, no_col = C.shape
    obs = C[:, -1]
    
    # Calculate class centroids
    s = np.sum(A_pos, axis=0) / x  # Positive class centroid
    h = np.sum(B_neg, axis=0) / x1  # Negative class centroid
    
    # Calculate maximum distance in negative class from its centroid
    DiffB = B_neg - np.tile(h, (B_neg.shape[0], 1))
    distancec1 = np.sqrt(np.diag(DiffB @ DiffB.T))
    rn = np.max(distancec1)
    
    # Distance between centroids
    diff = s - h
    db = np.sqrt(diff @ diff.T)
    
    # Calculate fuzzy membership for each sample
    memb = np.zeros((no_input, 1))
    
    for i in range(no_input):
        diff = C[i, :no_col-1] - s
        dist1 = np.sqrt(diff @ diff.T)  # Distance to positive centroid
        
        diff = C[i, :no_col-1] - h
        dist2 = np.sqrt(diff @ diff.T)  # Distance to negative centroid
        
        if obs[i] == 1:
            # Positive class samples get membership = 1
            memb[i, 0] = 1
        else:
            # Negative class samples get fuzzy membership based on exponential formula
            exp_arg = c0 * ((dist1 - dist2) / db - dist2 / rn)
            numerator = np.exp(exp_arg) - np.exp(-2 * c0)
            denominator = np.exp(c0) - np.exp(-2 * c0)
            memb[i, 0] = (1 / (ir + 1)) + (ir / (ir + 1)) * (numerator / denominator)
    
    finA = np.column_stack([C, memb])
    return finA