import numpy as np
# Use this for effective implementation of L-BFGS
from collections import defaultdict, deque
from utils import get_line_search_tool
import time


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x  
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """

    x_k = np.copy(x_0)
    # TODO: Implement Conjugate Gradients method.

    g_k = matvec(x_k) - b
    b_norm = np.linalg.norm(b)
    d_k = -g_k

    if trace:
        history = defaultdict(list)
        history['time'].append(0)
        history['residual_norm'].append(np.linalg.norm(g_k))
        history['x'].append(x_k)
    else:
        history = None
    time_start = time.time()

    if max_iter is None:
        max_iter = len(x_k)

    for _ in range(max_iter):
        Ad = matvec(d_k)
        alpha_k = (g_k @ g_k.T) / (d_k.T @ Ad)
        x_k = x_k + alpha_k * d_k
        g_k_new = g_k + alpha_k * Ad
        beta_k = (g_k_new.T @ g_k_new) / (g_k.T @ g_k)
        g_k = g_k_new
        d_k = -g_k + beta_k*d_k

        if trace:
            history['time'].append(time.time() - time_start)
            history['residual_norm'].append(np.linalg.norm(g_k))
            history['x'].append(x_k)

        if np.linalg.norm(g_k) < tolerance*b_norm:
            return x_k, 'success', history

    return x_k, 'iterations_exceeded', history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    def bfgs_multiply(v, H, gamma_0):
        if len(H) == 0:
            return gamma_0 * v
        s, y = H.pop()
        v_mark = v - ((s @ v) / (y @ s)) * y
        z = bfgs_multiply(v_mark, H, gamma_0)
        return z +  ( (s @ v - y @ z) / (y @ s) ) * s

    def lbfgs_direction():
        s, y = H[-1]
        l_0 = (y @ s) / (y @ y)
        H_copy = H.copy()
        return bfgs_multiply(-oracle.grad(x_k), H_copy, l_0)

    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    g_k = oracle.grad(x_k)
    g_k_norm = np.linalg.norm(g_k)
    g_k_norm_start = np.copy(g_k_norm)
    d_k = -g_k

    if trace:
        history = defaultdict(list)
        history['time'].append(0)
        history['grad_norm'].append(g_k_norm)
        history['x'].append(x_k)
        history['func'].append(oracle.func(x_k))
    else:
        history = None

    H = deque()
    timestart = time.time()

    for _ in range(max_iter):
        if len(H)>0:
            d_k = lbfgs_direction()
        alpha_k = line_search_tool.line_search(
            oracle, x_k, d_k, previous_alpha=0.5)  # чтобы пробовать единичный шаг
        x_new = x_k + alpha_k * d_k
        s_k = x_new - x_k
        y_k = oracle.grad(x_new) - oracle.grad(x_k)
        H.append((s_k, y_k))
        if len(H) > memory_size:
            H.popleft()
        x_k = x_new
        g_k = oracle.grad(x_k)
        g_k_norm = np.linalg.norm(g_k)

        history['time'].append(time.time() - timestart)
        history['grad_norm'].append(g_k_norm)
        history['x'].append(x_k)
        history['func'].append(oracle.func(x_k))

        if g_k_norm**2 <= tolerance* g_k_norm_start**2:
            return x_k, "success", history

    return x_k, 'iterations_exceeded', history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500,
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    g_k = oracle.grad(x_k)
    d_k = -g_k
    g_k_norm = np.linalg.norm(g_k)

    if trace:
        history = defaultdict(list)
        history['time'].append(0)
        history['grad_norm'].append(g_k_norm)
        history['x'].append(x_k)
        history['func'].append(oracle.func(x_k))
    else:
        history = None

    time_start = time.time()
    for _ in range(max_iter):
        eps_k = min(0.5, g_k_norm**0.5) * g_k_norm
        def matvec(x): return oracle.hess_vec(x_k, x)
        d_k = conjugate_gradients(matvec, -g_k, d_k, eps_k)[0]
        while oracle.grad(x_k)@d_k > 0:
            eps_k = eps_k / 10
            d_k = conjugate_gradients(matvec, -g_k, d_k, eps_k)[0]
        alpha_k = line_search_tool.line_search(
            oracle, x_k, d_k, previous_alpha=0.5)  # чтобы пробовать единичный шаг
        x_k = x_k + alpha_k * d_k
        g_k = oracle.grad(x_k)
        g_k_norm = np.linalg.norm(g_k)

        if trace:
            history['time'].append(time.time() - time_start)
            history['grad_norm'].append(np.linalg.norm(g_k))
            history['x'].append(x_k)
            history['func'].append(oracle.func(x_k))

        if g_k_norm**2 < tolerance:
            return x_k, 'success', history

    return x_k, 'iterations_exceeded', history


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)

    grad_0 = oracle.grad(x_0)
    norm_start = np.linalg.norm(grad_0)
    x_k = np.copy(x_0)
    grad_norm = np.copy(norm_start)
    alpha=None

    time_start = time.time()
    for _ in range(max_iter):
        grad = oracle.grad(x_k)
        grad_norm = np.linalg.norm(grad)
        alpha = line_search_tool.line_search(oracle, x_k, -grad,previous_alpha=alpha)

        if trace:
            history['time'].append(time.time() - time_start)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm)
            history['relative_norm'].append(grad_norm**2/norm_start**2)
            history['x'].append(x_k)

        x_k = x_k-alpha*grad

        if not np.all(np.isfinite(grad)) or not np.all(np.isfinite(grad_norm)) or not np.all(np.isfinite(x_k)):
            return x_k, 'computational_error', history

        if grad_norm**2 <= tolerance * norm_start**2:
            return x_k, 'success', history

    return x_k, 'iterations_exceeded', history