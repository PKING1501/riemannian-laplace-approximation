import numpy as np
import sys

import torch
from scipy.integrate import solve_ivp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
# import sklearn.neighbors.graph as knn_graph
from scipy.interpolate import CubicSpline
# from csaps import csaps


# Draws an elipsoid that correspond to the metric
def plot_metric(x, cov, color='r', inverse_metric=False, linewidth=1):
    eigvals, eigvecs = np.linalg.eig(cov)
    N = 100
    theta = np.linspace(0, 2 * np.pi, N)
    theta = theta.reshape(N, 1)
    points = np.concatenate((np.cos(theta), np.sin(theta)), axis=1)
    points = points * np.sqrt(eigvals)
    points = np.matmul(eigvecs, points.transpose()).transpose()
    points = points + x.flatten()
    plt.plot(points[:, 0], points[:, 1], c=color, linewidth=linewidth, label='Metric')


# This function evaluates the differential equation c'' = f(c, c')
def geodesic_system(manifold, c, dc):
    # Input: c, dc ( D x N )

    D, N = c.shape
    if (dc.shape[0] != D) | (dc.shape[1] != N):
        print('geodesic_system: second and third input arguments must have same dimensionality\n')
        sys.exit(1)

    # Evaluate the metric and the derivative
    M, dM = manifold.metric_tensor(c, nargout=2)

    # Prepare the output (D x N)
    ddc = np.zeros((D, N))

    # Diagonal Metric Case, M (N x D), dMdc_d (N x D x d=1,...,D) d-th column derivative with respect to c_d
    if manifold.is_diagonal():
        for n in range(N):
            dMn = np.squeeze(dM[n, :, :])
            ddc[:, n] = -0.5 * (2 * np.matmul(dMn * dc[:, n].reshape(-1, 1), dc[:, n])
                                - np.matmul(dMn.T, (dc[:, n] ** 2))) / M[n, :]

    # Non-Diagonal Metric Case, M ( N x D x D ), dMdc_d (N x D x D x d=1,...,D)
    else:
        M_inv = np.linalg.inv(M)  # N x D x D
        Term1 = dM.reshape(N, D, D * D, order='F')  # N x D x D^2
        Term2 = dM.reshape(N, D * D, D, order='F')  # N x D^2 x D

        for n in range(N):
            # Mn = np.squeeze(M[n, :, :])
            # if np.linalg.cond(Mn) < 1e-15:
            #     print('Ill-condition metric!\n')
            #     sys.exit(1)

            # dvecMdcn = dM[n, :, :, :].reshape(D * D, D, order='F')
            # blck = np.kron(np.eye(D), dc[:, n])

            ddc[:, n] = -0.5 * M_inv[n, :, :] @ ((2 * Term1[n, :, :] - Term2[n, :, :].T) @ np.kron(dc[:, n], dc[:, n]))
    return ddc


# This function changes the 2nd order ODE to two 1st order ODEs takes c, dc and returns dc, ddc.
def second2first_order(manifold, state, subset_of_weights):
    # Input: state [c; dc] (2D x N), y=[dc; ddc]: (2D x N)
    D = int(state.shape[0] / 2)

    # TODO: Something better for this?
    if state.ndim == 1:
        state = state.reshape(-1, 1)  # (2D,) -> (2D, 1)

    c = state[:D, :]  # D x N
    cm = state[D:, :]  # D x N
    if subset_of_weights == 'last_layer':
        # in the last layer case we can use the old implementation
        cmm = geodesic_system(manifold, c, cm)  # D x N
    else:
        # if we want full network we use the hvp implementation
        cmm = manifold.geodesic_system(c, cm)
    y = np.concatenate((cm, cmm), axis=0)
    return y

# ATTEMPT 4:
# from torchdiffeq import odeint as torch_odeint
# import torch
# import numpy as np

# def second2first_order(manifold, state, subset_of_weights):
#     """Convert second-order ODE to first-order system for PyTorch ODE solver."""
#     D = state.shape[0] // 2
    
#     # Ensure proper dimensions and tensor type
#     if state.ndim == 1:
#         state = state.unsqueeze(-1)  # (2D,) -> (2D, 1)
    
#     # Split position and velocity
#     c = state[:D, :]  # (D, N)
#     cm = state[D:, :]  # (D, N)
    
#     # Compute acceleration using manifold geometry
#     if subset_of_weights == 'last_layer':
#         with torch.enable_grad():
#             c = c.requires_grad_(True)
#             # Ensure all manifold operations return tensors
#             metric = manifold.metric(c)
#             christoffel = manifold.christoffel(c)(cm, cm)
#             prior = manifold.prior_term(c)
            
#             # Verify tensor types
#             if not isinstance(metric, torch.Tensor):
#                 metric = torch.tensor(metric, dtype=c.dtype, device=c.device)
#             if not isinstance(christoffel, torch.Tensor):
#                 christoffel = torch.tensor(christoffel, dtype=c.dtype, device=c.device)
#             if not isinstance(prior, torch.Tensor):
#                 prior = torch.tensor(prior, dtype=c.dtype, device=c.device)
                
#             cmm = -torch.linalg.solve(metric, christoffel + prior)
#     else:
#         with torch.enable_grad():
#             c = c.requires_grad_(True)
#             cmm = manifold.geodesic_system(c, cm)
#             if not isinstance(cmm, torch.Tensor):
#                 cmm = torch.tensor(cmm, dtype=c.dtype, device=c.device)
    
#     return torch.cat([cm, cmm], dim=0)

# If the solver failed provide the linear distance as the solution
def evaluate_failed_solution(p0, p1, t):
    # Input: p0, p1 (D x 1), t (T x 0)
    c = (1 - t) * p0 + t * p1  # D x T
    dc = np.repeat(p1 - p0, np.size(t), 1)  # D x T
    return c, dc


# If the solver_bvp() succeeded provide the solution.
def evaluate_solution(solution, t, t_scale):
    # Input: t (Tx0), t_scale is used from the Expmap to scale the curve in order to have correct length,
    #        solution is an object that solver_bvp() returns
    c_dc = solution.sol(t * t_scale)
    D = int(c_dc.shape[0] / 2)

    # TODO: Why the t_scale is used ONLY for the derivative component?
    if np.size(t) == 1:
        c = c_dc[:D].reshape(D, 1)
        dc = c_dc[D:].reshape(D, 1) * t_scale
    else:
        c = c_dc[:D, :]  # D x T
        dc = c_dc[D:, :] * t_scale  # D x T
    return c, dc


def evaluate_spline_solution(curve, dcurve, t):
    # Input: t (Tx0), t_scale is used from the Expmap to scale the curve in order to have correct length,
    #        solution is an object that solver_bvp() returns
    c = curve(t)
    dc = dcurve(t)
    D = int(c.shape[0])

    # TODO: Why the t_scale is used ONLY for the derivative component?
    if np.size(t) == 1:
        c = c.reshape(D, 1)
        dc = dc.reshape(D, 1)
    else:
        c = c.T  # Because the c([0,..,1]) -> N x D
        dc = dc.T
    return c, dc


# This function computes the infinitesimal small length on a curve
def local_length(manifold, curve, t):
    # Input: curve function of t returns (D X T), t (T x 0)
    c, dc = curve(t)  # [D x T, D x T]
    D = c.shape[0]
    M = manifold.metric_tensor(c, nargout=1)
    if manifold.is_diagonal():
        dist = np.sqrt(np.sum(M.transpose() * (dc ** 2), axis=0))  # T x 1, c'(t) M(c(t)) c'(t)
    else:
        dc = dc.T  # D x N -> N x D
        dc_rep = np.repeat(dc[:, :, np.newaxis], D, axis=2)  # N x D -> N x D x D
        Mdc = np.sum(M * dc_rep, axis=1)  # N x D
        dist = np.sqrt(np.sum(Mdc * dc, axis=1))  # N x 1
    return dist


# This function computes the length of the geodesic curve
# The smaller the approximation error (tol) the slower the computation.
def curve_length(manifold, curve, a=0, b=1, tol=1e-5, limit=50):
    # Input: curve a function of t returns (D x ?), [a,b] integration interval, tol error of the integration
    if callable(curve):
        # function returns: curve_length_eval = (integral_value, some_error)
        curve_length_eval = integrate.quad(lambda t: local_length(manifold, curve, t), a, b, epsabs=tol, limit=limit)  # , number of subintervals
    else:
        print("TODO: Not implemented yet integration for discrete curve!\n")
        sys.exit(1)

    return curve_length_eval[0]


# This function plots a curve that is given as a parametric function, curve: t -> (D x len(t)).
def plot_curve(curve, **kwargs):
    N = 1000
    T = np.linspace(0, 1, N)
    curve_eval = curve(T)[0]

    D = curve_eval.shape[0]  # Dimensionality of the curve

    if D == 2:
        plt.plot(curve_eval[0, :], curve_eval[1, :], **kwargs)
    elif D == 3:
        plt.plot(curve_eval[0, :], curve_eval[1, :], curve_eval[2, :], **kwargs)

# This function vectorizes an matrix by stacking the columns
def vec(x):
    # Input: x (NxD) -> (ND x 1)
    return x.flatten('F').reshape(-1, 1)



# This function implements the exponential map
def expmap(manifold, x, v, subset_of_weights='all'):
    assert subset_of_weights == 'all' or subset_of_weights == 'last_layer', 'subset_of_weights must be all or last_layer'

    # Input: v,x (Dx1)
    x = x.reshape(-1, 1)
    v = v.reshape(-1, 1)
    D = x.shape[0]

    ode_fun = lambda t, c_dc: second2first_order(manifold, c_dc, subset_of_weights).flatten()  # The solver needs this shape (D,)
    if np.linalg.norm(v) > 1e-5:
        # print('I think we should enter here')
        curve, failed = new_solve_expmap(manifold, x, v, ode_fun, subset_of_weights)
    else:
        curve = lambda t: (x.reshape(D, 1).repeat(np.size(t), axis=1),
                           v.reshape(D, 1).repeat(np.size(t), axis=1))  # Return tuple (2D x T)
        failed = True

    return curve, failed


# This function solves the initial value problem for the implementation of the expmap
def new_solve_expmap(manifold, x, v, ode_fun, subset_of_weights):
    D = x.shape[0]

    if isinstance(v, torch.Tensor):
        v = v.cpu().numpy()
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
        
    init = np.concatenate((x, v), axis=0).flatten()  # 2D x 1 -> (2D, ), the solver needs this shape

    failed = False

    prev_t = 0
    t = 1

    solution = solve_ivp(ode_fun, [prev_t, t], init, dense_output=True, atol = 1e-3, rtol= 1e-6)  # First solution of the IVP problem
    curve = lambda tt: evaluate_solution(solution, tt, 1)  # with length(c(t)) != ||v||_c
    
    return curve, failed


# # ATTEMPT 4:
# def new_solve_expmap(manifold, x, v, ode_fun, subset_of_weights):
#     """Optimized exponential map solver with full PyTorch compatibility."""
#     device = x.device if isinstance(x, torch.Tensor) else 'cpu'
#     dtype = x.dtype if isinstance(x, torch.Tensor) else torch.float32
    
#     # Convert and validate inputs
#     x = x if isinstance(x, torch.Tensor) else torch.from_numpy(np.asarray(x))
#     v = v if isinstance(v, torch.Tensor) else torch.from_numpy(np.asarray(v))
#     x = x.reshape(-1).to(device=device, dtype=dtype)
#     v = v.reshape(-1).to(device=device, dtype=dtype)
    
#     # Combine initial state
#     init = torch.cat([x, v])  # (2D,)
    
#     # Time points to evaluate at (fewer points for speed)
#     t_eval = torch.linspace(0, 1, 5, device=device, dtype=dtype)
    
#     # Define the ODE function with tensor type checking
#     def torch_ode_fun(t, y):
#         # Ensure input is tensor
#         if not isinstance(y, torch.Tensor):
#             y = torch.tensor(y, device=device, dtype=dtype)
#         return second2first_order(manifold, y, subset_of_weights)
    
#     try:
#         # Solve with adaptive step size
#         solution = torch_odeint(
#             torch_ode_fun,
#             init,
#             t_eval,
#             method='dopri5',
#             rtol=1e-4,
#             atol=1e-6,
#             options={'max_num_steps': 500}
#         ).transpose(0, 1)  # (time, dim)
        
#         # Create robust interpolation function
#         def curve(tt):
#             tt = torch.as_tensor(tt, device=device, dtype=dtype).reshape(-1)
#             t_eval = torch.linspace(0, 1, 5, device=device, dtype=dtype)
            
#             # Handle edge cases
#             if len(tt) == 0:
#                 return torch.empty(D,0), torch.empty(D,0)
            
#             # Find interpolation indices
#             idx = torch.clamp(
#                 torch.searchsorted(t_eval, tt), 
#                 1, len(t_eval)-1
#             )
            
#             # Linear interpolation
#             t0, t1 = t_eval[idx-1], t_eval[idx]
#             alpha = ((tt - t0)/(t1 - t0 + 1e-10)).unsqueeze(-1)
#             y0 = solution[idx-1]
#             y1 = solution[idx]
#             y_interp = y0 + alpha * (y1 - y0)
            
#             D = x.shape[0]
#             return y_interp[:,:D].T, y_interp[:,D:].T
        
#         return curve, False
        
#     except Exception as e:
#         print(f"ODE solver failed, using fallback: {str(e)}")
#         def fallback_curve(t):
#             return x.reshape(-1,1), v.reshape(-1,1)
#         return fallback_curve, True


# # ALTERNATE OPTION : 1
# from scipy.integrate import odeint  # Often faster for non-stiff problems

# def new_solve_expmap(manifold, x, v, ode_fun, subset_of_weights):
#     D = x.shape[0]
    
#     if isinstance(v, torch.Tensor):
#         v = v.cpu().numpy()
#     if isinstance(x, torch.Tensor):
#         x = x.cpu().numpy()
        
#     init = np.concatenate((x, v), axis=0).flatten()
#     failed = False
    
#     # Using odeint instead of solve_ivp
#     t_eval = np.linspace(0, 1, 10)  # Fewer evaluation points for speed
#     solution = odeint(ode_fun, init, t_eval, tfirst=True, rtol=1e-3, atol=1e-6)
    
#     # Create interpolation function
#     from scipy.interpolate import interp1d
#     interp_func = interp1d(t_eval, solution.T, kind='cubic')
    
#     curve = lambda tt: (interp_func(tt)[:D].reshape(D, -1), 
#                         interp_func(tt)[D:].reshape(D, -1))
    
#     return curve, failed



# # ATTEMPT 2 :
# def new_solve_expmap(manifold, x, v, ode_fun, subset_of_weights):
#     D = x.shape[0]
    
#     # Convert and normalize inputs
#     if isinstance(v, torch.Tensor):
#         v = v.cpu().numpy()
#     if isinstance(x, torch.Tensor):
#         x = x.cpu().numpy()
    
#     # Velocity scaling
#     max_norm = 10.0
#     v_norm = np.linalg.norm(v)
#     if v_norm > max_norm:
#         v = v * (max_norm / v_norm)
    
#     init = np.concatenate((x, v), axis=0).flatten()
#     t_eval = np.linspace(0, 1, 10)
#     failed = False
    
#     try:
#         # First try with BDF method
#         from scipy.integrate import solve_ivp
#         solution = solve_ivp(lambda t, y: ode_fun(t, y) + 1e-4*y,
#                            [0, 1], init,
#                            method='BDF',
#                            t_eval=t_eval,
#                            rtol=1e-4,
#                            atol=1e-6)
#         sol = solution.y.T
#     except:
#         # Fallback to simpler method
#         from scipy.integrate import odeint
#         sol = odeint(lambda y, t: ode_fun(t, y) + 1e-4*y,
#                     init, t_eval,
#                     tfirst=False,
#                     rtol=1e-3,
#                     atol=1e-5,
#                     mxstep=5000)
    
#     # Linear interpolation is more stable than cubic here
#     from scipy.interpolate import interp1d
#     interp_func = interp1d(t_eval, sol.T, kind='linear')
    
#     curve = lambda tt: (interp_func(tt)[:D].reshape(D, -1), 
#                        interp_func(tt)[D:].reshape(D, -1))
    
#     return curve, failed




# # ATTEMPT 3:
# from scipy.integrate import solve_ivp
# import numpy as np
# import torch
# from scipy.interpolate import interp1d

# def new_solve_expmap(manifold, x, v, ode_fun, subset_of_weights):
#     """Optimized exponential map solver with adaptive step control and fallback mechanisms.
    
#     Args:
#         manifold: The Riemannian manifold object
#         x: Initial point (Dx1 tensor or array)
#         v: Initial velocity (Dx1 tensor or array)
#         ode_fun: ODE function to solve
#         subset_of_weights: Which weights to optimize ('all' or 'last_layer')
    
#     Returns:
#         curve: Interpolated solution curve
#         failed: Boolean indicating if solver failed
#     """
#     # Convert inputs to numpy if they're torch tensors
#     if isinstance(v, torch.Tensor):
#         v = v.detach().cpu().numpy()
#     if isinstance(x, torch.Tensor):
#         x = x.detach().cpu().numpy()
    
#     D = x.shape[0]
#     x = x.reshape(-1)
#     v = v.reshape(-1)
    
#     # Adaptive velocity scaling based on manifold curvature
#     max_attempts = 3
#     base_norm = np.linalg.norm(v)
#     success = False
    
#     for attempt in range(max_attempts):
#         current_v = v * (0.5**attempt)  # Exponential backoff
#         init = np.concatenate([x, current_v])
        
#         try:
#             # Try with BDF method first (good for stiff problems)
#             sol = solve_ivp(
#                 lambda t, y: ode_fun(t, y),
#                 [0, 1],
#                 init,
#                 method='BDF',
#                 t_eval=np.linspace(0, 1, 10),
#                 rtol=1e-4,
#                 atol=1e-6,
#                 max_step=0.1
#             )
            
#             if sol.success:
#                 success = True
#                 break
                
#         except Exception as e:
#             continue
    
#     if not success:
#         # Final fallback to simple linear propagation
#         sol = type('', (), {})()  # Create empty object
#         sol.t = np.linspace(0, 1, 2)
#         sol.y = np.column_stack([init, init])
#         return lambda t: (x.reshape(-1, 1), v.reshape(-1, 1)), True
    
#     # Create efficient interpolation
#     interp_func = interp1d(sol.t, sol.y, kind='linear', axis=1)
    
#     curve = lambda t: (
#         interp_func(t)[:D].reshape(D, -1),
#         interp_func(t)[D:].reshape(D, -1)
#     )
    
#     return curve, False