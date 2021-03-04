# Functions to generate random models
import numpy as np
import itertools
import random
from copy import deepcopy
import warnings

# Internal
from savar.functions import create_random_mode, create_non_stationarity
from savar.spatial_models import savarModel


def check_stationarity(links, linear=True):
    """Returns stationarity according to a unit root test
    Assuming a Gaussian Vector autoregressive process
    Three conditions are necessary for stationarity of the VAR(p) model:
    - Absence of mean shifts;
    - The noise vectors are identically distributed;
    - Stability condition on Phi(t-1) coupling matrix (stabmat) of VAR(1)-version  of VAR(p).
    """

    links = deepcopy(links)
    N = len(links)
    # Check parameters
    max_lag = 0

    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]

            max_lag = max(max_lag, abs(lag))

    graph = np.zeros((N, N, max_lag))

    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            coeff = link_props[1]
            if abs(lag) > 0:
                graph[j, var, abs(lag) - 1] = coeff
        if not linear:
            couplings = []
            coupling = link_props[2]
            couplings.append(coupling)

    stabmat = np.zeros((N * max_lag, N * max_lag))
    index = 0

    for i in range(0, N * max_lag, N):
        stabmat[:N, i:i + N] = graph[:, :, index]
        if index < max_lag - 1:
            stabmat[i + N:i + 2 * N, i:i + N] = np.identity(N)
        index += 1

    eig = np.linalg.eig(stabmat)[0]
    # print "----> maxeig = ", np.abs(eig).max()
    if np.all(np.abs(eig) < 1.):
        stationary = True
    else:
        stationary = False

    return stationary, np.abs(eig).max()


def _get_random_link_strenght(mean: float, std: float, strength_threshold: float = 0.2, random_sign: float = 0.5):
    """
    Returns a random number from a gaussian distribution with a given mean and std, with a minimum link strength
    and a chance that the sign is inverted.
    :param mean: mean
    :param std: standard deviation
    :param random_sign: if not None change the sign randomly according to probability.
    :param strength_threshold: minimum link strength
    :return: the link strength
    """
    while True:
        strength = np.random.normal(mean, std)
        if strength < strength_threshold:
            continue

        if random_sign is not None:
            if np.abs(np.random.rand()) < random_sign:
                return -strength
            else:
                return strength


def generate_random_coeff(n_variables, n_links, auto_coeffs_mean: float = 0.3, auto_coffs_std: float = 0.2,
                          auto_strength_threshold: float = 0.2, auto_random_sign: float = 0.,
                          coupling_mean=0.3, coupling_std=0.2, coupling_strength_threshold: float = 0.2,
                          coupling_random_sign: float = 0.2, tau_max: int = 3,
                          n_trial=1000, n_non_linear: int = None, model_seed=None):
    """
    generates a random linear link_coeff.
    :param n_variables:
    :param n_links:
    :param auto_coeffs_mean:
    :param auto_coffs_std:
    :param auto_strength_threshold:
    :param auto_random_sign:
    :param coupling_mean:
    :param coupling_std:
    :param coupling_strength_threshold:
    :param coupling_random_sign:
    :param tau_max:
    :param n_trial:
    :param model_seed:
    :return:
    """
    np.random.seed(model_seed)

    for _ in range(n_trial):

        # Create empty link list
        links = {N: [] for N in range(n_variables)}

        # Autocorrelation
        for i in range(n_variables):
            links[i].append(((int(i), -1), _get_random_link_strenght(mean=auto_coeffs_mean, std=auto_coffs_std,
                                                                     strength_threshold=auto_strength_threshold,
                                                                     random_sign=auto_random_sign)))

        # Coupling_coffs
        # Generate couplings
        all_possible = np.array(list(itertools.permutations(range(n_variables), 2)))

        # TODO if we want more links than var**2
        all_possible_tau = [(i, j, tau) for i, j in all_possible for tau in range(1, tau_max + 1)]

        # Choose n_links
        chosen_links = all_possible[np.random.permutation(len(all_possible))[:n_links]]

        # TODO Change from random selection
        for (i, j) in chosen_links:
            # Choose lag with decaying probability
            tau_list = [x for x in range(1, tau_max + 1)]
            weights_tau = list(np.linspace(0.9, 0.2, tau_max))
            #tau = int(np.random.randint(1, tau_max + 1))
            tau = int(random.choices(tau_list, weights=weights_tau)[0])

            c = _get_random_link_strenght(mean=coupling_mean, std=coupling_std,
                                          strength_threshold=coupling_strength_threshold,
                                          random_sign=coupling_random_sign)
            links[j].append(((i, -tau), c))

        # Stationarity check assuming model with linear dependencies at least for large x
        if check_stationarity(links)[0]:
            if n_non_linear is not None:
                return _linear_coeffs_to_nonlinear(links, n_non_linear)
            else:
                return links

    # If not possible to find a coef
    raise RecursionError("Impossible to find stable coeffs, rise the number of trials")


mode_configuration = {
    "size": (30, 30),
    "mu": (0, 0),
    "var": (.5, .5),
    "position": (3, 3, 3, 3),
    "plot": False,
    "Sigma": None,
    "random": True,
}

def _linear_coeffs_to_nonlinear(links_coeffs, nonlinear_links, func = "f2"):
    """
    Randomly adds non-linear coefficients to
    :param links_coeffs:
    :param n_non_linear_links:
    :return:
    """
    n_links = 0

    # First we get the number of links
    for var_j, links in links_coeffs.items():
        for (_, _), coeff in links:
            n_links += 1

    if n_links < nonlinear_links:
        warnings.warn("selected non-linear links bigger than maximum links. Setting nonlinear_links to maximum",
                      Warning)
        nonlinear_links = n_links

    # Randomly choose which links are non-linear.
    sel_links = list(np.random.choice(n_links, size=nonlinear_links, replace=False))

    links_coeffs_nonlinear = {}

    count = 0
    for var_j, links in links_coeffs.items():
        links_coeffs_nonlinear[var_j] = []
        for (var_i, lag_var), coeff, in links:
            if count in sel_links:
                links_coeffs_nonlinear[var_j].append(((var_i, lag_var), coeff, func))
            else:
                links_coeffs_nonlinear[var_j].append(((var_i, lag_var), coeff, "linear"))
            count += 1

    return links_coeffs_nonlinear


def generate_weights(n_variables: int, resolution: tuple, dipole: int = None,
                     gaussian_shape: bool = True, random_mode: bool = True,
                     norm_weight=True):
    """
    Creates the Weight matrix.Spatial weight multiplies the modes, can vary slightly
    :param n_variables:
    :param resolution: resolution of the whole system (square)
    :param spatial_weight_mean:
    :param spatial_weight_std:
    :param gaussian_shape:
    :param random_mode:
    :return:
    """

    def find_mode_positions(res: tuple = (20, 30), n_var: int = 5):

        # Loop that checks
        ny, nx = res
        size = 1
        # Loop to find the maximum size that allow the number of our modes
        while True:
            n_horiz = nx // size
            n_verti = ny // size
            total_slots = n_horiz * n_verti
            if total_slots > n_var: # If there is free space rise size
                size += 1
                continue
            if total_slots == n_var:  # Match exactly
                break
            if total_slots < n_var:  # Too large, go back and output the result
                size -= 1
                n_horiz = nx // size
                n_verti = ny // size
                break

        # We get the position of each mode
        positions = []
        for i in range(n_verti):
            for j in range(n_horiz):
                y_1 = i*size
                y_2 = i*size + size
                x_1 = j * size
                x_2 = j*size + size
                positions.append((y_1, y_2, x_1, x_2))

        return  size, positions[:n_var]

    def shaped_mode(size: int, dipole: bool = False, gaussian_shape: bool = True) -> np.ndarray:
        """
        returns either a mode with gaussian shape or squared.
        """
        frac = size//2
        rest = size%2
        if dipole:
            if gaussian_shape:
                if random_mode:
                    mode_1 = create_random_mode((frac, size))
                    mode_2 = create_random_mode((frac, size))
                    mode = np.zeros((size, size))
                    mode[:, :frac] = mode_1
                    mode[:, frac+rest:] = -mode_2
                    return mode
                else:
                    mode_1 = create_random_mode((frac, size), random=False)
                    mode_2 = create_random_mode((frac, size), random=False)
                    mode = np.zeros((size, size))
                    mode[:, :frac] = mode_1
                    mode[:, frac+rest:] = -mode_2
                    return mode
            else:
                mode = np.zeros((size, size))
                mode[:, :frac] = 1
                mode[:, frac+rest:] = -1
                return mode
        # No dipole
        else:
            if gaussian_shape:
                if random_mode:
                    return create_random_mode((size, size))
                else:
                    return create_random_mode((size, size), random=False)
            else:
                return np.ones((size, size))

    # First estimate the size of of each mode and its location
    ny, nx = resolution
    size, positions = find_mode_positions(res=resolution, n_var=n_variables)
    weights = np.zeros((n_variables, ny, nx))

    if dipole is not None:  # If we include dipoles
        dipol_list = random.sample(list(range(n_variables)), dipole)
        for i in range(n_variables):
            y_1, y_2, x_1, x_2 = positions[i]
            if i in dipol_list:  # If this variable is a dipole
                weights[i, y_1:y_2, x_1: x_2] = shaped_mode(size=size, gaussian_shape=gaussian_shape,
                                                    dipole=True)
            else:
                weights[i, y_1:y_2, x_1: x_2] = shaped_mode(size=size, gaussian_shape=gaussian_shape,
                                                    dipole=False)
            # Add constraint |W|_1 = 1
            if norm_weight:
                weights[i, y_1:y_2, x_1: x_2] /= weights[i, y_1:y_2, x_1: x_2].sum()
    else:
        # Add the modes
        for i in range(n_variables):
            y_1, y_2, x_1, x_2 = positions[i]
            weights[i, y_1:y_2, x_1:x_2] = shaped_mode(size=size, gaussian_shape=gaussian_shape,
                                                dipole=False)
            if norm_weight:
                weights[i, y_1:y_2, x_1:x_2] /= weights[i, y_1:y_2, x_1:x_2].sum()
    return size, weights


def generate_savar_model(
        # Savar
        n_variables, spatial_covariance=1., time_length=500, variance_noise=1., noise_weights=None, links_coeffs = None,
        covariance_noise_method="weights_transposed", transient=200, forcing=None, season=None,
        season_weight=None, linearity="linear", nonl_coeff=0.5, forcing_field: np.ndarray=None, w_t: np.ndarray = False,
        verbose=True, ornstein_sigma=None, n_var_ornstein=None,
        # Weights
        resolution=(10, 10), spatial_weight_mean=0.5, spatial_weight_std=0.2, gaussian_shape: bool = True,
        random_mode=True, dipole: int = None, mask_variables: int = None,  norm_weight: bool = True,
        # Links
        n_links=3, auto_coeffs_mean: float = 0.3, auto_coffs_std: float = 0.2,
        auto_strength_threshold: float = 0.2, auto_random_sign: float = 0.,
        coupling_mean=0.3, coupling_std=0.2, coupling_strength_threshold: float = 0.2,
        coupling_random_sign: float = 0.2, tau_max: int = 3,
        n_trial=1000, n_non_linear: int = None, model_seed=None):

    links_dict = {
        "n_variables": deepcopy(n_variables),
        "n_links": n_links,
        "auto_coeffs_mean": auto_coeffs_mean,
        "auto_coffs_std": auto_coffs_std,
        "auto_strength_threshold": auto_strength_threshold,
        "auto_random_sign": auto_random_sign,
        "coupling_mean": coupling_mean,
        "coupling_std": coupling_std,
        "coupling_strength_threshold": coupling_strength_threshold,
        "coupling_random_sign": coupling_random_sign,
        "n_non_linear": n_non_linear,
        "tau_max": tau_max,
        "n_trial": n_trial,
        "model_seed": model_seed,
    }

    if mask_variables is not None:
        links_dict["n_variables"] = mask_variables

    weights_dict = {
        "n_variables": n_variables,
        "resolution": resolution,  # (80, 120) allows to archive 100 variables
        "dipole": dipole,
        "gaussian_shape": gaussian_shape,
        "random_mode": random_mode,
        "norm_weight": norm_weight
    }

    # Links
    if links_coeffs is None:
        links_coeffs = generate_random_coeff(**links_dict)

    # weights
    size, modes_weights = generate_weights(**weights_dict)

    if mask_variables is not None:
        modes_weights = modes_weights[:mask_variables, ...]
        n_variables = mask_variables

    if noise_weights is None:
        noise_weights = deepcopy(modes_weights)

    if n_non_linear is not None:
        linearity = "nonlinear"

    if n_var_ornstein is not None:
        ornstein_proces = create_non_stationarity(N_var=n_var_ornstein,
                                                  t_sample=time_length,
                                                  sigma=ornstein_sigma,
                                                  tau=0.3)  # *variance_noise
        weights_inv = np.linalg.pinv(deepcopy(modes_weights).reshape(n_variables, -1))
        forcing_field = (weights_inv @ ornstein_proces.transpose()).transpose()  # T x L (as demanded by code)

    savar_dict = {
        "links_coeffs": links_coeffs,
        "modes_weights": modes_weights,
        "ny": resolution[0],  # Latitude
        "nx": resolution[1],
        "T": time_length,
        "noise_weights": noise_weights,
        "variance_noise": variance_noise,
        "spatial_covariance": spatial_covariance,
        "covariance_noise_method": covariance_noise_method,
        "transient": transient,
        "forcing": forcing,
        "n_variables": n_variables,
        "season": season,
        "season_weight": season_weight,
        "linearity": linearity,
        "nonl_coeff": nonl_coeff,
        "forcing_field": forcing_field,
        "w_t": w_t,
        "verbose": verbose,
    }

    return savarModel(**savar_dict)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from eval_tools import DmMethod, Evaluation
    resuloution = (5, 5)
    savar = generate_savar_model(2, resolution=resuloution, spatial_covariance=0.5, n_var_ornstein=None,
                                 norm_weight=True, variance_noise=10, ornstein_sigma=0.5)
    savar.create_savar_data()
    print(savar.data_field.shape)

    if True:
        for i in range(2):
            plt.imshow(savar.noise_weights[i, ...])
            plt.colorbar()
            plt.show()
    plt.imshow(savar.cov)
    plt.colorbar()
    plt.show()

    dm_method = DmMethod(savar, verbose=True)
    dm_method.perform_dm()
    dm_method.get_pcmci_results()
    dm_method.get_phi_and_predict()

    eval = Evaluation(dm_method, grid_threshold_per=95)
    eval.obtain_score_metrics(perform_grid=True)
    # eval.obtain_score_metrics()
    # print(eval.metrics)


    #for i in range(2):
    #    plt.imshow(dm_method.weights["varimax"][i, ...].reshape(resuloution))
    #    plt.colorbar()
    #    plt.show()

    #plt.plot(savar.network_data)
    #plt.show()

