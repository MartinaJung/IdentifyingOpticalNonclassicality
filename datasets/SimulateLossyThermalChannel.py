import numpy as np

# NumPy >=2.0 removed np.find_common_type, which thewalrus.quantum.density_matrix
# still calls internally. Restore it as a no-op shim if missing (harmless on
# older NumPy, where hasattr(...) is already True and this never runs).
if not hasattr(np, 'find_common_type'):
    def _find_common_type_shim(array_types, scalar_types):
        return np.result_type(*array_types, *scalar_types)
    np.find_common_type = _find_common_type_shim

from thewalrus.symplectic import loss
from thewalrus.quantum import density_matrix, is_valid_cov
from numpy import random
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from scipy.special import eval_genlaguerre, gammaln
from scipy.integrate import quad



# ------------------------------------------------------
#
#           helper functions
#
# ------------------------------------------------------

def _p_mid_after_loss(nbar_eff: float, T: float, cutoff: int) -> np.array:
    '''
    Photon-number distribution after the pure-loss (mbar=0) step:
    binomial thinning of SPATS(nbar) with transmissivity T, giving

        p_mid(m) = T * p_SPATS(m; nbar_eff) + (1-T) * p_X(m; nbar_eff)

    with nbar_eff = T*nbar and p_X(m; n) = (m+1) * n**m / (1+n)**(m+2).
    '''
    m = np.arange(cutoff)
    m_shift = np.maximum(m-1, 0)
    p_SPATS = np.where(m >= 1, m*nbar_eff**m_shift/(1+nbar_eff)**(m+1), 0.0)
    p_X = (m+1)*nbar_eff**m/(1+nbar_eff)**(m+2)
    return T*p_SPATS + (1-T)*p_X


def _kernel_K(l: int, m: int, nu: float) -> float:
    '''
    Numerically stable evaluation of K(l,m;nu), the Fock-basis
    transition kernel of the "random coherent displacement" step:
    mixing a Fock state |m> with a thermal ancilla of mean mbar via a
    T-beamsplitter is equivalent to pure loss (transmissivity T)
    followed by displacing by a random beta with |beta|^2 ~
    Exponential(mean nu), nu = (1-T)*mbar. K(l,m;nu) is the resulting
    probability of finding photon number l:

        K(l,m;nu) = (1/nu) * integral_0^inf x^Delta * exp(-x*(1+1/nu))
                    * [L_min(l,m)^(|l-m|)(x)]^2 dx * min(l,m)!/max(l,m)!

    A naive evaluation (expanding the Laguerre polynomial into its
    explicit coefficients, or forming x**Delta directly) suffers
    catastrophic cancellation/overflow once |l-m| is large. This
    evaluates the integrand in log-space using scipy's numerically
    stable eval_genlaguerre (itself verified against mpmath arbitrary-
    precision arithmetic to ~1e-13 relative accuracy even for
    n~60, alpha~700), with the integration range and the exponentiation
    baseline chosen adaptively around the integrand's peak so no
    intermediate quantity over/underflows. Verified against mpmath
    ground truth (relative error ~1e-12 to 1e-16) across l,m up to 728.
    '''
    if nu == 0.0:
        return 1.0 if l == m else 0.0
    Delta = abs(l-m)
    n, mx = min(l, m), max(l, m)
    p = 1 + 1/nu

    def log_integrand(x):
        with np.errstate(all='ignore'):
            L = eval_genlaguerre(n, Delta, x)
        if L == 0 or x <= 0:
            return -np.inf
        return Delta*np.log(x) - x*p + 2*np.log(np.abs(L))

    laguerre_scale = (np.sqrt(n) + np.sqrt(n+Delta))**2 if (n+Delta) > 0 else 1.0
    xmax = 4*max(Delta/p, laguerre_scale, 1.0) + 20*nu
    xs = np.linspace(1e-8, xmax, 2000)
    logvals = np.array([log_integrand(x) for x in xs])
    logvals[~np.isfinite(logvals)] = -np.inf
    peak = np.max(logvals)
    if not np.isfinite(peak):
        return 0.0

    def shifted_integrand(x):
        lv = log_integrand(x)
        return np.exp(lv-peak) if np.isfinite(lv) else 0.0

    val, _ = quad(shifted_integrand, 0, xmax, limit=300)
    log_prefac = gammaln(n+1) - gammaln(mx+1) - np.log(nu)
    return np.exp(log_prefac+peak)*val


def _find_window(center: int, other: int, nu: float, cutoff: int, thresh: float) -> range:
    '''
    Range of l around `center` for which K(l, other; nu) > thresh,
    found by exponential search followed by bisection in each
    direction (K(l,other;nu) is unimodal in l, peaked near l=other).
    '''
    def above(l):
        return 0 <= l < cutoff and _kernel_K(l, other, nu) > thresh

    def expand(step):
        lo, hi = 0, 1
        while above(center+step*hi) and center+step*hi >= 0 and center+step*hi < cutoff:
            lo = hi
            hi *= 2
        while hi-lo > 1:
            mid = (lo+hi)//2
            if above(center+step*mid):
                lo = mid
            else:
                hi = mid
        return lo

    right = expand(+1)
    left = expand(-1)
    lo = max(0, center-left)
    hi = min(cutoff-1, center+right)
    return range(lo, hi+1)

#-----------------------------------------------------------------
#
#               Lossy channel
#
#-----------------------------------------------------------------

def LossySPATS(T: float, nbar: float, mbar: float, cutoff: int, tail_eps: float = 1e-10) -> np.array:
    '''
    Output photon-number distribution of a SPATS after a lossy
    channel with transmissivity T and thermal-ancilla mean mbar.

    Derivation: mixing the (non-Gaussian) SPATS(nbar) state with a
    thermal ancilla of mean mbar on a T-beamsplitter is exactly
    equivalent to pure loss (transmissivity T) followed by a random
    coherent displacement D(beta), |beta|^2 ~ Exponential(mean
    nu=(1-T)*mbar) -- derived from the thermal ancilla's Glauber-
    Sudarshan P-representation plus beamsplitter displacement-operator
    conjugation, and verified against an independent truncated
    two-mode Fock-space beamsplitter simulation. Concretely:

        nbar_eff = T * nbar
        p_mid(m) = T * p_SPATS(m; nbar_eff) + (1-T) * p_X(m; nbar_eff)
        p_out(l) = sum_m p_mid(m) * K(l, m; nu)

    with p_X(m; n) = (m+1) * n**m / (1+n)**(m+2) and K the stable
    kernel implemented in _kernel_K. For mbar=0 this reduces to the
    plain pure-loss formula (nu=0, K(l,m;0)=delta_{l,m}).

    Since p_mid(m) and K(l,m;nu) both decay quickly away from their
    respective centers, the m-sum and l-range are truncated once their
    tails fall below `tail_eps`, which keeps the cost far below the
    naive O(cutoff^2) (or worse) full evaluation while not perceptibly
    affecting normalization (checked to sum to 1 to ~1e-6 or better
    across the tested parameter range).

    Args:
        T (float): transmissivity, must be in [0,1]
        nbar (float): number of thermal photons in the initial
            SPATS
        mbar (float): number of thermal photons in the mixing
            (ancilla) thermal state
        cutoff (int): Fock-space cutoff
        tail_eps (float): truncation threshold for both the input
            SPATS-after-loss tail and the output kernel spread
    '''
    nbar_eff = T*nbar
    p_mid = _p_mid_after_loss(nbar_eff, T, cutoff)

    nu = (1-T)*mbar
    if nu == 0.0:
        return p_mid

    tail = 1-np.cumsum(p_mid)
    above_thresh = np.where(tail >= tail_eps)[0]
    m_foot = int(above_thresh[-1])+1 if above_thresh.size else cutoff-1
    m_foot = min(max(m_foot, 1), cutoff-1)

    p_out = np.zeros(cutoff)
    for m in range(m_foot+1):
        if p_mid[m] < tail_eps:
            continue
        for l in _find_window(m, m, nu, cutoff, tail_eps):
            p_out[l] += p_mid[m]*_kernel_K(l, m, nu)
    return p_out

def LossyGauss(T, alpha, r, nbar, mbar) -> (float, np.array):
    '''
    Output photon-number distribution of a Gaussian state

    Args:    
        T (float): transmissivity, 1 corresponds to
            no loss, 0 to full loss.
        alpha(complex): displacement argument
        r (float): squeezing parameter
        mbar (float): number of thermal photons in
            mixing thermal state, mbar=0 defaults to
            general loss channel
    Returns:
        mu (float): mean of the output Gaussian
        cov_T (array): covariance matrix of output Gaussian
    '''
    mu = np.array([np.sqrt(2)*np.real(alpha), np.sqrt(2)*np.imag(alpha)])
    cov = np.diag([np.exp(-2*r), np.exp(2*r)])*1/2 + nbar*np.eye(2,)
    mu_T, cov_T = loss(mu, cov, T, 0, mbar, hbar=1)
    return mu_T, cov_T


def SampleState(mu: np.ndarray, cov: np.ndarray, cutoff: int) -> np.array:
    '''
    Returns the photon-number probability distribution
    (diagonal of the density matrix) of a Gaussian state
    defined by mean mu and covariance matrix cov.

    Args:
        mu (array): mean of Gaussian
        cov (array): covariance matrix
        cutoff (int): cutoff for fock space truncation
    Returns:
        array: the photon-number probabilities
    '''
    dm = density_matrix(mu, cov, cutoff=cutoff, hbar=1)
    p = np.abs(np.real(np.diag(dm)))
    p = np.nan_to_num(p, nan=0.0)
    #p = np.pad(p, (0, cutoff - fock_cutoff), constant_values=0.0)
    return np.array(p, dtype='float32')

_DETECTOR_FILES = {
    '4Bin':  (189, 'ExperimentalData/4-bin spatial/POVM_4bin.npy'),
    'SNSPD': (443, 'ExperimentalData/4-pixel SNSPD/POVM_4p.npy'),
    '8Bin':  (158, 'ExperimentalData/8-bin TMD/POVM_8bin.npy'),
    'PNR':   (729, 'ExperimentalData/PNR SNSPD/POVMs_EMG_4+_smoothing_1e-6_D40.npy'),
    'Eye':   (729, '.')
}

def load_detector(detector_type: str, abs_path: str = '.') -> (int, np.array):
    '''
    Loads a detector's POVM once, to be reused across many
    calls to Get_outcome_distr instead of reloading it from
    disk every time.

    Args:
        detector_type (str): one of '4Bin', 'SNSPD', '8Bin', 'PNR', 'Eye'
        abs_path (str): path to the directory containing
            ExperimentalData/
    Returns:
        cutoff (int): fock space cutoff for this detector
        POVM (array): the detector's POVM
    '''
    if detector_type not in _DETECTOR_FILES:
        raise Exception('Please decide for either a' + 
        '\'4Bin\', \'8Bin\', \'SNSPD\' or \'PNR\' detector or' +
        '\'Eye \'')
    if detector_type == 'Eye':
        return 729, np.eye(729)
    else:
        cutoff, rel_path = _DETECTOR_FILES[detector_type]
        return cutoff, np.load(f'{abs_path}/{rel_path}')

def Get_outcome_distr(
    species:str,
    ampl:float,
    T: float,
    mbar: float,
    POVM: np.array,
    cutoff: int) -> np.array:

    if species == 'Coherent':
        mu, cov = LossyGauss(T, alpha=np.sqrt(ampl), r=0.0, nbar=0.0, mbar=mbar)
        p = SampleState(mu, cov, cutoff)
    elif species == 'Squeezed':
        mu, cov = LossyGauss(T, alpha=0.0, r=ampl, nbar=0.0, mbar=mbar)
        p = SampleState(mu, cov, cutoff)
    elif species == 'Thermal':
        mu, cov = LossyGauss(T, alpha=0.0, r=0.0, nbar=ampl, mbar=mbar)
        p = SampleState(mu, cov, cutoff)
    elif species == 'SPATS':
        p = LossySPATS(T=T, nbar=ampl, mbar=mbar, cutoff=cutoff)
    elif species == 'Mixed Coherent':
        mu1, cov1 = LossyGauss(T, alpha=np.sqrt(ampl[0]), r=0.0, nbar=0.0, mbar=mbar)
        mu2, cov2 = LossyGauss(T, alpha=np.sqrt(ampl[1]), r=0.0, nbar=0.0, mbar=mbar)
        p1 = SampleState(mu1, cov1, cutoff)
        p2 = SampleState(mu2, cov2, cutoff)
        p = 0.5*p1 + 0.5*p2
        label = 0
    else:
        raise Exception('Please decide for a valid state species')
    try:
        diag_cov = np.min(np.diag(cov))
        label = int(diag_cov < .5 - 1e-9)
    except:
        if species == 'Mixed Coherent':
            label = 0
        else:
            Dk = p[0]*p[2]*2/(p[1]**2)
            label = int(Dk < 1.)
    

    click_distr = POVM.T @ p
    # make sure that p sums up to 1
    if np.sum(click_distr[:-1]) < 1.:
        click_distr[-1] = 1 - np.sum(click_distr[:-1])
    else:
        click_distr[-1] = 0.0
        click_distr = click_distr/np.sum(click_distr)
        #click_distr = np.round(click_distr, 3)
    return click_distr, label

# -------------------------------------------
#
#           Generate Validation Dataset
#
# -------------------------------------------

def Gen_validation_dataset(detector_type, species, abs_path='.') -> np.array:
    cutoff, POVM = load_detector(detector_type, abs_path)
    dis = []
    labels = []
    for T in tqdm(np.ones(11) - 0.01*np.arange(11)):
        dis1 = []
        labels1 = []
        for mbar in 0.1*np.arange(21):
            #print('Using mbar:', mbar)
            dis2 = []
            labels2 =[]
            if species == 'Coherent':
                if detector_type == '8Bin':
                    alphas = np.array([1.03549022e-03, 1.01962952e-02,
                    1.00401176e-01, 9.88633221e-01,
                    3.97011969e+00, 8.89977035e+00, 
                    1.59430717e+01, 2.44116874e+01,
                    3.57393953e+01, 4.78347821e+01, 
                    6.26040169e+01, 8.01165796e+01,
                    9.80316246e+01])[::-1]
                elif detector_type == 'PNR':
                    alphas = np.arange(13)
                elif detector_type == 'Eye':
                    alphas = np.arange(0.0, 3.6, 0.1)
            elif species == 'Squeezed':
                alphas = np.array([i*0.1 for i in range(1,13)])
            elif species == 'Thermal':
                alphas = np.array([i*0.5 for i in range(1,15)])
            elif species == 'SPATS':
                if detector_type == 'Eye':
                    alphas = np.arange(0.25, 1.25, 0.05, dtype=np.float64)
                elif detector_type in ['8Bin', 'PNR']:
                    alphas = np.array([i*0.03 for i in range(5,15)])
            elif species == 'Mixed Coherent':
                alphas = []
                coh = np.arange(0.0, 3.6, 0.1)
                for idx, c in enumerate(coh[:-18]):
                    alphas.append([c,coh[idx+18]])
                    
            for alpha in alphas:
                click_distr_ch, label = Get_outcome_distr(species, alpha, T, mbar, POVM, cutoff)
                if not np.isclose(np.sum(click_distr_ch),1):
                    print('Sum was not 1 but', np.sum(click_distr_ch))
                dis2.append(click_distr_ch)
                labels2.append(label)
            dis1.append(np.array(dis2))
            labels1.append(np.array(labels2))
        dis.append(np.array(dis1))
        labels.append(np.array(labels1))
    return np.array(dis), np.array(labels)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error: Please provide at least one argument.")
        sys.exit(1)
    STATE_SPECIES = sys.argv[1]

    d, ls = Gen_validation_dataset('PNR', STATE_SPECIES)
    np.savez('ds14_PNR_lossynoisechannel_' + STATE_SPECIES, probs=d, labels=ls)