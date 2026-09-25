import pytest, os, sys
import numpy as np
from thewalrus.symplectic import loss
from thewalrus.quantum import density_matrix_element, is_valid_cov
from numpy import random
from scipy import stats
from scipy import sparse
from scipy.special import factorial, binom

SCRIPTPATH = os.path.abspath(__file__)
REPODIR = os.path.split(os.path.split(SCRIPTPATH)[0])[0]
sys.path.append(REPODIR)

from datasets import SimulateLossyThermalChannel as slt
from datasets import GenDatasets_SingleModeExperimentalPOVM as Gen

def _g2(p, cutoff=158):
    '''g2 = <n(n-1)>/<n>^2, i.e. sub/super-Poissonian statistic.
    Returns nan (not a division warning) for a vacuum state (<n>=0),
    where g2 is genuinely undefined.'''
    n = np.arange(cutoff)
    mean = n @ p
    if np.isclose(mean, 0.0):
        return np.nan
    return ((n**2 - n) @ p) / mean**2

@pytest.mark.parametrize("alpha,r,nbar,mbar",[
    (1.0, 0.5, 1.0, 0.0),
    (0.0, 0.0, 0.0, 0.0)])
def test_GS_tau_1(alpha, r, nbar, mbar):
    # Squeezed
    mu_sq, cov_sq = slt.LossyGauss(T=1.0, alpha=0.0, r=r, nbar=0.0, mbar=mbar)
    p_lossy_sq = slt.SampleState(mu_sq, cov_sq, cutoff=158)
    p_direct_sq = Gen.ProbDistr1Mode('8Bin', 'Squeezed', ampl=r)
    # Coherent
    mu_ch, cov_ch = slt.LossyGauss(T=1.0, alpha=alpha, r=0.0, nbar=0.0, mbar=mbar)
    p_lossy_ch = slt.SampleState(mu_ch, cov_ch, cutoff=158)
    p_direct_ch = Gen.ProbDistr1Mode('8Bin', 'Coherent', ampl=alpha)
    # Thermal
    mu_th, cov_th = slt.LossyGauss(T=1.0, alpha=0.0, r=0.0, nbar=nbar, mbar=mbar)
    p_lossy_th = slt.SampleState(mu_th, cov_th, cutoff=158)
    p_direct_th = Gen.ProbDistr1Mode('8Bin', 'Thermal', ampl=nbar)

    for cov in [cov_sq, cov_ch, cov_th]:
        assert is_valid_cov(cov, hbar=1)
        print('Valid cov matrix:',is_valid_cov(cov, hbar=1) )

    assert np.all(np.isclose(p_lossy_ch, p_direct_ch))
    assert np.all(np.isclose(p_lossy_sq, p_direct_sq))
    assert np.all(np.isclose(p_lossy_th, p_direct_th))

@pytest.mark.parametrize("alpha,nbar,mbar",[
    (1.0, 1.0, 0.0),
    (0.0, 0.0, 0.0)])    
def test_SPATS_tau_1(alpha, nbar, mbar):
    # Check SPATS
    p_direct = Gen.ProbDistr1Mode('8Bin', 'PATS', ampl=alpha)
    assert np.isclose(np.sum(p_direct),1)
    p_lossy = slt.LossySPATS(T=1, nbar=nbar, mbar=mbar, cutoff=158)
    assert np.all(np.isclose(p_lossy,p_direct))
    #return (np.isclose(np.sum(p_direct),1), np.all(np.isclose(p_lossy,p_direct)) )

@pytest.mark.parametrize("nbar",[
    0.0, 1.0,0.5])
def test_tau_0(nbar):
    p_direct = Gen.ProbDistr1Mode('8Bin', 'Thermal', ampl=nbar)
    mu_ch, cov_ch = slt.LossyGauss(T=0.0, alpha=2.0, r=0.0, nbar=1.0, mbar=nbar)
    p_lossy_ch = slt.SampleState(mu_ch, cov_ch, cutoff=158)
    assert np.all(np.isclose(p_direct, p_lossy_ch))
    #return np.all(np.isclose(p_direct, p_lossy_ch))

@pytest.mark.parametrize("T, alpha,r,nbar",[
    (0.5, 1.5, 0.2, 1.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.5, 0.0, 0.7, 0.0),
    (1.0, 1.0, 0.2, 1.0)
    ])
def test_tau_less_1_Gaussian_States(T,alpha, r, nbar):
    # Squeezed
    mu_sq, cov_sq = slt.LossyGauss(T=T, alpha=0.0, r=r, nbar=0.0, mbar=0.0)
    p_lossy_sq = slt.SampleState(mu_sq, cov_sq, cutoff=158)
    p_direct_sq = Gen.ProbDistr1Mode('8Bin', 'Squeezed', ampl=r)
    assert np.isclose(_g2(p_lossy_sq), _g2(p_direct_sq), equal_nan=True)

    # Coherent
    mu, cov = slt.LossyGauss(T=T, alpha=alpha, r=0.0, nbar=0.0, mbar=0.0)
    p_lossy = slt.SampleState(mu, cov, cutoff=158)
    p_direct = Gen.ProbDistr1Mode('8Bin', 'Coherent', ampl=alpha)
    assert np.isclose(_g2(p_lossy), _g2(p_direct), equal_nan=True)

    # Thermal
    mu, cov = slt.LossyGauss(T=T, alpha=0.0, r=0.0, nbar=nbar, mbar=0.0)
    p_lossy = slt.SampleState(mu, cov, cutoff=158)    
    p_direct = Gen.ProbDistr1Mode('8Bin', 'Thermal', ampl=nbar)
    assert np.isclose(_g2(p_lossy), _g2(p_direct), equal_nan=True)

@pytest.mark.parametrize("T, nbar",[
    (0.5, 1.0),
    (0.5, 1.5),
    (1.0, 1.0)
    ])
def test_tau_less_1_SPATS(T,nbar):
    # SPATS
    p_direct = Gen.ProbDistr1Mode('8Bin', 'PATS', ampl=nbar)
    p_lossy = slt.LossySPATS(T=T, nbar=nbar, mbar=0.0, cutoff=158)    
    if nbar == 0.0:
        pass
    else:    
        assert _g2(p_direct) >= 1.0
        assert _g2(p_lossy) >= 1.0

@pytest.mark.parametrize("T, nbar, mbar",[
    (0.5, 1.2, 1.0),
    (0.25, 1.2, 2.0),
    (0.75, 1.2, 0.5),
    (1.0, 1.2, 2.0),   # T=1 -> mbar must have no effect at all
    ])
def test_LossySPATS_mbar_gt_0(T, nbar, mbar):
    p = slt.LossySPATS(T=T, nbar=nbar, mbar=mbar, cutoff=158)

    # basic distribution sanity
    assert np.isclose(np.sum(p), 1.0, atol=1e-6)
    assert np.all(p >= -1e-12)

    # exact analytic check on the mean photon number:
    # SPATS(nbar) has mean 2*nbar+1; pure loss T scales this by T;
    # the random-displacement step adds nu=(1-T)*mbar in expectation
    # (cross terms vanish since the displacement is phase-averaged to
    # zero mean) -- this holds exactly, not just approximately.
    mean_n = np.arange(158) @ p
    expected_mean = T*(2*nbar+1) + (1-T)*mbar
    assert np.isclose(mean_n, expected_mean, rtol=1e-3)

    # at T=1 the ancilla never couples in (nu=(1-T)*mbar=0 regardless
    # of mbar), so the result must be identical to the mbar=0 case
    if T == 1.0:
        p_mbar0 = slt.LossySPATS(T=T, nbar=nbar, mbar=0.0, cutoff=158)
        assert np.allclose(p, p_mbar0)

# ------------------------------------------------------
#
#           _p_mid_after_loss
#
# ------------------------------------------------------

def test_p_mid_after_loss_T1_is_pure_SPATS():
    '''At T=1 the (1-T)*p_X cross term vanishes, so the result must
    reduce to the plain SPATS(nbar_eff) distribution -- checked against
    the independent reference implementation in
    GenDatasets_SingleModeExperimentalPOVM.'''
    cutoff = 158
    nbar_eff = 1.3
    p = slt._p_mid_after_loss(nbar_eff, T=1.0, cutoff=cutoff)
    p_ref = Gen.ProbDistr1Mode('8Bin', 'PATS', ampl=nbar_eff)
    assert np.allclose(p, p_ref, atol=1e-8)

def test_p_mid_after_loss_T0_is_pure_pX():
    '''At T=0 the T*p_SPATS term vanishes, so the result must reduce to
    the plain p_X(m;n) = (m+1)*n**m/(1+n)**(m+2) cross term.'''
    cutoff = 158
    nbar_eff = 1.3
    p = slt._p_mid_after_loss(nbar_eff, T=0.0, cutoff=cutoff)
    m = np.arange(cutoff)
    p_X = (m+1)*nbar_eff**m/(1+nbar_eff)**(m+2)
    assert np.allclose(p, p_X)
    assert np.isclose(np.sum(p), 1.0, atol=1e-8)

# ------------------------------------------------------
#
#           _kernel_K
#
# ------------------------------------------------------

def test_kernel_K_sums_to_one_over_l():
    '''For fixed m, K(l,m;nu) is a genuine probability distribution over
    the output photon number l, so summing over all l (with a cutoff
    wide enough to capture the full spread) must give ~1.'''
    cutoff = 200
    for m, nu in [(0, 2.0), (5, 1.5), (20, 3.0)]:
        total = sum(slt._kernel_K(l, m, nu) for l in range(cutoff))
        assert np.isclose(total, 1.0, atol=1e-6)

def test_kernel_K_nu_zero_is_delta_function():
    '''nu=0 means no thermal ancilla couples in, i.e. pure loss: the
    kernel must degenerate to the identity K(l,m;0) = delta_{l,m}.'''
    for l, m in [(0, 0), (3, 3), (2, 5), (5, 2), (0, 10)]:
        expected = 1.0 if l == m else 0.0
        assert slt._kernel_K(l, m, 0.0) == expected

def test_kernel_K_is_symmetric_in_l_and_m():
    '''K only depends on {l,m} through Delta=|l-m|, min(l,m) and
    max(l,m), all of which are symmetric under swapping l and m.'''
    for l, m, nu in [(3, 7, 1.2), (0, 5, 0.5), (10, 2, 3.0)]:
        assert np.isclose(slt._kernel_K(l, m, nu), slt._kernel_K(m, l, nu), rtol=1e-9)

def test_kernel_K_known_values_from_vacuum():
    '''Displacing |0> by a random beta with |beta|^2 ~ Exponential(nu)
    reproduces exactly a thermal distribution with mean nu:
    K(l,0;nu) = (1/(1+nu)) * (nu/(1+nu))**l -- a whole family of
    closed-form values, not just the trivial l=m=0 case.'''
    nu = 1.7
    for l in range(15):
        expected = (1/(1+nu)) * (nu/(1+nu))**l
        assert np.isclose(slt._kernel_K(l, 0, nu), expected, rtol=1e-6)

# ------------------------------------------------------
#
#           _find_window
#
# ------------------------------------------------------

@pytest.mark.parametrize("center,nu", [(15, 2.0), (25, 4.0)])
def test_find_window_brackets_mass_above_threshold(center, nu):
    '''Every l inside the returned window must be above threshold, and
    (when not clipped by the cutoff) the points just outside it must
    be at or below threshold -- i.e. the window is a tight bracket.'''
    cutoff = 50
    thresh = 1e-4
    window = slt._find_window(center, center, nu, cutoff, thresh)
    for l in window:
        assert slt._kernel_K(l, center, nu) > thresh
    if window.start > 0:
        assert slt._kernel_K(window.start-1, center, nu) <= thresh
    if window.stop < cutoff:
        assert slt._kernel_K(window.stop, center, nu) <= thresh

def test_find_window_clips_at_lower_boundary():
    '''A window centered at 0 with a wide-spreading kernel must clip at
    0 rather than search into negative photon numbers.'''
    cutoff = 50
    window = slt._find_window(0, 0, nu=5.0, cutoff=cutoff, thresh=1e-4)
    assert window.start == 0
    assert all(l >= 0 for l in window)

def test_find_window_clips_at_upper_boundary():
    '''A window centered at cutoff-1 must clip at cutoff-1 rather than
    search past the Fock-space truncation.'''
    cutoff = 50
    window = slt._find_window(cutoff-1, cutoff-1, nu=5.0, cutoff=cutoff, thresh=1e-4)
    assert window.stop-1 == cutoff-1
    assert all(l < cutoff for l in window)

# ------------------------------------------------------
#
#           Gen_validation_dataset
#
# ------------------------------------------------------

def test_Gen_validation_dataset_smoke(tmp_path, monkeypatch):
    '''Full smoke test of the validation-dataset generator. load_detector
    is monkeypatched so the 'Eye' detector uses a small Fock cutoff
    instead of the full 729, keeping the 11(T) x 21(mbar) x 12(amplitude)
    grid fast; 'Squeezed' is used since its amplitude list (12 values)
    is already the smallest of any species, independent of detector
    type. Also exercises the save/reload round-trip performed by the
    __main__ entry point, deleting the file once the test has passed.'''
    small_cutoff = 12
    monkeypatch.setattr(
        slt, 'load_detector',
        lambda detector_type, abs_path='.': (small_cutoff, np.eye(small_cutoff))
    )

    dis, labels = slt.Gen_validation_dataset('Eye', 'Squeezed')

    n_T, n_mbar, n_amp = 11, 21, 12
    assert dis.shape == (n_T, n_mbar, n_amp, small_cutoff)
    assert labels.shape == (n_T, n_mbar, n_amp)
    assert np.all(np.isclose(np.sum(dis, axis=-1), 1.0, atol=1e-6))
    assert np.all(np.isin(labels, [0, 1]))

    out_file = tmp_path / 'smoke_test_dataset.npz'
    np.savez(out_file, probs=dis, labels=labels)
    assert out_file.exists()

    with np.load(out_file) as loaded:
        assert np.allclose(loaded['probs'], dis)
        assert np.array_equal(loaded['labels'], labels)

    out_file.unlink()
    assert not out_file.exists()

@pytest.mark.parametrize("ampl,T,mbar",[
    ([1.0, 1.0], 1.0, 0.0),   # degenerate: both branches identical -> should trivially match
    ([0.0, 3.0], 1.0, 0.0),   # well-separated, noiseless -> two Gaussians clearly not one Gaussian
    ([1.0, 2.5], 0.7, 0.5),   # separated, with loss + thermal noise
    ])
def test_MixedCoherent_is_true_statistical_mixture(ampl, T, mbar):
    '''
    A 50/50 mixture of two distinct coherent states is not Gaussian, 
    so this function checks that the output of 'Mixed Coherent' is
    the true mixture distribution 0.5*p1 + 0.5*p2.
    '''
    cutoff = 158
    POVM = np.eye(cutoff)

    mixed_distr, _ = slt.Get_outcome_distr('Mixed Coherent', ampl, T, mbar, POVM, cutoff)

    d1, _ = slt.Get_outcome_distr('Coherent', ampl[0], T, mbar, POVM, cutoff)
    d2, _ = slt.Get_outcome_distr('Coherent', ampl[1], T, mbar, POVM, cutoff)
    true_mixture = 0.5*d1 + 0.5*d2

    tvd = 0.5*np.sum(np.abs(mixed_distr - true_mixture))
    assert tvd < 1e-3

# ------------------------------------------------------
#
#           load_detector
#
# ------------------------------------------------------

DATASETSDIR = os.path.join(REPODIR, 'datasets')

@pytest.mark.parametrize("detector_type,expected_cutoff", [
    ('4Bin', 189),
    ('SNSPD', 443),
    ('8Bin', 158),
    ('PNR', 729),
])
def test_load_detector_valid(detector_type, expected_cutoff):
    cutoff, POVM = slt.load_detector(detector_type, abs_path=DATASETSDIR)
    assert cutoff == expected_cutoff
    # click_distr = POVM.T @ p requires len(p) == POVM.shape[0] == cutoff
    assert POVM.shape[0] == cutoff

def test_load_detector_eye():
    cutoff, POVM = slt.load_detector('Eye')
    assert cutoff == 729
    assert np.array_equal(POVM, np.eye(729))

def test_load_detector_invalid_type():
    with pytest.raises(Exception):
        slt.load_detector('not_a_real_detector')

# ------------------------------------------------------
#
#           Get_outcome_distr
#
# ------------------------------------------------------

@pytest.mark.parametrize("species,ampl", [
    ('Coherent', 0.8),
    ('Squeezed', 0.3),
    ('Thermal', 0.6),
    ('SPATS', 0.5),
])
def test_Get_outcome_distr_species_smoke(species, ampl):
    cutoff = 158
    POVM = np.eye(cutoff)
    click_distr, label = slt.Get_outcome_distr(species, ampl, T=1.0, mbar=0.0, POVM=POVM, cutoff=cutoff)
    assert np.isclose(np.sum(click_distr), 1.0)
    assert np.all(click_distr >= -1e-9)
    assert label in (0, 1)

def test_Get_outcome_distr_invalid_species():
    with pytest.raises(Exception):
        slt.Get_outcome_distr('not_a_real_species', 1.0, T=1.0, mbar=0.0, POVM=np.eye(158), cutoff=158)

@pytest.mark.parametrize("species,ampl,expected_label", [
    ('Coherent', 1.0, 0),   # diag_cov == 0.5 exactly -> not < 0.5 - 1e-9
    ('Squeezed', 0.5, 1),   # squeezed quadrature variance e^-1 * 0.5 < 0.5
    ('Thermal', 1.0, 0),    # thermal noise only ever increases variance above 0.5
])
def test_Get_outcome_distr_label_gaussian(species, ampl, expected_label):
    '''The nonclassicality label for Gaussian species is derived from
    whether the smallest quadrature variance dips below the T=1,mbar=0
    coherent-state (shot-noise) level of 0.5.'''
    cutoff = 158
    POVM = np.eye(cutoff)
    _, label = slt.Get_outcome_distr(species, ampl, T=1.0, mbar=0.0, POVM=POVM, cutoff=cutoff)
    assert label == expected_label

def test_Get_outcome_distr_label_SPATS_matches_Dk_formula():
    '''SPATS states have no covariance matrix, so Get_outcome_distr's
    try/except falls back to the discriminant Dk = p0*p2*2/p1**2
    (< 1 indicates a nonclassical photon-number distribution). Check
    the function's label agrees with computing Dk directly from the
    same underlying distribution.'''
    cutoff = 158
    T, nbar, mbar = 0.7, 1.2, 0.3
    POVM = np.eye(cutoff)
    _, label = slt.Get_outcome_distr('SPATS', nbar, T=T, mbar=mbar, POVM=POVM, cutoff=cutoff)

    p = slt.LossySPATS(T=T, nbar=nbar, mbar=mbar, cutoff=cutoff)
    Dk = p[0]*p[2]*2/(p[1]**2)
    expected_label = int(Dk < 1.)
    assert label == expected_label

@pytest.mark.parametrize("ampl", [[0.5, 0.5], [0.2, 1.8]])
def test_Get_outcome_distr_label_MixedCoherent_always_zero(ampl):
    '''"Mixed Coherent" never binds a local `cov`, so the try/except
    always falls into the except-branch default of label=0, regardless
    of amplitudes -- this pins down that (perhaps unintended) behavior.'''
    cutoff = 158
    POVM = np.eye(cutoff)
    _, label = slt.Get_outcome_distr('Mixed Coherent', ampl, T=1.0, mbar=0.0, POVM=POVM, cutoff=cutoff)
    assert label == 0

def test_Get_outcome_distr_normalization_tail_fill_branch():
    '''When sum(click_distr[:-1]) < 1 (e.g. a heavily truncated Fock
    cutoff), the function fills the last bin with the missing mass
    instead of renormalizing, leaving every other bin untouched.'''
    cutoff = 5
    T, ampl, mbar = 1.0, 50.0, 0.0
    POVM = np.eye(cutoff)
    mu, cov = slt.LossyGauss(T, alpha=0.0, r=0.0, nbar=ampl, mbar=mbar)
    p = slt.SampleState(mu, cov, cutoff)
    assert np.sum(p[:-1], dtype=np.float64) < 1.0

    click_distr, _ = slt.Get_outcome_distr('Thermal', ampl, T, mbar, POVM, cutoff)
    assert np.allclose(click_distr[:-1], p[:-1])
    assert np.isclose(click_distr[-1], 1 - np.sum(p[:-1], dtype=np.float64))
    assert np.isclose(np.sum(click_distr, dtype=np.float64), 1.0)

def test_Get_outcome_distr_normalization_renormalize_branch():
    '''An (artificial) POVM that is not sub-stochastic can push
    sum(click_distr[:-1]) above 1; the function then zeroes the last
    bin and renormalizes the whole vector instead of tail-filling it.'''
    cutoff = 158
    T, ampl, mbar = 1.0, 0.3, 0.0
    POVM = 100*np.eye(cutoff)
    mu, cov = slt.LossyGauss(T, alpha=0.0, r=0.0, nbar=ampl, mbar=mbar)
    p = slt.SampleState(mu, cov, cutoff)
    assert np.sum(100*p[:-1], dtype=np.float64) >= 1.0

    click_distr, _ = slt.Get_outcome_distr('Thermal', ampl, T, mbar, POVM, cutoff)
    expected = np.array(100*p, dtype=np.float64)
    expected[-1] = 0.0
    expected = expected/np.sum(expected)
    assert np.allclose(click_distr, expected)
    assert click_distr[-1] == 0.0
    assert np.isclose(np.sum(click_distr, dtype=np.float64), 1.0)

# print(test_GS_tau_1(alpha=1.0, r=0.5, nbar=1.0, mbar=0.0))
# print(test_SPATS_tau_1(alpha=1.0, nbar=1.0, mbar=0.0))
# print('If we lose all:',test_tau_0(1.0))
# print(test_tau_less_1(T=0.5, nbar=1.1))