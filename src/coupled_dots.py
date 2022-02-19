import numpy as np
import scipy.special

def random_array(*shape, complex=True):
  """Random array of a given shape. Elements are normal random variables with
  unit RMS on average.
  """
  R = np.random.randn(*shape)
  if complex:
    I = np.random.randn(*shape)
    return (R + 1j * I) / np.sqrt(2)
  else:
    return R

def ge_matrix(n, ensemble='U'):
  """Random nxn matrix sampled from a Gaussian ensemble."""
  assert ensemble in ('O', 'U', 'S')
  if ensemble == 'O':
    M = random_array(n, n, complex=False)
  elif ensemble == 'U':
    M = random_array(n, n, complex=True)
  elif ensemble == 'S':
    assert n == 2 * (n // 2)
    dtype = np.complex128
    e0 = np.eye(2, dtype=dtype)
    e1 = np.array([[1j, 0], [0, -1j]], dtype=dtype)
    e2 = np.array([[0, 1], [-1, 0]], dtype=dtype)
    e3 = np.array([[0, 1j], [1j, 0]], dtype=dtype)
    m = n // 2
    M = np.kron(random_array(m, m, complex=False), e0)
    M += np.kron(random_array(m, m, complex=False), e1)
    M += np.kron(random_array(m, m, complex=False), e2)
    M += np.kron(random_array(m, m, complex=False), e3)
    M /= np.sqrt(2)
  return (M + M.T.conj()) / np.sqrt(2)

def hamiltonian(N, g, ensemble='U'):
  """Quantum dot Hamiltonian. The Hamiltonian is random, but one (GOE/GUE) or two (GSE) states are treated as special and their coupling to the other states is reduced by a factor of np.sqrt(g).

  Args:
    N: Number of states.
    g: Effective line width.
    ensemble: 'O', 'U', or 'S' for GOE, GUE, GSE.

  Returns:
    NxN Hamiltonian matrix.
  """
  assert ensemble in ('O', 'U', 'S')
  assert N == 2 * (N // 2)
  H = ge_matrix(N, ensemble=ensemble) / np.sqrt(N)
  H -= np.eye(N) * np.trace(H) / N
  partition = 2 if ensemble == 'S' else 1
  H[:partition, partition:] *= np.sqrt(g)
  H[partition:, :partition] *= np.sqrt(g)
  return H

def random_state(N1, N2, complex=True):
  """Random initial state on first N1 states of the full N1 + N2-dimensional
  space.
  """
  psi1 = np.random.randn(N1)
  if complex:
    psi1 = psi1 + 1j * np.random.randn(N1)
  psi1 /= np.linalg.norm(psi1)
  psi2 = np.zeros(N2)
  return np.concatenate((psi1, psi2))

def basis_state(i, N):
  """A basis state at location `i`."""
  psi = np.zeros(N)
  psi[i] = 1
  return psi

def level_spacing_ratios(E):
  """Energy level spacing ratios from neighboring pairs of levels in a spectrum
  E.
  """
  s = np.diff(E)
  r = s[1:] / s[:-1]
  return np.minimum(r, 1/r)

def density_of_states(E):
  """Estimate the density of states at zero energy."""
  mask = np.abs(E) < 0.25 * np.std(E)
  E_mid = E[mask]
  N_mid = mask.sum()
  pars = np.polyfit(E_mid, np.arange(N_mid), 3)
  return pars[2]

def final_p_res(N1, N2, g):
  """Tobias' result for the probability of staying on the first dot at long
  times.
  """
  gamma = g * N2 / N1
  large_gamma = 1e2
  small_gamma = 1e-2
  if gamma < small_gamma:
    P = 1 - np.sqrt(np.pi * gamma) / 2
  elif gamma > large_gamma:
    P = 1 / gamma
  else:
    P = 1 - gamma - 0.5 * np.sqrt(np.pi * gamma) * (1 - 2 * gamma) \
      * np.exp(gamma) * scipy.special.erfc(np.sqrt(gamma))
  return P

@np.vectorize
def _p_res_integrand(gamma, tau, lb, lf):
  x = 2 * tau - (lb - lf)
  if x < 0:
    return 0
  else:
    mub = np.sqrt(lb * lb - 1)
    zb = 2 * gamma * x * mub
    I0 = scipy.special.ive(0, zb)
    I1 = scipy.special.ive(1, zb)
    numerator = x * x * np.exp(-zb * (lb / mub - 1)) * (lb * I0 - mub * I1)
    denominator = lb - lf
    return numerator / denominator

@np.vectorize
def p_res(t, N1, N2, g, num=100):
  """Tobias' result for time dependence of the probability of staying on the
  first dot.
  """
  gamma = g * N2 / N1
  tau = t / (2 * N2)

  lfmin, lfmax = max(1 - 2 * tau, -1), 1
  dlf = (lfmax - lfmin) / num
  lfgrid = np.linspace(lfmin + 0.5 * dlf, lfmax - 0.5 * dlf, num=num)

  integral = 0
  for lf in lfgrid:
    lbmin, lbmax = 1, max(2 * tau + lf, 1)
    dlb = (lbmax - lbmin) / num
    lbgrid = np.linspace(lbmin + 0.5 * dlb, lbmax - 0.5 * dlb, num=num)
    integral += dlf * dlb * np.sum(_p_res_integrand(gamma, tau, lbgrid, lf))

  return np.exp(-4 * gamma * tau) + 2 * gamma * gamma * integral

@np.vectorize
def p_res_fgr(t, N1, N2, g):
  """The time dependence of the probability of staying on the first dot in the
  FGR regime.
  """
  gamma = g * N2 / N1
  tau = t / (2 * N2)
  if tau < 1:
    return np.exp(-4 * gamma * tau) + (1 + tau) / (2 * gamma)
  else:
    return 1 / gamma
