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
    return (M + M.T.conj()) / np.sqrt(2)
  elif ensemble == 'U':
    M = random_array(n, n, complex=True)
    return (M + M.T.conj()) / np.sqrt(2)
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
    return (M + M.T.conj()) / 2

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

def final_p_res(N, g):
  """Tobias' result for the probability of staying on the first dot at long
  times.
  """
  gamma = g * N
  large_gamma = 1e2
  small_gamma = 1e-2
  if gamma < small_gamma:
    P = 1 - np.sqrt(np.pi * gamma) / 2
  elif gamma > large_gamma:
    P = 1 / gamma
  else:
    P = 1 - gamma - 0.5 * np.sqrt(np.pi * gamma) * (1 - 2 * gamma) \
      * np.exp(gamma) * scipy.special.erfc(np.sqrt(gamma))
  correction = 1 + (gamma / N)
  return correction * P

def integrand(lb, lf, tau, gamma):
  x = 2 * tau - (lb - lf)
  mub = np.sqrt(lb * lb - 1)
  zb = 2 * gamma * x * mub
  I0 = scipy.special.ive(0, zb)
  I1 = scipy.special.ive(1, zb)
  numerator = x * x * np.exp(-zb * (lb / mub - 1)) * (lb * I0 - mub * I1)
  denominator = lb - lf
  return numerator / denominator

def inner_integral(lf, tau, gamma):
  lbmin, lbmax = 1, max(2 * tau + lf, 1)
  return scipy.integrate.quad(integrand, lbmin, lbmax, args=(lf, tau, gamma))[0]

def integral(tau, gamma):
  lfmin, lfmax = max(1 - 2 * tau, -1), 1
  return scipy.integrate.quad(inner_integral, lfmin, lfmax, args=(tau, gamma))[0]

def p_res(ts, N, g):
  """Tobias' result for time dependence of the probability of staying in the
  initial state.
  """
  gamma = g * N
  taus = ts / (2 * N)
  integrals = np.zeros_like(taus)
  corrections = np.zeros_like(taus)
  for i, tau in enumerate(taus):
    integrals[i] = integral(tau, gamma)
  p = np.exp(-4 * gamma * taus) + 2 * gamma * gamma * integrals
  corrections = 1 + (gamma / N) * np.minimum(taus * gamma, 1)
  return corrections * p

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
