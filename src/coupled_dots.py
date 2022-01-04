import numpy as np
import scipy.special

def random_array(*shape):
  """Random array of a given shape. Elements are complex normal random
  variables with unit norm on average.
  """
  R = np.random.randn(*shape)
  I = np.random.randn(*shape)
  return (R + 1j * I) / np.sqrt(2)

def gue_matrix(n):
  """Random nxn matrix sampled from the Gaussian unitary ensemble."""
  M = random_array(n, n)
  return (M + M.T.conj()) / np.sqrt(2)

def hamiltonian(N1, N2, g):
  """Hamiltonian of two coupled quantum dots. Each dot has a random Hamiltonian,
  and the coupling between dots is random.

  Args:
    N1: Number of states in the first dot.
    N2: Number of states in the second dot.
    g: Effective rate of coupling between dots.

  Returns:
    Hamiltonian matrix.
  """
  Ntot = N1 + N2
  t12 = np.sqrt(g / (N1 * N2))
  # intra dot Hamiltonians
  H1 = gue_matrix(N1) / np.sqrt(N1)
  H1 = H1 - np.eye(N1) * np.trace(H1) / N1
  H2 = gue_matrix(N2) / np.sqrt(N2)
  H2 = H2 - np.eye(N2) * np.trace(H2) / N2
  # inter dot couplings
  W = t12 * random_array(N1, N2)
  # full Hamiltonian
  H = np.vstack((
    np.hstack((H1, W,)),
    np.hstack((W.T.conj(), H2,)),
  ))
  return H

def initial_state(N1, N2):
  """Random initial state on first N1 states of the full N1 + N2-dimensional
  space.
  """
  psi1 = np.random.randn(N1) + 1j * np.random.randn(N1)
  psi1 /= np.linalg.norm(psi1)
  psi2 = np.zeros(N2)
  return np.concatenate((psi1, psi2))

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

def tobias_approx_result(N1, N2, g):
  """Tobias' approximate result for the probability of staying on the first dot at long times.
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
