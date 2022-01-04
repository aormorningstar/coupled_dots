import numpy as np
import h5py
import sys
sys.path.append("../")
from src.coupled_dots import *

# size of system
N1 = 1
N2 = 500
N = N1 + N2
# coupling between the dots
ngammas = 20
gammas = 10**np.linspace(-4, 3, ngammas)
gs = gammas * N1 / N2
# number of samples
nsamples = 500
# number of time points
nts = 100

for gamma, g in zip(gammas, gs):
  # Fermi's Golden Rule timescale
  t_fgr = 1 / g
  # times to record
  tf = t_fgr * 10
  ts = np.linspace(0, tf, nts)
  # Tobias' approximate result for P1
  P1_inf_tobias = tobias_approx_result(N1, N2, g)
  # for storing results
  data = {
    'E': [],
    'P(E)': [],
    'P1(inf)': [],
    'P1(t)': [],
    't': ts,
  }
  for sample in range(nsamples):
    # sample the Hamiltonian
    H = hamiltonian(N1, N2, g)
    # initial state
    psi = initial_state(N1, N2)
    # energy eigenstates
    E, V = np.linalg.eigh(H)

    # <E|psi> = sum_i <E|i><i|psi>
    amp_psi_E = V.T.conj().dot(psi)
    # |<E|psi>|^2
    prob_psi_E = np.abs(amp_psi_E)**2
    # |<i|E>|^2
    prob_i_E = np.abs(V)**2
    # sum_E |<i|E>|^2 |<E|psi>|^2
    prob_psi_i = prob_i_E.dot(prob_psi_E)
    # prob first dot at infinite time
    P1_inf = prob_psi_i[:N1].sum()
    # time evolution of P1
    A_i_E = np.einsum("iE,E->iE", V[:N1], amp_psi_E)
    P1_t = np.zeros_like(ts)
    for j in range(len(ts)):
      amp_i_psit = A_i_E.dot(np.exp(-1j * E * ts[j]))
      P1_t[j] = np.vdot(amp_i_psit, amp_i_psit).real

    # record data
    data['E'].append(E)
    data['P(E)'].append(prob_psi_E)
    data['P1(inf)'].append(P1_inf)
    data['P1(t)'].append(P1_t)

  # save data
  barcode = str(np.random.rand())[2:8]
  # dir = "/scratch/gpfs/aorm/coupled_dots"
  dir = "../data"
  fname = f"{dir}/N1={N1}_N2={N2}_gamma={gamma}_{barcode}.hdf5"
  f = h5py.File(fname, "w")
  f.attrs['N1'] = N1
  f.attrs['N2'] = N2
  f.attrs['gamma'] = gamma
  f.attrs['g'] = g
  f.attrs['Tobias P1(inf)'] = P1_inf_tobias,
  for ky in data.keys():
    f.create_dataset(ky, data=data[ky])
  f.close()
