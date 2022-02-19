import numpy as np
import h5py
import sys
sys.path.append("../")
from src.coupled_dots import *

"""In these simulations we fix N1 = 1 so that we have one state decaying into
a dot. We can choose GOE, GUE, or GSE Hamiltonians.
"""

# system size
N = int(sys.argv[1])
# ensemble
ensemble = str(sys.argv[2])
# coupling between the dots
ngammas = 10
gammas = np.logspace(-np.log10(N), np.log10(N), ngammas)
gs = gammas / N
# number of samples
nsamples = 500
# time points
nts = 100
# chosen states (these work for all ensembles)
k1, k2a, k2b = 0, 2, 4

for gamma, g in zip(gammas, gs):
  # times
  ti = 0.1 * g
  tf = 5 * max(2 * N, N / np.sqrt(gamma))
  ts = np.logspace(np.log10(ti), np.log10(tf), nts)
  # for storing results
  data = {
    'E': [],
    'P1(E)': [],
    'P2(E)': [],
    'P11(inf)': [],
    'P22(inf)': [],
    'P22b(inf)': [],
    'P11(t)': [],
    'P22(t)': [],
    'P22b(t)': [],
    't': ts,
  }
  for sample in range(nsamples):
    # sample the Hamiltonian
    H = hamiltonian(N, g, ensemble)
    # energy eigenstates
    E, V = np.linalg.eigh(H)

    # <E|i>
    amp_1_E = V[k1].conj()
    amp_2a_E = V[k2a].conj()
    amp_2b_E = V[k2b].conj()
    # |<E|i>|^2
    prob_1_E = np.abs(amp_1_E)**2
    prob_2a_E = np.abs(amp_2a_E)**2
    prob_2b_E = np.abs(amp_2b_E)**2
    # infinite time transition probabilities
    T11_inf = np.dot(prob_1_E, prob_1_E)
    T2a2a_inf = np.dot(prob_2a_E, prob_2a_E)
    T2a2b_inf = np.dot(prob_2a_E, prob_2b_E)

    # time-dependent transition probabilities
    T11_t = np.zeros_like(ts)
    T2a2a_t = np.zeros_like(ts)
    T2a2b_t = np.zeros_like(ts)
    for j, t in enumerate(ts):
      phases = np.exp(-1j * E * t)
      T11_t[j] = np.abs(np.sum(phases * amp_1_E * amp_1_E.conj()))**2
      T2a2a_t[j] = np.abs(np.sum(phases * amp_2a_E * amp_2a_E.conj()))**2
      T2a2b_t[j] = np.abs(np.sum(phases * amp_2a_E * amp_2b_E.conj()))**2

    # record data
    data['E'].append(E)
    data['P1(E)'].append(prob_1_E)
    data['P2(E)'].append(prob_2a_E)
    data['P11(inf)'].append(T11_inf)
    data['P22(inf)'].append(T2a2a_inf)
    data['P22b(inf)'].append(T2a2b_inf)
    data['P11(t)'].append(T11_t)
    data['P22(t)'].append(T2a2a_t)
    data['P22b(t)'].append(T2a2b_t)

  # save data
  barcode = str(np.random.rand())[2:8]
  # dir = "/scratch/gpfs/aorm/coupled_dots"
  dir = "../data"
  fname = f"{dir}/N={N}_ensemble={ensemble}_gamma={np.round(gamma, 6)}_{barcode}.hdf5"
  f = h5py.File(fname, "w")
  f.attrs['N'] = N
  f.attrs['gamma'] = gamma
  f.attrs['g'] = g
  f.attrs['ensemble'] = ensemble
  for ky in data.keys():
    f.create_dataset(ky, data=data[ky])
  f.close()
