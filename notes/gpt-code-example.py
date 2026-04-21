import numpy as np
import matplotlib.pyplot as plt

from qiskit.primitives import StatevectorEstimator

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper

from qiskit.circuit.library import EfficientSU2

# ============================================================
# 🔧 Função para gerar Hamiltoniano
# ============================================================

def get_qubit_hamiltonian(atom_string, charge=0, spin=0):
    driver = PySCFDriver(atom=atom_string, basis="sto3g", charge=charge, spin=spin)
    problem = driver.run()

    mapper = ParityMapper()
    second_q_op = problem.hamiltonian.second_q_op()

    qubit_op = mapper.map(second_q_op)

    return qubit_op


# ============================================================
# 🔧 Função para calcular energia com ansatz
# ============================================================

def compute_energy(ansatz, hamiltonian, parameters):
    estimator = StatevectorEstimator()
    # Primitive V2 expects a list of PUBs in the format:
    # (circuit, observable, parameter_values)
    pub = (ansatz, hamiltonian, [parameters])
    job = estimator.run([pub])
    result = job.result()[0]
    return float(result.data.evs[0])


# ============================================================
# 🔧 Varredura de parâmetro
# ============================================================

def scan_parameter(ansatz, hamiltonian, param_index=0, n_points=50):
    thetas = np.linspace(0, 2*np.pi, n_points)
    energies = []

    base_params = np.zeros(ansatz.num_parameters)

    for theta in thetas:
        params = base_params.copy()
        params[param_index] = theta

        energy = compute_energy(ansatz, hamiltonian, params)
        energies.append(energy)

    return thetas, np.array(energies)


# ============================================================
# 🔧 Fourier + Entropia
# ============================================================

def compute_fourier(energies):
    coeffs = np.fft.fft(energies)
    freqs = np.fft.fftfreq(len(energies))

    magnitudes = np.abs(coeffs)

    # normalizar
    p = magnitudes**2 / np.sum(magnitudes**2)

    entropy = -np.sum(p * np.log(p + 1e-12))

    return freqs, magnitudes, entropy


# ============================================================
# 🧪 H2
# ============================================================

print("Running H2...")

h2 = "H 0 0 0; H 0 0 0.735"
ham_h2 = get_qubit_hamiltonian(h2)

ansatz_h2 = EfficientSU2(ham_h2.num_qubits, reps=1)

thetas_h2, energies_h2 = scan_parameter(ansatz_h2, ham_h2)

freqs_h2, mag_h2, entropy_h2 = compute_fourier(energies_h2)

print("H2 entropy:", entropy_h2)


# ============================================================
# 🧪 LiH
# ============================================================

print("Running LiH...")

lih = "Li 0 0 0; H 0 0 1.6"
ham_lih = get_qubit_hamiltonian(lih)

ansatz_lih = EfficientSU2(ham_lih.num_qubits, reps=1)

thetas_lih, energies_lih = scan_parameter(ansatz_lih, ham_lih)

freqs_lih, mag_lih, entropy_lih = compute_fourier(energies_lih)

print("LiH entropy:", entropy_lih)


# ============================================================
# 📊 PLOTS
# ============================================================

# Energia vs theta
plt.figure()
plt.plot(thetas_h2, energies_h2, label="H2")
plt.plot(thetas_lih, energies_lih, label="LiH")
plt.xlabel("Theta")
plt.ylabel("Energy")
plt.legend()
plt.title("Energy vs Parameter")
plt.savefig("energy_curves.png")

# Espectro
plt.figure()
plt.stem(np.abs(freqs_h2), mag_h2, label="H2")
plt.stem(np.abs(freqs_lih), mag_lih, linefmt='r-', markerfmt='ro', label="LiH")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.legend()
plt.title("Fourier Spectrum")
plt.savefig("fourier_spectrum.png")

plt.show()