from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2


def build_ansatz(num_qubits: int, reps: int = 2) -> QuantumCircuit:
    """
    Constrói o ansatz variacional utilizado no algoritmo VQE.

    O circuito é baseado no EfficientSU2, que aplica rotações
    parametrizadas intercaladas com camadas de emaranhamento completo
    entre os qubits.

    Parameters
    ----------
    num_qubits : int
        Número de qubits do sistema quântico.
    reps : int, optional
        Número de repetições das camadas variacionais.

    Returns
    -------
    QuantumCircuit
        Circuito quântico parametrizado que representa o ansatz.
    """

    ansatz = EfficientSU2(
        num_qubits=num_qubits,
        reps=reps,
        entanglement="full"
    )

    return ansatz
