from qiskit import QuantumCircuit
from qiskit.circuit import Gate


class QHTGate(Gate):
    """
    QHTGate -> a gate that implements Quantum Hartley Transform
    using either Linear Combination of Unitaries (LCU) or
    Recursive method, depending on the 'type' attribute.
    
    LCU:
    Quantum Hartley Transform using LCU
    
    Warning: number of qubits one will need to use this gate is:
     if num_qubits=1 -> # qubits = 1
     if num_qubits >=2 -> # qubits = 2*num_qubits
    
    Recursvie:
    Quantum Hartley Transform using recursive method.
    
    Warning: number of qubits one will need to use this gate is:
     if num_qubits=1 -> # qubits = 1
     if num_qubits =2 -> # qubits = 3
     if num_qubits >=3 -> # qubits = 2*num_qubits - 2

    Example:
        QHTGate(num_qubits=3, type="LCU")   # internally allocates ancillas
    """

    def __init__(
        self,
        num_qubits: int,
        type: str = "LCU",
        swap: bool = True,
    ):
        """
        Args:
            num_qubits: number of data qubits
            type: Either "LCU" or "REC".
            swap: Only relevant if type="REC".
        """
        self.type = type
        self.swap = swap

        num_data = num_qubits 
        
             
        if self.type == "LCU":
            if num_data == 1:
                num_ancillas = 0
            else:
                num_ancillas = num_data
        elif self.type == "REC":
            if num_data == 1:
                num_ancillas = 0
            elif num_data == 2:
                num_ancillas = 1
                
            else:
                num_ancillas = num_data - 2
        else:
            raise ValueError(f"Unsupported QHT type: {self.type}")

        data_qubits = list(range(num_data))
        ancilla_qubits = list(range(num_data, num_data + num_ancillas))
        total_qubits = num_data + num_ancillas
        
        self.data_qubits = data_qubits
        self.ancilla_qubits = ancilla_qubits



    
        if self.type == "LCU":
            if len(data_qubits) == 1:
                pass
            else:
                if len(ancilla_qubits) != len(data_qubits):
                    raise ValueError(
                        f"Ancilla qubits must be exactly data_qubits"
                        f"(got {len(ancilla_qubits)} for {len(data_qubits)} data)"
                                         )
        elif self.type == "REC":
            if len(data_qubits) == 1 or len(data_qubits) == 2:
                pass
            else:
                if len(ancilla_qubits) != len(data_qubits) - 2:
                    raise ValueError("Number of ancilla qubits must be data_qubits - 2")
        else:
            raise ValueError(f"Unsupported QHT type: {self.type}")

        super().__init__("QHTGate", total_qubits, [])
        self.label = f"QHT_{self.type}"

    # ---- Qiskit definition ----
    def _define(self):
        definition_circuit = QuantumCircuit(self.num_qubits, name=self.name)
        
        if self.type == "LCU":
            from qht import _build_qht_lcu
            if len(self.data_qubits) == 1:
                _build_qht_lcu(definition_circuit, None, None, self.data_qubits, self.ancilla_qubits)
            else:
                helper = self.ancilla_qubits[0]
                selector = self.ancilla_qubits[1]
                data_qubits = self.data_qubits
                anc_qubits = self.ancilla_qubits[2:]

                
                _build_qht_lcu(definition_circuit, helper, selector, data_qubits, anc_qubits)
        
        elif self.type == "REC":
            from qht import _build_qht_rec
            
        
            if len(self.data_qubits) == 1:
                _build_qht_rec(definition_circuit, None, self.data_qubits, self.ancilla_qubits)
        
            else:
                auxiliary_q = self.ancilla_qubits[0]
                anc_rest = self.ancilla_qubits[1:]
                

                
                _build_qht_rec(definition_circuit, auxiliary_q, self.data_qubits, anc_rest)

            if self.swap:
                n = len(self.data_qubits)
                for i in range(n // 2):
                    definition_circuit.swap(self.data_qubits[i], self.data_qubits[n - 1 - i])

        self.definition = definition_circuit





    
