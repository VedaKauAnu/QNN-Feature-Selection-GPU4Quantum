"""
Quantum Encoding Methods
Converts classical data into quantum states
"""

import numpy as np
from abc import ABC, abstractmethod


class QuantumEncoder(ABC):
    """Base class for encoding classical data into quantum states"""
    
    @abstractmethod
    def encode(self, classical_data):
        """Convert classical features to quantum representation"""
        pass


class AngleEncoder(QuantumEncoder):
    """
    Angle encoding: features → rotation angles
    Each feature becomes an RY rotation angle.
    
    Pros: 
    - Simple and interpretable
    - Works with any number of features ≤ qubits
    - Computationally efficient
    
    Cons: 
    - Limited expressiveness
    - One feature per qubit only
    """
    
    def __init__(self, scale=np.pi):
        """
        Args:
            scale: Maximum rotation angle (default π for [0,1]→[0,π])
        """
        self.scale = scale
        
    def encode(self, classical_data):
        """
        Convert features to rotation angles in [0, scale]
        
        Args:
            classical_data: numpy array of shape (n_features,)
        
        Returns:
            angles: array of rotation angles
        """
        # Ensure data is in [0, 1] range, then scale
        angles = np.clip(classical_data, 0, 1) * self.scale
        return angles


class AmplitudeEncoder(QuantumEncoder):
    """
    Amplitude encoding: features → quantum state amplitudes
    Encodes 2^n values in n qubits (exponentially efficient!)
    
    Pros: 
    - Exponentially efficient storage
    - Can encode many features in few qubits
    
    Cons: 
    - Requires normalization
    - Circuit depth increases
    - More complex implementation
    """
    
    def encode(self, classical_data):
        """
        Prepare data for amplitude encoding
        
        Args:
            classical_data: numpy array of shape (n_features,)
        
        Returns:
            normalized_amplitudes: L2-normalized amplitudes
        """
        # Pad to power of 2 if needed
        n_qubits_needed = int(np.ceil(np.log2(len(classical_data))))
        target_length = 2 ** n_qubits_needed
        
        if len(classical_data) < target_length:
            padded = np.zeros(target_length)
            padded[:len(classical_data)] = classical_data
        else:
            padded = classical_data[:target_length]
        
        # L2 normalize for valid quantum state
        norm = np.linalg.norm(padded)
        if norm > 0:
            amplitudes = padded / norm
        else:
            amplitudes = padded
            
        return amplitudes


# Optional: Add more encoding methods later
class BasisEncoder(QuantumEncoder):
    """
    Basis encoding: features → computational basis states
    Simple but not commonly used for ML
    """
    
    def encode(self, classical_data):
        """
        Encode features as binary basis states
        
        Args:
            classical_data: numpy array of shape (n_features,)
        
        Returns:
            binary_state: array of 0s and 1s
        """
        # Threshold at 0.5
        binary = (classical_data > 0.5).astype(int)
        return binary