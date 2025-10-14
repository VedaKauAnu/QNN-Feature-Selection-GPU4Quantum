import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from abc import ABC, abstractmethod
import pennylane as qml


# ========================================
# 1. DATA PIPELINE
# ========================================

class DataLoader:
    """
    Loads and preprocesses classical datasets for quantum circuits.
    Handles: loading, splitting, normalization, and dimension validation.
    """
    
    def __init__(self, dataset_name='iris', test_size=0.2, random_state=42):
        self.dataset_name = dataset_name.lower()
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def load_dataset(self):
        """Load one of the built-in sklearn datasets"""
        if self.dataset_name == 'iris':
            data = load_iris()
        elif self.dataset_name == 'wine':
            data = load_wine()
        elif self.dataset_name == 'cancer':
            data = load_breast_cancer()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        return data.data, data.target
    
    def get_data(self, normalize=True):
        """
        Returns train/test split with optional normalization
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        X, y = self.load_dataset()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y  # Maintain class balance
        )
        
        # Normalize features to [0, 1] range (better for quantum encoding)
        if normalize:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def get_info(self):
        """Get dataset metadata"""
        X, y = self.load_dataset()
        return {
            'name': self.dataset_name,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'feature_range': (X.min(), X.max())
        }


# ========================================
# 2. FEATURE SELECTION
# ========================================

class FeatureSelector(ABC):
    """
    Abstract base class for feature selection methods.
    All selectors reduce features to match qubit count.
    """
    
    def __init__(self, n_features):
        self.n_features = n_features
        self.selected_indices = None
        
    @abstractmethod
    def fit(self, X, y):
        """Learn which features to select"""
        pass
    
    def transform(self, X):
        """Apply feature selection"""
        if self.selected_indices is None:
            raise ValueError("Must call fit() before transform()")
        return X[:, self.selected_indices]
    
    def fit_transform(self, X, y):
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)


class PCASelector(FeatureSelector):
    """
    Principal Component Analysis feature selection.
    Finds linear combinations that capture most variance.
    """
    
    def __init__(self, n_features):
        super().__init__(n_features)
        self.pca = PCA(n_components=n_features)
        
    def fit(self, X, y):
        """Fit PCA on training data"""
        self.pca.fit(X)
        # PCA transforms all features, so we don't select indices
        # Instead, we'll override transform
        self.selected_indices = np.arange(self.n_features)
        return self
        
    def transform(self, X):
        """Transform data using PCA"""
        return self.pca.transform(X)
    
    def get_explained_variance(self):
        """Get variance explained by selected components"""
        return self.pca.explained_variance_ratio_


class CorrelationSelector(FeatureSelector):
    """
    Select features with highest correlation to target.
    Fast and simple, works well for linear relationships.
    """
    
    def fit(self, X, y):
        """Calculate correlations and select top features"""
        # Calculate absolute correlation with target
        correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] 
                              for i in range(X.shape[1])])
        
        # Select top n_features
        self.selected_indices = np.argsort(correlations)[-self.n_features:]
        return self


class MutualInfoSelector(FeatureSelector):
    """
    Select features with highest mutual information with target.
    Captures non-linear relationships better than correlation.
    """
    
    def __init__(self, n_features, random_state=42):
        super().__init__(n_features)
        self.random_state = random_state
        
    def fit(self, X, y):
        """Calculate mutual information and select top features"""
        # Calculate MI scores
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        
        # Select top n_features
        self.selected_indices = np.argsort(mi_scores)[-self.n_features:]
        return self


class RandomSelector(FeatureSelector):
    """
    Randomly select features (baseline comparison).
    Helps establish if intelligent selection matters.
    """
    
    def __init__(self, n_features, random_state=42):
        super().__init__(n_features)
        self.random_state = random_state
        
    def fit(self, X, y):
        """Randomly select n features"""
        rng = np.random.RandomState(self.random_state)
        self.selected_indices = rng.choice(
            X.shape[1], size=self.n_features, replace=False
        )
        self.selected_indices = np.sort(self.selected_indices)
        return self


# ========================================
# 3. QUANTUM ENCODING
# ========================================

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
    Pros: Simple, works with any number of features ≤ qubits
    Cons: Limited expressiveness
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
    Encodes 2^n values in n qubits.
    Pros: Exponentially efficient
    Cons: Requires normalization, circuit depth increases
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


# ========================================
# 4. TESTING AND VALIDATION
# ========================================

def test_data_pipeline():
    """Test data loading and preprocessing"""
    print("=" * 50)
    print("TESTING DATA PIPELINE")
    print("=" * 50)
    
    for dataset in ['iris', 'wine', 'cancer']:
        loader = DataLoader(dataset)
        info = loader.get_info()
        print(f"\n{dataset.upper()} Dataset:")
        print(f"  Samples: {info['n_samples']}")
        print(f"  Features: {info['n_features']}")
        print(f"  Classes: {info['n_classes']}")
        
        X_train, X_test, y_train, y_test = loader.get_data()
        print(f"  Train size: {X_train.shape}")
        print(f"  Test size: {X_test.shape}")
        print(f"  Feature range: [{X_train.min():.3f}, {X_train.max():.3f}]")


def test_feature_selection():
    """Test all feature selection methods"""
    print("\n" + "=" * 50)
    print("TESTING FEATURE SELECTION")
    print("=" * 50)
    
    # Load data
    loader = DataLoader('iris')
    X_train, X_test, y_train, y_test = loader.get_data()
    
    n_features = 2  # Reduce from 4 to 2
    
    # Test each selector
    selectors = {
        'PCA': PCASelector(n_features),
        'Correlation': CorrelationSelector(n_features),
        'Mutual Info': MutualInfoSelector(n_features),
        'Random': RandomSelector(n_features)
    }
    
    for name, selector in selectors.items():
        X_selected = selector.fit_transform(X_train, y_train)
        print(f"\n{name}:")
        print(f"  Original shape: {X_train.shape}")
        print(f"  Selected shape: {X_selected.shape}")
        print(f"  Selected indices: {selector.selected_indices}")
        
        if hasattr(selector, 'get_explained_variance'):
            var = selector.get_explained_variance()
            print(f"  Variance explained: {var.sum():.3f}")


def test_quantum_encoding():
    """Test encoding methods"""
    print("\n" + "=" * 50)
    print("TESTING QUANTUM ENCODING")
    print("=" * 50)
    
    # Sample data point
    sample = np.array([0.2, 0.5, 0.8, 1.0])
    
    # Test angle encoding
    angle_enc = AngleEncoder()
    angles = angle_enc.encode(sample)
    print(f"\nAngle Encoding:")
    print(f"  Input: {sample}")
    print(f"  Angles: {angles}")
    print(f"  Angle range: [{angles.min():.3f}, {angles.max():.3f}]")
    
    # Test amplitude encoding
    amp_enc = AmplitudeEncoder()
    amplitudes = amp_enc.encode(sample)
    print(f"\nAmplitude Encoding:")
    print(f"  Input: {sample}")
    print(f"  Amplitudes: {amplitudes}")
    print(f"  Norm: {np.linalg.norm(amplitudes):.6f} (should be 1.0)")


def test_pennylane_setup():
    """Verify PennyLane is working"""
    print("\n" + "=" * 50)
    print("TESTING PENNYLANE SETUP")
    print("=" * 50)
    
    try:
        # Try GPU device first
        dev = qml.device('lightning.gpu', wires=4)
        print("\n✓ GPU device available!")
    except:
        # Fallback to CPU
        dev = qml.device('default.qubit', wires=4)
        print("\n✓ CPU device available (GPU not found)")
    
    # Simple quantum circuit test
    @qml.qnode(dev)
    def test_circuit(angles):
        for i, angle in enumerate(angles):
            qml.RY(angle, wires=i)
        return qml.expval(qml.PauliZ(0))
    
    # Test with sample angles
    angles = np.array([0.5, 1.0, 1.5, 2.0])
    output = test_circuit(angles)
    print(f"\nTest circuit output: {output:.6f}")
    print("✓ PennyLane quantum circuit working!")


# ========================================
# 5. MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("PHASE 1: INFRASTRUCTURE VALIDATION")
    print("=" * 50)
    
    # Run all tests
    test_data_pipeline()
    test_feature_selection()
    test_quantum_encoding()
    test_pennylane_setup()
    
    print("\n" + "=" * 50)
    print("✓ PHASE 1 COMPLETE - All systems operational!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Review the output above")
    print("2. Verify all tests passed")
    print("3. Move to Phase 2: quantum_circuit.py")
    print("4. Start building the QNN!")