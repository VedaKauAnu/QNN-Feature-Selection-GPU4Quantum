"""
Noisy 10-Qubit Adaptive QNN Experiment
Demonstrates performance with realistic NISQ error rates (1-1.5%)

This is the main experiment demonstrating your adaptive approach
works under realistic noise conditions.
"""

import numpy as np
import pennylane as qml
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
from collections import defaultdict
import warnings
import json
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION WITH REALISTIC NISQ NOISE
# ============================================================================

class NoisyConfig:
    """Experiment configuration with realistic NISQ noise parameters"""
    
    # Quantum system
    N_QUBITS = 10
    N_LAYERS = 2
    
    # Training
    LEARNING_RATE = 0.01
    N_EPOCHS = 30  # Good data collection
    BATCH_SIZE = 16
    
    # Realistic NISQ error rates (based on IBM/Google hardware)
    NOISE_PARAMS = {
        'single_qubit_error': 0.01,      # 1% error per single-qubit gate
        'two_qubit_error': 0.015,        # 1.5% error per two-qubit gate
        'measurement_error': 0.02,       # 2% measurement error
        'description': 'Realistic NISQ (IBM Quantum, Google Sycamore)'
    }
    
    # Adaptive strategy
    STRATEGY = 'ucb'
    UCB_C = 2.0
    
    # Dataset
    DATASET = 'wine'
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    
    # Reporting
    VERBOSE = True
    PLOT_RESULTS = True
    SAVE_RESULTS = True

config = NoisyConfig()

# ============================================================================
# NOISY QUANTUM DEVICE
# ============================================================================

class NoisyDevice:
    """Quantum device with realistic NISQ noise"""
    
    def __init__(self, n_qubits, noise_params):
        self.n_qubits = n_qubits
        self.noise_params = noise_params
        
        # Use mixed state simulator for noise
        try:
            self.dev = qml.device('default.mixed', wires=n_qubits)
            self.noise_enabled = True
            print(f"✓ Noise simulation enabled: {noise_params['description']}")
            print(f"  Single-qubit error: {noise_params['single_qubit_error']*100}%")
            print(f"  Two-qubit error: {noise_params['two_qubit_error']*100}%")
            print(f"  Measurement error: {noise_params['measurement_error']*100}%")
        except:
            # Fallback to noiseless if mixed state not available
            self.dev = qml.device('default.qubit', wires=n_qubits)
            self.noise_enabled = False
            print("⚠️  Warning: Mixed state device not available, using noiseless simulation")
    
    def apply_single_qubit_noise(self, wires):
        """Apply noise after single-qubit gates"""
        if self.noise_enabled:
            for wire in (wires if isinstance(wires, list) else [wires]):
                qml.DepolarizingChannel(self.noise_params['single_qubit_error'], wires=wire)
    
    def apply_two_qubit_noise(self, wires):
        """Apply noise after two-qubit gates"""
        if self.noise_enabled:
            # Two-qubit gates have higher error rates
            for wire in wires:
                qml.DepolarizingChannel(self.noise_params['two_qubit_error'], wires=wire)
    
    def apply_measurement_noise(self, wires):
        """Apply measurement errors"""
        if self.noise_enabled:
            for wire in (wires if isinstance(wires, list) else [wires]):
                qml.BitFlip(self.noise_params['measurement_error'], wires=wire)

# ============================================================================
# NOISY QUANTUM NEURAL NETWORK
# ============================================================================

class NoisyQuantumNeuralNetwork:
    """QNN with realistic NISQ noise"""
    
    def __init__(self, n_qubits, n_layers, encoding_method='angle', noise_params=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding_method = encoding_method
        
        # Create noisy device
        if noise_params is None:
            noise_params = config.NOISE_PARAMS
        
        self.device_manager = NoisyDevice(n_qubits, noise_params)
        self.dev = self.device_manager.dev
        
        # Initialize parameters
        self.n_params = n_layers * n_qubits * 3
        self.params = np.random.randn(self.n_params) * 0.1
        
        # Create QNode
        self.qnode = qml.QNode(self._circuit, self.dev)
        
        # Track statistics
        self.gate_count = 0
        self.circuit_depth = 0
    
    def _circuit(self, features, params):
        """Noisy quantum circuit"""
        wires = range(self.n_qubits)
        gate_count = 0
        
        # ────────────────────────────────────────────────────────────────
        # Data Encoding
        # ────────────────────────────────────────────────────────────────
        if self.encoding_method == 'angle':
            for i, feature in enumerate(features[:self.n_qubits]):
                qml.RY(feature * np.pi, wires=i)
                self.device_manager.apply_single_qubit_noise(i)
                gate_count += 1
        elif self.encoding_method == 'amplitude':
            # Amplitude encoding
            norm = np.linalg.norm(features)
            if norm > 0:
                normalized = features / norm
            else:
                normalized = features
            
            n_amplitudes = 2 ** self.n_qubits
            if len(normalized) < n_amplitudes:
                padded = np.zeros(n_amplitudes)
                padded[:len(normalized)] = normalized
                normalized = padded
            else:
                normalized = normalized[:n_amplitudes]
            
            normalized = normalized / (np.linalg.norm(normalized) + 1e-10)
            qml.AmplitudeEmbedding(normalized, wires=wires, normalize=True)
            
            # Amplitude embedding involves multiple gates internally
            gate_count += self.n_qubits * 2
            
            # Apply noise after amplitude encoding
            for i in range(self.n_qubits):
                self.device_manager.apply_single_qubit_noise(i)
        
        # ────────────────────────────────────────────────────────────────
        # Parameterized Layers
        # ────────────────────────────────────────────────────────────────
        for layer in range(self.n_layers):
            # Single-qubit rotations
            for i in range(self.n_qubits):
                idx = layer * self.n_qubits * 3 + i * 3
                
                qml.RX(params[idx], wires=i)
                self.device_manager.apply_single_qubit_noise(i)
                gate_count += 1
                
                qml.RY(params[idx + 1], wires=i)
                self.device_manager.apply_single_qubit_noise(i)
                gate_count += 1
                
                qml.RZ(params[idx + 2], wires=i)
                self.device_manager.apply_single_qubit_noise(i)
                gate_count += 1
            
            # Entangling layer
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                # Two-qubit gates have higher noise
                self.device_manager.apply_two_qubit_noise([i, (i + 1) % self.n_qubits])
                gate_count += 1
        
        # Store gate count
        self.gate_count = gate_count
        self.circuit_depth = self.n_layers * 4 + 1  # 3 rotations + 1 CNOT per layer + encoding
        
        # ────────────────────────────────────────────────────────────────
        # Measurement with noise
        # ────────────────────────────────────────────────────────────────
        self.device_manager.apply_measurement_noise(0)
        return qml.expval(qml.PauliZ(0))
    
    def predict_single(self, features):
        output = self.qnode(features, self.params)
        return 1 if output > 0 else 0
    
    def predict_batch(self, X):
        return np.array([self.predict_single(x) for x in X])
    
    def compute_loss(self, X, y):
        predictions = self.predict_batch(X)
        return np.mean((predictions - y) ** 2)
    
    def train_step(self, X, y, learning_rate):
        """Training step with noisy gradients"""
        shift = np.pi / 2
        gradients = np.zeros_like(self.params)
        
        # Compute gradients (affected by noise!)
        for i in range(len(self.params)):
            self.params[i] += shift
            forward = self.compute_loss(X, y)
            self.params[i] -= 2 * shift
            backward = self.compute_loss(X, y)
            self.params[i] += shift
            gradients[i] = (forward - backward) / 2
        
        # Update parameters
        self.params -= learning_rate * gradients
        
        return self.compute_loss(X, y)

# ============================================================================
# FEATURE SELECTION (Same as before)
# ============================================================================

class FeatureSelector:
    """Feature selection methods with padding"""
    
    @staticmethod
    def _pad_features(X, target_features):
        if X.shape[1] < target_features:
            padding = np.zeros((X.shape[0], target_features - X.shape[1]))
            return np.hstack([X, padding])
        return X
    
    @staticmethod
    def pca(X_train, X_test, n_features):
        n_components = min(n_features, X_train.shape[1])
        pca = PCA(n_components=n_components)
        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)
        
        X_train_reduced = FeatureSelector._pad_features(X_train_reduced, n_features)
        X_test_reduced = FeatureSelector._pad_features(X_test_reduced, n_features)
        
        return X_train_reduced, X_test_reduced
    
    @staticmethod
    def correlation(X_train, X_test, y_train, n_features):
        n_available = X_train.shape[1]
        n_select = min(n_features, n_available)
        
        correlations = np.abs([np.corrcoef(X_train[:, i], y_train)[0, 1] 
                              for i in range(n_available)])
        top_indices = np.argsort(correlations)[-n_select:]
        
        X_train_selected = X_train[:, top_indices]
        X_test_selected = X_test[:, top_indices]
        
        X_train_selected = FeatureSelector._pad_features(X_train_selected, n_features)
        X_test_selected = FeatureSelector._pad_features(X_test_selected, n_features)
        
        return X_train_selected, X_test_selected
    
    @staticmethod
    def mutual_info(X_train, X_test, y_train, n_features):
        n_available = X_train.shape[1]
        n_select = min(n_features, n_available)
        
        mi_scores = mutual_info_classif(X_train, y_train, random_state=config.RANDOM_STATE)
        top_indices = np.argsort(mi_scores)[-n_select:]
        
        X_train_selected = X_train[:, top_indices]
        X_test_selected = X_test[:, top_indices]
        
        X_train_selected = FeatureSelector._pad_features(X_train_selected, n_features)
        X_test_selected = FeatureSelector._pad_features(X_test_selected, n_features)
        
        return X_train_selected, X_test_selected
    
    @staticmethod
    def random_selection(X_train, X_test, n_features):
        n_available = X_train.shape[1]
        n_select = min(n_features, n_available)
        
        np.random.seed(config.RANDOM_STATE)
        indices = np.random.choice(n_available, n_select, replace=False)
        
        X_train_selected = X_train[:, indices]
        X_test_selected = X_test[:, indices]
        
        X_train_selected = FeatureSelector._pad_features(X_train_selected, n_features)
        X_test_selected = FeatureSelector._pad_features(X_test_selected, n_features)
        
        return X_train_selected, X_test_selected

# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    """Load and preprocess datasets"""
    
    @staticmethod
    def load_dataset(name='wine'):
        datasets = {
            'iris': load_iris,
            'wine': load_wine,
            'breast_cancer': load_breast_cancer
        }
        
        if name not in datasets:
            raise ValueError(f"Dataset {name} not recognized")
        
        data = datasets[name]()
        X, y = data.data, data.target
        
        # Binary classification
        if name == 'iris':
            mask = y != 2
            X, y = X[mask], y[mask]
        elif name == 'wine':
            y = (y == 0).astype(int)
        
        return X, y, data.feature_names
    
    @staticmethod
    def preprocess(X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

# ============================================================================
# ADAPTIVE CONTROLLER
# ============================================================================

class AdaptiveController:
    """UCB-based adaptive configuration selection"""
    
    def __init__(self, configurations, strategy='ucb'):
        self.configurations = configurations
        self.strategy = strategy
        self.rewards = defaultdict(list)
        self.selection_counts = defaultdict(int)
        self.total_selections = 0
        self.selection_history = []
        self.reward_history = []
    
    def select_configuration(self):
        if self.strategy == 'ucb':
            return self._ucb()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _ucb(self):
        # Try each configuration at least once
        if self.total_selections < len(self.configurations):
            for cfg in self.configurations:
                if cfg not in self.rewards:
                    return cfg
        
        # UCB selection
        ucb_values = {}
        for cfg in self.configurations:
            if cfg not in self.rewards:
                ucb_values[cfg] = float('inf')
            else:
                mean_reward = np.mean(self.rewards[cfg])
                exploration_bonus = config.UCB_C * np.sqrt(
                    np.log(self.total_selections) / self.selection_counts[cfg]
                )
                ucb_values[cfg] = mean_reward + exploration_bonus
        
        return max(ucb_values, key=ucb_values.get)
    
    def update(self, configuration, reward):
        self.rewards[configuration].append(reward)
        self.selection_counts[configuration] += 1
        self.total_selections += 1
        self.selection_history.append(configuration)
        self.reward_history.append(reward)
    
    def get_best_configuration(self):
        if not self.rewards:
            return None
        mean_rewards = {cfg: np.mean(self.rewards[cfg]) for cfg in self.rewards}
        return max(mean_rewards, key=mean_rewards.get)
    
    def get_statistics(self):
        stats = []
        for cfg in self.configurations:
            if cfg in self.rewards:
                rewards = self.rewards[cfg]
                stats.append({
                    'configuration': cfg,
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'count': len(rewards)
                })
        return sorted(stats, key=lambda x: x['mean_reward'], reverse=True)

# ============================================================================
# MAIN TRAINER
# ============================================================================

class NoisyAdaptiveQNNTrainer:
    """Main training pipeline with noise"""
    
    def __init__(self):
        print("="*70)
        print("NOISY ADAPTIVE QNN EXPERIMENT")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Qubits: {config.N_QUBITS}")
        print(f"  Layers: {config.N_LAYERS}")
        print(f"  Epochs: {config.N_EPOCHS}")
        print(f"  Dataset: {config.DATASET}")
        print(f"  Noise: {config.NOISE_PARAMS['description']}")
        print()
        
        # Load data
        X, y, feature_names = DataLoader.load_dataset(config.DATASET)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        X_train, X_test = DataLoader.preprocess(X_train, X_test)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Data loaded:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Test: {len(X_test)} samples")
        print(f"  Original features: {X_train.shape[1]}")
        print(f"  Target features: {config.N_QUBITS}")
        print()
        
        # Define configurations
        feature_methods = ['PCA', 'Correlation', 'MutualInfo', 'Random']
        encoding_methods = ['angle', 'amplitude']
        self.configurations = [
            f"{fm}+{em}" for fm in feature_methods for em in encoding_methods
        ]
        
        # Initialize controller
        self.controller = AdaptiveController(self.configurations, config.STRATEGY)
        
        # Tracking
        self.training_history = {
            'train_acc': [],
            'val_acc': [],
            'config': [],
            'epoch_time': [],
            'gate_count': [],
            'circuit_depth': []
        }
        
        self.start_time = time.time()
    
    def apply_feature_selection(self, method):
        if method == 'PCA':
            return FeatureSelector.pca(self.X_train, self.X_test, config.N_QUBITS)
        elif method == 'Correlation':
            return FeatureSelector.correlation(
                self.X_train, self.X_test, self.y_train, config.N_QUBITS
            )
        elif method == 'MutualInfo':
            return FeatureSelector.mutual_info(
                self.X_train, self.X_test, self.y_train, config.N_QUBITS
            )
        elif method == 'Random':
            return FeatureSelector.random_selection(
                self.X_train, self.X_test, config.N_QUBITS
            )
    
    def train_epoch(self, qnn, X_train, X_test, y_train, y_test):
        epoch_start = time.time()
        
        n_samples = len(X_train)
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, config.BATCH_SIZE):
            batch_indices = indices[i:i + config.BATCH_SIZE]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            qnn.train_step(X_batch, y_batch, config.LEARNING_RATE)
        
        # Evaluation
        train_pred = qnn.predict_batch(X_train)
        train_acc = np.mean(train_pred == y_train)
        
        val_pred = qnn.predict_batch(X_test)
        val_acc = np.mean(val_pred == y_test)
        
        epoch_time = time.time() - epoch_start
        
        return train_acc, val_acc, epoch_time, qnn.gate_count, qnn.circuit_depth
    
    def run(self):
        print("="*70)
        print("STARTING TRAINING")
        print("="*70)
        print()
        
        for epoch in range(config.N_EPOCHS):
            # Select configuration
            config_name = self.controller.select_configuration()
            feature_method, encoding_method = config_name.split('+')
            
            # Apply feature selection
            X_train_reduced, X_test_reduced = self.apply_feature_selection(feature_method)
            
            # Create QNN with noise
            qnn = NoisyQuantumNeuralNetwork(
                config.N_QUBITS,
                config.N_LAYERS,
                encoding_method,
                config.NOISE_PARAMS
            )
            
            # Train
            train_acc, val_acc, epoch_time, gate_count, circuit_depth = self.train_epoch(
                qnn, X_train_reduced, X_test_reduced, self.y_train, self.y_test
            )
            
            # Update controller
            self.controller.update(config_name, val_acc)
            
            # Track history
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['config'].append(config_name)
            self.training_history['epoch_time'].append(epoch_time)
            self.training_history['gate_count'].append(gate_count)
            self.training_history['circuit_depth'].append(circuit_depth)
            
            if config.VERBOSE:
                best_config = self.controller.get_best_configuration()
                best_reward = np.mean(self.controller.rewards[best_config]) if best_config else 0
                
                print(f"Epoch {epoch+1:3d}/{config.N_EPOCHS} | "
                      f"Config: {config_name:25s} | "
                      f"Train: {train_acc:.3f} | "
                      f"Val: {val_acc:.3f} | "
                      f"Best: {best_config:20s} ({best_reward:.3f}) | "
                      f"Time: {epoch_time:.1f}s")
        
        self.total_time = time.time() - self.start_time
        self.generate_report()
        
        if config.SAVE_RESULTS:
            self.save_results()
    
    def generate_report(self):
        print(f"\n{'='*70}")
        print("EXPERIMENT REPORT - NOISY 10-QUBIT ADAPTIVE QNN")
        print(f"{'='*70}")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\nSYSTEM CONFIGURATION")
        print("-" * 70)
        print(f"Qubits: {config.N_QUBITS}")
        print(f"Layers: {config.N_LAYERS}")
        print(f"Noise Model: {config.NOISE_PARAMS['description']}")
        print(f"  Single-qubit error: {config.NOISE_PARAMS['single_qubit_error']*100}%")
        print(f"  Two-qubit error: {config.NOISE_PARAMS['two_qubit_error']*100}%")
        print(f"  Measurement error: {config.NOISE_PARAMS['measurement_error']*100}%")
        
        print(f"\nTRAINING SUMMARY")
        print("-" * 70)
        print(f"Total Epochs: {config.N_EPOCHS}")
        print(f"Total Time: {self.total_time:.2f}s")
        print(f"Avg Time/Epoch: {self.total_time/config.N_EPOCHS:.2f}s")
        
        best_config = self.controller.get_best_configuration()
        best_rewards = self.controller.rewards[best_config]
        
        print(f"\nBEST CONFIGURATION")
        print("-" * 70)
        print(f"Configuration: {best_config}")
        print(f"Mean Reward: {np.mean(best_rewards):.4f}")
        print(f"Std Dev: {np.std(best_rewards):.4f}")
        print(f"Times Selected: {len(best_rewards)}")
        
        print(f"\nALL CONFIGURATIONS PERFORMANCE")
        print("-" * 70)
        print(f"{'Configuration':<25s} {'Mean':<10s} {'Std':<10s} {'Count':<8s}")
        print("-" * 70)
        
        stats = self.controller.get_statistics()
        for stat in stats:
            print(f"{stat['configuration']:<25s} "
                  f"{stat['mean_reward']:<10.4f} "
                  f"{stat['std_reward']:<10.4f} "
                  f"{stat['count']:<8d}")
        
        print(f"\n{'='*70}\n")
        
        if config.PLOT_RESULTS:
            self.plot_results()
    
    def plot_results(self):
        """Generate comprehensive visualizations"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Training Progress
        ax = fig.add_subplot(gs[0, :2])
        epochs = range(1, len(self.training_history['train_acc']) + 1)
        ax.plot(epochs, self.training_history['train_acc'], 
               label='Train', linewidth=2, alpha=0.7)
        ax.plot(epochs, self.training_history['val_acc'], 
               label='Validation', linewidth=2, alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Training Progress with Noise', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        # Plot 2: Configuration Frequency
        ax = fig.add_subplot(gs[0, 2])
        config_counts = {cfg: self.controller.selection_counts[cfg] 
                        for cfg in self.configurations}
        configs = list(config_counts.keys())
        counts = list(config_counts.values())
        
        y_pos = np.arange(len(configs))
        ax.barh(y_pos, counts)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(configs, fontsize=9)
        ax.set_xlabel('Times Selected', fontsize=11)
        ax.set_title('Config Selection Frequency', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Plot 3: Reward Over Time
        ax = fig.add_subplot(gs[1, :2])
        ax.plot(self.controller.reward_history, linewidth=2, color='green', alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Reward (Val Accuracy)', fontsize=12)
        ax.set_title('Reward Over Time', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Plot 4: Configuration Timeline
        ax = fig.add_subplot(gs[1, 2])
        config_to_y = {cfg: i for i, cfg in enumerate(self.configurations)}
        
        for i, (cfg, reward) in enumerate(zip(self.controller.selection_history, 
                                              self.controller.reward_history)):
            ax.scatter(i, config_to_y[cfg], c=[reward], vmin=0, vmax=1, 
                      cmap='viridis', s=80, edgecolors='black', linewidth=0.5)
        
        ax.set_yticks(range(len(self.configurations)))
        ax.set_yticklabels(self.configurations, fontsize=8)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_title('Config Timeline', fontsize=12, fontweight='bold')
        
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Reward', fontsize=10)
        
        # Plot 5: Performance Box Plot
        ax = fig.add_subplot(gs[2, :2])
        
        data_for_plot = []
        labels_for_plot = []
        for cfg in self.configurations:
            if cfg in self.controller.rewards and len(self.controller.rewards[cfg]) > 0:
                data_for_plot.append(self.controller.rewards[cfg])
                labels_for_plot.append(cfg)
        
        bp = ax.boxplot(data_for_plot, labels=labels_for_plot, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel('Validation Accuracy', fontsize=12)
        ax.set_title('Performance Distribution by Configuration', 
                    fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 6: Circuit Statistics
        ax = fig.add_subplot(gs[2, 2])
        
        # Average gate count by configuration
        avg_gates = {}
        for i, cfg in enumerate(self.training_history['config']):
            if cfg not in avg_gates:
                avg_gates[cfg] = []
            avg_gates[cfg].append(self.training_history['gate_count'][i])
        
        configs = list(avg_gates.keys())
        avg_counts = [np.mean(avg_gates[cfg]) for cfg in configs]
        
        y_pos = np.arange(len(configs))
        ax.barh(y_pos, avg_counts, color='coral')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(configs, fontsize=8)
        ax.set_xlabel('Avg Gate Count', fontsize=11)
        ax.set_title('Circuit Complexity', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle(f'Noisy 10-Qubit Adaptive QNN Results\n'
                    f'{config.NOISE_PARAMS["description"]}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'noisy_10q_results_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Results saved: {filename}")
        plt.show()
    
    def save_results(self):
        """Save results to JSON for analysis"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results = {
            'timestamp': timestamp,
            'config': {
                'n_qubits': config.N_QUBITS,
                'n_layers': config.N_LAYERS,
                'n_epochs': config.N_EPOCHS,
                'dataset': config.DATASET,
                'noise_params': config.NOISE_PARAMS
            },
            'training_history': self.training_history,
            'controller_rewards': {k: v for k, v in self.controller.rewards.items()},
            'best_configuration': self.controller.get_best_configuration(),
            'statistics': self.controller.get_statistics(),
            'total_time': self.total_time
        }
        
        filename = f'noisy_10q_results_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        print(f"✓ Results saved: {filename}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  NOISY 10-QUBIT ADAPTIVE QNN EXPERIMENT                          ║
    ║                                                                   ║
    ║  This experiment demonstrates your adaptive approach works       ║
    ║  under realistic NISQ noise conditions (1-1.5% error rates)      ║
    ║                                                                   ║
    ║  Key Features:                                                    ║
    ║  • 10 qubits (practical NISQ scale)                             ║
    ║  • Realistic noise model (IBM/Google hardware)                   ║
    ║  • 30 epochs (robust data collection)                            ║
    ║  • Adaptive UCB selection                                        ║
    ║  • Comprehensive performance tracking                            ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    trainer = NoisyAdaptiveQNNTrainer()
    trainer.run()
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("\n✓ Noisy 10-qubit experiment finished successfully!")
    print(f"✓ Results saved with timestamp")
    print(f"✓ Ready for scalability analysis")