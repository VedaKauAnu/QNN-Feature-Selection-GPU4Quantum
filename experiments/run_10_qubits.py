"""
Adaptive Quantum Neural Network with 10 Qubits
Feature Selection + Encoding Method Optimization
"""

import numpy as np
import pennylane as qml
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# CONFIGURATION

class Config:
    """Experiment configuration"""
    N_QUBITS = 10
    N_LAYERS = 2
    LEARNING_RATE = 0.01
    N_EPOCHS = 30
    BATCH_SIZE = 16
    
    # Adaptive strategy parameters
    STRATEGY = 'ucb'  # Options: 'ucb', 'epsilon_greedy', 'round_robin'
    UCB_C = 2.0  # Exploration parameter for UCB
    EPSILON = 0.2  # For epsilon-greedy
    
    # Dataset
    DATASET = 'wine'  # Options: 'iris' (4 features), 'wine' (13 features), 'breast_cancer' (30 features)
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    
    # Device
    DEVICE_NAME = 'default.qubit'  # CPU-based quantum simulator
    
    # Reporting
    VERBOSE = True
    PLOT_RESULTS = True

config = Config()


# DATA LOADING AND PREPROCESSING

class DataLoader:
    """Load and preprocess datasets"""
    
    @staticmethod
    def load_dataset(name='iris'):
        """Load specified dataset"""
        datasets = {
            'iris': load_iris,
            'wine': load_wine,
            'breast_cancer': load_breast_cancer
        }
        
        if name not in datasets:
            raise ValueError(f"Dataset {name} not recognized. Choose from {list(datasets.keys())}")
        
        data = datasets[name]()
        X, y = data.data, data.target
        
        # Convert to binary classification for simplicity
        if name == 'iris':
            # Setosa vs rest
            mask = y != 2
            X, y = X[mask], y[mask]
        elif name == 'wine':
            # Class 0 vs rest
            y = (y == 0).astype(int)
        elif name == 'breast_cancer':
            # Already binary
            pass
        
        return X, y, data.feature_names
    
    @staticmethod
    def preprocess(X_train, X_test, y_train, y_test):
        """Standardize features"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test


# FEATURE SELECTION METHODS

class FeatureSelector:
    """Various feature selection methods"""
    
    @staticmethod
    def _pad_features(X, target_features):
        """Pad features with zeros if needed"""
        if X.shape[1] < target_features:
            padding = np.zeros((X.shape[0], target_features - X.shape[1]))
            return np.hstack([X, padding])
        return X
    
    @staticmethod
    def pca(X_train, X_test, n_features):
        """PCA dimensionality reduction"""
        # Use minimum of requested features and available features
        n_components = min(n_features, X_train.shape[1])
        pca = PCA(n_components=n_components)
        X_train_reduced = pca.fit_transform(X_train)
        X_test_reduced = pca.transform(X_test)
        
        # Pad with zeros if needed
        X_train_reduced = FeatureSelector._pad_features(X_train_reduced, n_features)
        X_test_reduced = FeatureSelector._pad_features(X_test_reduced, n_features)
        
        return X_train_reduced, X_test_reduced
    
    @staticmethod
    def correlation(X_train, X_test, y_train, n_features):
        """Select features with highest correlation to target"""
        n_available = X_train.shape[1]
        n_select = min(n_features, n_available)
        
        correlations = np.abs([np.corrcoef(X_train[:, i], y_train)[0, 1] 
                              for i in range(n_available)])
        top_indices = np.argsort(correlations)[-n_select:]
        
        X_train_selected = X_train[:, top_indices]
        X_test_selected = X_test[:, top_indices]
        
        # Pad with zeros if needed
        X_train_selected = FeatureSelector._pad_features(X_train_selected, n_features)
        X_test_selected = FeatureSelector._pad_features(X_test_selected, n_features)
        
        return X_train_selected, X_test_selected
    
    @staticmethod
    def mutual_info(X_train, X_test, y_train, n_features):
        """Select features with highest mutual information"""
        n_available = X_train.shape[1]
        n_select = min(n_features, n_available)
        
        mi_scores = mutual_info_classif(X_train, y_train, random_state=config.RANDOM_STATE)
        top_indices = np.argsort(mi_scores)[-n_select:]
        
        X_train_selected = X_train[:, top_indices]
        X_test_selected = X_test[:, top_indices]
        
        # Pad with zeros if needed
        X_train_selected = FeatureSelector._pad_features(X_train_selected, n_features)
        X_test_selected = FeatureSelector._pad_features(X_test_selected, n_features)
        
        return X_train_selected, X_test_selected
    
    @staticmethod
    def random_selection(X_train, X_test, n_features):
        """Random feature selection (baseline)"""
        n_available = X_train.shape[1]
        n_select = min(n_features, n_available)
        
        np.random.seed(config.RANDOM_STATE)
        indices = np.random.choice(n_available, n_select, replace=False)
        
        X_train_selected = X_train[:, indices]
        X_test_selected = X_test[:, indices]
        
        # Pad with zeros if needed
        X_train_selected = FeatureSelector._pad_features(X_train_selected, n_features)
        X_test_selected = FeatureSelector._pad_features(X_test_selected, n_features)
        
        return X_train_selected, X_test_selected


# QUANTUM ENCODING METHODS

class QuantumEncoder:
    """Encode classical data into quantum states"""
    
    @staticmethod
    def angle_encoding(features, wires):
        """Angle encoding: encode features as rotation angles"""
        for i, feature in enumerate(features[:len(wires)]):
            qml.RY(feature * np.pi, wires=wires[i])
    
    @staticmethod
    def amplitude_encoding(features, wires):
        """Amplitude encoding: encode as amplitudes of quantum state"""
        # Normalize features to create valid quantum state
        norm = np.linalg.norm(features)
        if norm > 0:
            normalized = features / norm
        else:
            normalized = features
        
        # Pad to 2^n_qubits if necessary
        n_amplitudes = 2 ** len(wires)
        if len(normalized) < n_amplitudes:
            padded = np.zeros(n_amplitudes)
            padded[:len(normalized)] = normalized
            normalized = padded
        else:
            normalized = normalized[:n_amplitudes]
        
        # Normalize again to ensure valid quantum state
        normalized = normalized / np.linalg.norm(normalized)
        
        qml.AmplitudeEmbedding(normalized, wires=wires, normalize=True)


# QUANTUM NEURAL NETWORK

class QuantumNeuralNetwork:
    """Parameterized quantum circuit for classification"""
    
    def __init__(self, n_qubits, n_layers, encoding_method='angle'):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.encoding_method = encoding_method
        
        # Initialize quantum device
        try:
            self.dev = qml.device(config.DEVICE_NAME, wires=n_qubits)
        except:
            print(f"Warning: {config.DEVICE_NAME} not available, using default.qubit")
            self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Initialize parameters
        self.n_params = n_layers * n_qubits * 3  # 3 rotations per qubit per layer
        self.params = np.random.randn(self.n_params) * 0.1
        
        # Create QNode
        self.qnode = qml.QNode(self._circuit, self.dev)
    
    def _circuit(self, features, params):
        """Quantum circuit architecture"""
        wires = range(self.n_qubits)
        
        # Data encoding
        if self.encoding_method == 'angle':
            QuantumEncoder.angle_encoding(features, wires)
        elif self.encoding_method == 'amplitude':
            QuantumEncoder.amplitude_encoding(features, wires)
        else:
            raise ValueError(f"Unknown encoding method: {self.encoding_method}")
        
        # Parameterized layers
        for layer in range(self.n_layers):
            # Single-qubit rotations
            for i in range(self.n_qubits):
                idx = layer * self.n_qubits * 3 + i * 3
                qml.RX(params[idx], wires=i)
                qml.RY(params[idx + 1], wires=i)
                qml.RZ(params[idx + 2], wires=i)
            
            # Entangling layer (circular connectivity)
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
        
        # Measurement
        return qml.expval(qml.PauliZ(0))
    
    def predict_single(self, features):
        """Predict for a single sample"""
        output = self.qnode(features, self.params)
        return 1 if output > 0 else 0
    
    def predict_batch(self, X):
        """Predict for multiple samples"""
        return np.array([self.predict_single(x) for x in X])
    
    def compute_loss(self, X, y):
        """Compute mean squared error loss"""
        predictions = self.predict_batch(X)
        return np.mean((predictions - y) ** 2)
    
    def train_step(self, X, y, learning_rate):
        """Single training step with parameter shift rule"""
        shift = np.pi / 2
        gradients = np.zeros_like(self.params)
        
        # Compute gradients using parameter shift rule
        for i in range(len(self.params)):
            # Shift parameter forward
            self.params[i] += shift
            forward = self.compute_loss(X, y)
            
            # Shift parameter backward
            self.params[i] -= 2 * shift
            backward = self.compute_loss(X, y)
            
            # Restore parameter and compute gradient
            self.params[i] += shift
            gradients[i] = (forward - backward) / 2
        
        # Update parameters
        self.params -= learning_rate * gradients
        
        return self.compute_loss(X, y)


# ADAPTIVE CONTROLLER

class AdaptiveController:
    """Manages configuration selection using various strategies"""
    
    def __init__(self, configurations, strategy='ucb'):
        self.configurations = configurations
        self.strategy = strategy
        
        # Track performance
        self.rewards = defaultdict(list)
        self.selection_counts = defaultdict(int)
        self.total_selections = 0
        
        # History
        self.selection_history = []
        self.reward_history = []
    
    def select_configuration(self):
        """Select next configuration to try"""
        if self.strategy == 'round_robin':
            return self._round_robin()
        elif self.strategy == 'epsilon_greedy':
            return self._epsilon_greedy()
        elif self.strategy == 'ucb':
            return self._ucb()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _round_robin(self):
        """Round-robin selection"""
        idx = self.total_selections % len(self.configurations)
        return self.configurations[idx]
    
    def _epsilon_greedy(self):
        """Epsilon-greedy selection"""
        if np.random.random() < config.EPSILON:
            # Explore
            return np.random.choice(self.configurations)
        else:
            # Exploit
            if not self.rewards:
                return np.random.choice(self.configurations)
            
            mean_rewards = {cfg: np.mean(self.rewards[cfg]) 
                          for cfg in self.configurations if cfg in self.rewards}
            if not mean_rewards:
                return np.random.choice(self.configurations)
            
            return max(mean_rewards, key=mean_rewards.get)
    
    def _ucb(self):
        """Upper Confidence Bound selection"""
        if self.total_selections < len(self.configurations):
            # Try each configuration at least once
            for cfg in self.configurations:
                if cfg not in self.rewards:
                    return cfg
        
        # Compute UCB values
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
        """Update performance tracking"""
        self.rewards[configuration].append(reward)
        self.selection_counts[configuration] += 1
        self.total_selections += 1
        
        self.selection_history.append(configuration)
        self.reward_history.append(reward)
    
    def get_best_configuration(self):
        """Get configuration with highest mean reward"""
        if not self.rewards:
            return None
        
        mean_rewards = {cfg: np.mean(self.rewards[cfg]) for cfg in self.rewards}
        return max(mean_rewards, key=mean_rewards.get)
    
    def get_statistics(self):
        """Get performance statistics"""
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


# TRAINING PIPELINE

class AdaptiveQNNTrainer:
    """Main training pipeline"""
    
    def __init__(self):
        # Load and preprocess data
        print("Loading dataset...")
        X, y, feature_names = DataLoader.load_dataset(config.DATASET)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        X_train, X_test, y_train, y_test = DataLoader.preprocess(
            X_train, X_test, y_train, y_test
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Dataset: {config.DATASET}")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Original features: {X_train.shape[1]}, Reduced to: {config.N_QUBITS}")
        
        # Define all configurations
        feature_methods = ['PCA', 'Correlation', 'MutualInfo', 'Random']
        encoding_methods = ['angle', 'amplitude']
        
        self.configurations = [
            f"{fm}+{em}" for fm in feature_methods for em in encoding_methods
        ]
        
        # Initialize adaptive controller
        self.controller = AdaptiveController(self.configurations, config.STRATEGY)
        
        # Tracking
        self.training_history = {
            'train_acc': [],
            'val_acc': [],
            'config': [],
            'epoch_time': []
        }
        
        self.start_time = time.time()
    
    def apply_feature_selection(self, method):
        """Apply specified feature selection method"""
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
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
    
    def train_epoch(self, qnn, X_train, X_test, y_train, y_test):
        """Train for one epoch"""
        epoch_start = time.time()
        
        # Training
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
        
        return train_acc, val_acc, epoch_time
    
    def run(self):
        """Run full adaptive training experiment"""
        print(f"\n{'='*70}")
        print(f"STARTING ADAPTIVE QNN TRAINING")
        print(f"{'='*70}")
        print(f"Configuration: {config.N_QUBITS} qubits, {config.N_LAYERS} layers")
        print(f"Strategy: {config.STRATEGY}")
        print(f"Epochs: {config.N_EPOCHS}")
        print(f"{'='*70}\n")
        
        for epoch in range(config.N_EPOCHS):
            # Select configuration
            config_name = self.controller.select_configuration()
            feature_method, encoding_method = config_name.split('+')
            
            # Apply feature selection
            X_train_reduced, X_test_reduced = self.apply_feature_selection(feature_method)
            
            # Create QNN with selected encoding
            qnn = QuantumNeuralNetwork(
                config.N_QUBITS, config.N_LAYERS, encoding_method
            )
            
            # Train for one epoch
            train_acc, val_acc, epoch_time = self.train_epoch(
                qnn, X_train_reduced, X_test_reduced, self.y_train, self.y_test
            )
            
            # Update controller with reward (validation accuracy)
            self.controller.update(config_name, val_acc)
            
            # Track history
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['config'].append(config_name)
            self.training_history['epoch_time'].append(epoch_time)
            
            # Progress output
            if config.VERBOSE:
                print(f"Epoch {epoch+1:3d}/{config.N_EPOCHS} | "
                      f"Config: {config_name:25s} | "
                      f"Train: {train_acc:.4f} | "
                      f"Val: {val_acc:.4f} | "
                      f"Time: {epoch_time:.2f}s")
        
        self.total_time = time.time() - self.start_time
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive report"""
        print(f"\n{'='*70}")
        print("ADAPTIVE QNN RESEARCH REPORT")
        print(f"{'='*70}")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\nSYSTEM CONFIGURATION")
        print("-" * 70)
        print(f"Qubits: {config.N_QUBITS}")
        print(f"Layers: {config.N_LAYERS}")
        print(f"Learning Rate: {config.LEARNING_RATE}")
        print(f"Adaptation Strategy: {config.STRATEGY}")
        print(f"Dataset: {config.DATASET}")
        
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
        print(f"Times Selected: {len(best_rewards)}")
        print(f"Std Dev: {np.std(best_rewards):.4f}")
        
        print(f"\nALL CONFIGURATIONS PERFORMANCE")
        print("-" * 70)
        print(f"{'Configuration':<25s} {'Mean':<10s} {'Count':<8s} {'Std':<10s}")
        print("-" * 70)
        
        stats = self.controller.get_statistics()
        for stat in stats:
            print(f"{stat['configuration']:<25s} "
                  f"{stat['mean_reward']:<10.4f} "
                  f"{stat['count']:<8d} "
                  f"{stat['std_reward']:<10.4f}")
        
        print(f"\n{'='*70}\n")
        
        if config.PLOT_RESULTS:
            self.plot_results()
    
    def plot_results(self):
        """Generate visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Training Progress
        ax = axes[0, 0]
        epochs = range(1, len(self.training_history['train_acc']) + 1)
        ax.plot(epochs, self.training_history['train_acc'], label='Train', linewidth=2)
        ax.plot(epochs, self.training_history['val_acc'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: Configuration Selection Frequency
        ax = axes[0, 1]
        config_counts = {cfg: self.controller.selection_counts[cfg] 
                        for cfg in self.configurations}
        configs = list(config_counts.keys())
        counts = list(config_counts.values())
        
        ax.barh(configs, counts)
        ax.set_xlabel('Times Selected', fontsize=12)
        ax.set_ylabel('Configuration', fontsize=12)
        ax.set_title('Configuration Selection Frequency', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Plot 3: Reward Over Time
        ax = axes[1, 0]
        ax.plot(self.controller.reward_history, linewidth=2, color='green')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Reward (Val Accuracy)', fontsize=12)
        ax.set_title('Reward Over Time', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Plot 4: Configuration Selection Timeline
        ax = axes[1, 1]
        
        # Create mapping for configurations to y-axis
        config_to_y = {cfg: i for i, cfg in enumerate(self.configurations)}
        
        # Plot each selection as a scatter point colored by reward
        for i, (cfg, reward) in enumerate(zip(self.controller.selection_history, 
                                              self.controller.reward_history)):
            ax.scatter(i, config_to_y[cfg], c=[reward], vmin=0, vmax=1, 
                      cmap='viridis', s=100, edgecolors='black', linewidth=0.5)
        
        ax.set_yticks(range(len(self.configurations)))
        ax.set_yticklabels(self.configurations)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Configuration', fontsize=12)
        ax.set_title('Configuration Selection Timeline', fontsize=14, fontweight='bold')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Reward', fontsize=12)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'adaptive_qnn_results_{config.N_QUBITS}q_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Results saved to: {filename}")
        
        plt.show()


# MAIN EXECUTION

if __name__ == "__main__":
    # Run experiment
    trainer = AdaptiveQNNTrainer()
    trainer.run()
    
    print("\n✓ Experiment complete! Check the generated plots and report.")
    print(f"  Total configurations tested: {len(trainer.controller.configurations)}")
    print(f"  Best configuration: {trainer.controller.get_best_configuration()}")
    print(f"  Best accuracy: {np.mean(trainer.controller.rewards[trainer.controller.get_best_configuration()]):.4f}")