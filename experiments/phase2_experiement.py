"""
Phase 2 Experiment: Complete End-to-End QNN Pipeline
Combines Phase 1 (feature selection) with Phase 2 (QNN)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data_pipeline import DataLoader, PCASelector, CorrelationSelector
from src.quantum_encoding import AngleEncoder, AmplitudeEncoder
from src.quantum_circuit import QuantumNeuralNetwork


def experiment_full_pipeline():
    """
    Complete experiment: Data → Feature Selection → Encoding → QNN
    """
    print("\n" + "="*70)
    print("PHASE 2 EXPERIMENT: FULL PIPELINE TEST")
    print("="*70)
    
    # 1. LOAD DATA
    print("\n[Step 1/5] Loading Iris dataset...")
    loader = DataLoader('iris')
    X_train, X_test, y_train, y_test = loader.get_data()
    
    # Binary classification (classes 0 and 1 only)
    train_mask = y_train < 2
    test_mask = y_test < 2
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    
    print(f"✓ Training samples: {X_train.shape}")
    print(f"✓ Test samples: {X_test.shape}")
    
    # 2. FEATURE SELECTION
    print("\n[Step 2/5] Applying feature selection (PCA)...")
    n_features = 4  # Keep all 4 features for now
    selector = PCASelector(n_features=n_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"✓ Selected {n_features} features")
    print(f"✓ Explained variance: {selector.get_explained_variance().sum():.3f}")
    
    # 3. QUANTUM ENCODING
    print("\n[Step 3/5] Encoding classical data for quantum circuit...")
    encoder = AngleEncoder(scale=np.pi)
    
    # Encode data (already normalized from DataLoader)
    X_train_encoded = np.array([encoder.encode(x) for x in X_train_selected])
    X_test_encoded = np.array([encoder.encode(x) for x in X_test_selected])
    
    print(f"✓ Encoded data shape: {X_train_encoded.shape}")
    print(f"✓ Encoding range: [{X_train_encoded.min():.3f}, {X_train_encoded.max():.3f}]")
    
    # 4. CREATE AND TRAIN QNN
    print("\n[Step 4/5] Creating quantum neural network...")
    qnn = QuantumNeuralNetwork(
        n_qubits=n_features,
        n_layers=2,
        encoding_type='angle',
        learning_rate=0.05
    )
    
    circuit_info = qnn.get_circuit_info()
    print(f"✓ Circuit created:")
    print(f"  - Qubits: {circuit_info['n_qubits']}")
    print(f"  - Layers: {circuit_info['n_layers']}")
    print(f"  - Parameters: {circuit_info['n_parameters']}")
    print(f"  - Gate count: {circuit_info['gate_count']}")
    
    print("\n[Step 5/5] Training quantum neural network...")
    history = qnn.train(
        X_train_encoded, y_train,
        X_test_encoded, y_test,
        epochs=30,
        verbose=True
    )
    
    # 5. FINAL EVALUATION
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    test_metrics = qnn.evaluate(X_test_encoded, y_test)
    train_metrics = qnn.evaluate(X_train_encoded, y_train)
    
    print(f"\nTraining Performance:")
    print(f"  Accuracy: {train_metrics['accuracy']:.2%}")
    print(f"  Loss: {train_metrics['loss']:.4f}")
    
    print(f"\nTest Performance:")
    print(f"  Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    
    print(f"\nConfusion Matrix (Test):")
    print(f"  True Positives:  {test_metrics['true_positives']}")
    print(f"  True Negatives:  {test_metrics['true_negatives']}")
    print(f"  False Positives: {test_metrics['false_positives']}")
    print(f"  False Negatives: {test_metrics['false_negatives']}")
    
    print(f"\nTraining Time: {history['training_time']:.2f} seconds")
    
    print("\n" + "="*70)
    print("✓ EXPERIMENT COMPLETE!")
    print("="*70)
    
    return qnn, history, test_metrics


def compare_configurations():
    """
    Compare different feature selection and encoding combinations
    """
    print("\n" + "="*70)
    print("COMPARING CONFIGURATIONS")
    print("="*70)
    
    # Load data once
    loader = DataLoader('iris')
    X_train, X_test, y_train, y_test = loader.get_data()
    
    # Binary classification
    train_mask = y_train < 2
    test_mask = y_test < 2
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]
    
    # Configurations to test
    configs = [
        {'selector': 'PCA', 'encoder': 'angle'},
        {'selector': 'PCA', 'encoder': 'amplitude'},
        {'selector': 'Correlation', 'encoder': 'angle'},
    ]
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"Configuration {i}/{len(configs)}: {config['selector']} + {config['encoder']}")
        print(f"{'='*70}")
        
        # Feature selection
        if config['selector'] == 'PCA':
            selector = PCASelector(n_features=4)
        else:  # Correlation
            selector = CorrelationSelector(n_features=4)
        
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)
        
        # Encoding
        if config['encoder'] == 'angle':
            encoder = AngleEncoder(scale=np.pi)
        else:  # Amplitude
            encoder = AmplitudeEncoder()
        
        X_train_enc = np.array([encoder.encode(x) for x in X_train_sel])
        X_test_enc = np.array([encoder.encode(x) for x in X_test_sel])
        
        # Train QNN
        qnn = QuantumNeuralNetwork(
            n_qubits=4,
            n_layers=2,
            encoding_type=config['encoder'],
            learning_rate=0.05
        )
        
        history = qnn.train(
            X_train_enc, y_train,
            X_test_enc, y_test,
            epochs=25,
            verbose=False
        )
        
        metrics = qnn.evaluate(X_test_enc, y_test)
        
        results.append({
            'config': f"{config['selector']} + {config['encoder']}",
            'test_accuracy': metrics['accuracy'],
            'train_accuracy': history['train_accuracy'][-1],
            'training_time': history['training_time'],
            'final_loss': metrics['loss']
        })
        
        print(f"Test Accuracy: {metrics['accuracy']:.2%}")
        print(f"Training Time: {history['training_time']:.2f}s")
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"{'Configuration':<25} {'Test Acc':<12} {'Train Acc':<12} {'Time (s)':<10}")
    print("-"*70)
    
    for res in results:
        print(f"{res['config']:<25} "
              f"{res['test_accuracy']:<12.2%} "
              f"{res['train_accuracy']:<12.2%} "
              f"{res['training_time']:<10.2f}")
    
    print("="*70)
    
    # Find best configuration
    best = max(results, key=lambda x: x['test_accuracy'])
    print(f"\n🏆 Best Configuration: {best['config']}")
    print(f"   Test Accuracy: {best['test_accuracy']:.2%}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Run full pipeline experiment
    qnn, history, metrics = experiment_full_pipeline()
    
    # Optional: Compare multiple configurations
    print("\n" + "="*70)
    response = input("Run configuration comparison? (y/n): ")
    if response.lower() == 'y':
        compare_configurations()
    
    print("\n✓ Phase 2 experiments complete!")
    print("✓ Ready for Phase 3: Adaptive Controller")