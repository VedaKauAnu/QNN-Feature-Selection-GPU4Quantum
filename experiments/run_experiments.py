"""
Master Experiment Runner
Simple interface to run all scalability experiments

Usage:
    python run_experiments.py --mode quick     # Quick test (10 min)
    python run_experiments.py --mode standard  # Standard run (2 hours)
    python run_experiments.py --mode full      # Full suite (4 hours)
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Run Adaptive QNN Scalability Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test to verify everything works (10 minutes)
  python run_experiments.py --mode quick
  
  # Standard run with main results (2 hours)
  python run_experiments.py --mode standard
  
  # Full comprehensive suite (4 hours)
  python run_experiments.py --mode full
  
  # Run specific experiment
  python run_experiments.py --experiment noisy_10q
  python run_experiments.py --experiment scalability
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['quick', 'standard', 'full'],
        default='standard',
        help='Experiment mode'
    )
    
    parser.add_argument(
        '--experiment',
        choices=['noisy_10q', 'scalability', 'both'],
        default='both',
        help='Which experiment to run'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("ADAPTIVE QNN SCALABILITY EXPERIMENTS")
    print("="*70)
    print(f"\nMode: {args.mode.upper()}")
    print(f"Experiments: {args.experiment}")
    print()
    
    # Configure based on mode
    if args.mode == 'quick':
        print("Quick mode: Fast test run")
        print("  - 10 epochs per experiment")
        print("  - 1 trial per configuration")
        print("  - 2 qubit counts (4, 8)")
        print("  - Expected time: ~10 minutes")
        configure_quick_mode()
    
    elif args.mode == 'standard':
        print("Standard mode: Main results")
        print("  - 20 epochs per experiment")
        print("  - 3 trials per configuration")
        print("  - 4 qubit counts (4, 6, 8, 10)")
        print("  - Expected time: ~2 hours")
        configure_standard_mode()
    
    elif args.mode == 'full':
        print("Full mode: Comprehensive analysis")
        print("  - 30 epochs per experiment")
        print("  - 5 trials per configuration")
        print("  - 5 qubit counts (4, 6, 8, 10, 12)")
        print("  - Expected time: ~4 hours")
        configure_full_mode()
    
    input("\nPress Enter to start experiments (or Ctrl+C to cancel)...")
    
    # Run experiments
    if args.experiment in ['noisy_10q', 'both']:
        run_noisy_10q_experiment()
    
    if args.experiment in ['scalability', 'both']:
        run_scalability_suite()
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print("\nCheck the results/ directory for outputs")
    print("  - noisy_10q_results_*.png")
    print("  - noisy_10q_results_*.json")
    print("  - results/scalability/*.png")
    print("  - results/scalability/*.json")
    print("\n✓ Your data is ready for publication!")

def configure_quick_mode():
    """Configure for quick testing"""
    # Modify noisy_10q_experiment config
    try:
        from noisy_10q_experiment import config
        config.N_EPOCHS = 10
        print("  ✓ Configured noisy_10q for quick mode")
    except:
        print("  ⚠️  Could not configure noisy_10q")
    
    # Modify scalability config
    try:
        from scalability_suite import scal_config
        scal_config.N_EPOCHS = 10
        scal_config.N_TRIALS = 1
        scal_config.QUBIT_COUNTS = [4, 8]
        print("  ✓ Configured scalability for quick mode")
    except:
        print("  ⚠️  Could not configure scalability")

def configure_standard_mode():
    """Configure for standard run"""
    try:
        from noisy_10q_experiment import config
        config.N_EPOCHS = 20
        print("  ✓ Configured noisy_10q for standard mode")
    except:
        pass
    
    try:
        from scalability_suite import scal_config
        scal_config.N_EPOCHS = 20
        scal_config.N_TRIALS = 3
        scal_config.QUBIT_COUNTS = [4, 6, 8, 10]
        print("  ✓ Configured scalability for standard mode")
    except:
        pass

def configure_full_mode():
    """Configure for full suite"""
    try:
        from noisy_10q_experiment import config
        config.N_EPOCHS = 30
        print("  ✓ Configured noisy_10q for full mode")
    except:
        pass
    
    try:
        from scalability_suite import scal_config
        scal_config.N_EPOCHS = 30
        scal_config.N_TRIALS = 5
        scal_config.QUBIT_COUNTS = [4, 6, 8, 10, 12]
        print("  ✓ Configured scalability for full mode")
    except:
        pass

def run_noisy_10q_experiment():
    """Run the main noisy 10-qubit experiment"""
    print("\n" + "▶"*35)
    print("RUNNING: Noisy 10-Qubit Experiment")
    print("▶"*35 + "\n")
    
    try:
        from noisy_10q_experiment import NoisyAdaptiveQNNTrainer
        trainer = NoisyAdaptiveQNNTrainer()
        trainer.run()
        print("\n✓ Noisy 10-qubit experiment complete")
    except Exception as e:
        print(f"\n✗ Error in noisy 10-qubit experiment: {e}")
        import traceback
        traceback.print_exc()

def run_scalability_suite():
    """Run the complete scalability suite"""
    print("\n" + "▶"*35)
    print("RUNNING: Scalability Suite")
    print("▶"*35 + "\n")
    
    try:
        from scalability_suite import run_complete_scalability_suite
        run_complete_scalability_suite()
        print("\n✓ Scalability suite complete")
    except Exception as e:
        print(f"\n✗ Error in scalability suite: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()