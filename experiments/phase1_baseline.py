"""
Phase 1 Baseline - Run all validation tests
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_pipeline import (
    test_data_pipeline,
    test_feature_selection, 
    test_quantum_encoding,
    test_pennylane_setup
)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PHASE 1: RUNNING ALL VALIDATION TESTS")
    print("="*60)
    
    test_data_pipeline()
    test_feature_selection()
    test_quantum_encoding()
    test_pennylane_setup()
    
    print("\n" + "="*60)
    print("✓ PHASE 1 COMPLETE!")
    print("="*60)