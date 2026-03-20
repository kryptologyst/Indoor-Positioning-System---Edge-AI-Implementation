#!/usr/bin/env python3
"""
Quick test script for the Indoor Positioning System.

This script runs a quick test of the system to verify everything works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.indoor_positioning import IndoorPositioningSystem


def main():
    """Run a quick test of the Indoor Positioning System."""
    print("Testing Indoor Positioning System...")
    print("=" * 50)
    
    try:
        # Initialize system
        print("1. Initializing system...")
        ips = IndoorPositioningSystem()
        print("   ✓ System initialized successfully")
        
        # Generate data
        print("2. Generating synthetic data...")
        features, labels = ips.generate_synthetic_data()
        print(f"   ✓ Generated {len(features)} samples")
        
        # Prepare data
        print("3. Preparing data...")
        X_train, X_test, y_train, y_test = ips.prepare_data(features, labels)
        print(f"   ✓ Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Build model
        print("4. Building model...")
        model = ips.build_model()
        print("   ✓ Model built successfully")
        
        # Train model (quick training)
        print("5. Training model (quick test)...")
        ips.config['model']['epochs'] = 5  # Quick training
        history = ips.train_model(X_train, y_train, X_test, y_test)
        print("   ✓ Model trained successfully")
        
        # Evaluate model
        print("6. Evaluating model...")
        metrics = ips.evaluate_model(X_test, y_test)
        print(f"   ✓ MAE: {metrics['mae']:.2f}m")
        print(f"   ✓ Accuracy within 1m: {metrics['accuracy_1m']:.1f}%")
        print(f"   ✓ Accuracy within 2m: {metrics['accuracy_2m']:.1f}%")
        
        # Test quantization
        print("7. Testing quantization...")
        quantized_model = ips.quantize_model(X_train)
        if quantized_model:
            print(f"   ✓ Quantized model size: {len(quantized_model) / 1024:.1f} KB")
        else:
            print("   ⚠ Quantization not available")
        
        print("\n" + "=" * 50)
        print("✓ All tests passed successfully!")
        print("The Indoor Positioning System is working correctly.")
        print("\nTo run the full pipeline:")
        print("  python main.py train --config configs/default.yaml")
        print("\nTo launch the demo:")
        print("  python main.py demo --port 8501")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
