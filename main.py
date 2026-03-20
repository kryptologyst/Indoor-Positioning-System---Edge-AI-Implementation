#!/usr/bin/env python3
"""
Indoor Positioning System - Main Entry Point

This is the main entry point for the Indoor Positioning System.
It provides a command-line interface for training, evaluation, and deployment.

NOT FOR SAFETY-CRITICAL USE - Research and educational purposes only.
"""

import argparse
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.indoor_positioning import IndoorPositioningSystem
from utils.logger import setup_logging
from utils.config import load_config

def main():
    """Main entry point for the Indoor Positioning System."""
    parser = argparse.ArgumentParser(
        description="Indoor Positioning System - WiFi-based location estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train and evaluate the model
  python main.py train --config configs/default.yaml
  
  # Run with specific device configuration
  python main.py train --config configs/default.yaml --device-config configs/device/raspberry_pi.yaml
  
  # Export models for edge deployment
  python main.py export --model-path assets/models/model.h5 --target tflite
  
  # Run evaluation only
  python main.py evaluate --model-path assets/models/model.h5 --data-path data/processed/test_data.npz
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--device-config",
        type=str,
        help="Path to device-specific configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the indoor positioning model")
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="assets/models",
        help="Output directory for trained models"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    eval_parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    eval_parser.add_argument(
        "--data-path",
        type=str,
        help="Path to test data"
    )
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model for edge deployment")
    export_parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    export_parser.add_argument(
        "--target",
        type=str,
        choices=["tflite", "onnx", "coreml"],
        required=True,
        help="Target deployment format"
    )
    export_parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for exported model"
    )
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run interactive demo")
    demo_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for demo server"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    if args.device_config:
        device_config = load_config(args.device_config)
        config.update(device_config)
    
    logger.info("Starting Indoor Positioning System")
    logger.info(f"Configuration loaded from: {args.config}")
    if args.device_config:
        logger.info(f"Device configuration loaded from: {args.device_config}")
    
    # Execute command
    if args.command == "train":
        train_model(config, args.output_dir)
    elif args.command == "evaluate":
        evaluate_model(config, args.model_path, args.data_path)
    elif args.command == "export":
        export_model(config, args.model_path, args.target, args.output_path)
    elif args.command == "demo":
        run_demo(config, args.port)
    else:
        parser.print_help()
        sys.exit(1)

def train_model(config: dict, output_dir: str):
    """Train the indoor positioning model."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
    
    # Initialize system
    ips = IndoorPositioningSystem(config)
    
    # Run training pipeline
    results = ips.run_full_pipeline()
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / "model.h5"
    ips.model.save(str(model_path))
    logger.info(f"Model saved to: {model_path}")
    
    # Print results summary
    print("\n" + "="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    print(f"Mean Absolute Error: {results['metrics']['mae']:.2f} meters")
    print(f"Root Mean Square Error: {results['metrics']['rmse']:.2f} meters")
    print(f"Accuracy within 1m: {results['metrics']['accuracy_1m']:.1f}%")
    print(f"Accuracy within 2m: {results['metrics']['accuracy_2m']:.1f}%")
    print(f"Accuracy within 5m: {results['metrics']['accuracy_5m']:.1f}%")
    print(f"Model size: {results['model_size_mb']:.2f} MB")
    if results['quantized_model_size_kb']:
        print(f"Quantized model size: {results['quantized_model_size_kb']:.1f} KB")
        print(f"Compression ratio: {results['model_size_mb'] * 1024 / results['quantized_model_size_kb']:.1f}x")
    print("="*60)

def evaluate_model(config: dict, model_path: str, data_path: str = None):
    """Evaluate model performance."""
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating model: {model_path}")
    
    # Initialize system
    ips = IndoorPositioningSystem(config)
    
    # Load model
    ips.model = tf.keras.models.load_model(model_path)
    
    if data_path:
        # Load test data
        data = np.load(data_path)
        X_test, y_test = data['X_test'], data['y_test']
    else:
        # Generate test data
        features, labels = ips.generate_synthetic_data()
        _, X_test, _, y_test = ips.prepare_data(features, labels)
    
    # Evaluate
    metrics = ips.evaluate_model(X_test, y_test)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("="*60)

def export_model(config: dict, model_path: str, target: str, output_path: str = None):
    """Export model for edge deployment."""
    logger = logging.getLogger(__name__)
    logger.info(f"Exporting model to {target} format...")
    
    # Initialize system
    ips = IndoorPositioningSystem(config)
    
    # Load model
    ips.model = tf.keras.models.load_model(model_path)
    
    # Generate representative data for quantization
    features, _ = ips.generate_synthetic_data()
    X_train, _, _, _ = ips.prepare_data(features, np.zeros((len(features), 2)))
    
    # Export based on target
    if target == "tflite":
        quantized_model = ips.quantize_model(X_train)
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(quantized_model)
        logger.info(f"TFLite model exported successfully")
    elif target == "onnx":
        # TODO: Implement ONNX export
        logger.warning("ONNX export not yet implemented")
    elif target == "coreml":
        # TODO: Implement CoreML export
        logger.warning("CoreML export not yet implemented")

def run_demo(config: dict, port: int):
    """Run the interactive demo."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting demo server on port {port}")
    
    # Import demo module
    try:
        from demo.app import run_streamlit_app
        run_streamlit_app(config, port)
    except ImportError:
        logger.error("Demo module not found. Please ensure demo/app.py exists.")
        sys.exit(1)

if __name__ == "__main__":
    main()
