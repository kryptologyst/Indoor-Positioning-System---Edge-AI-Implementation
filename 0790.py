"""
Indoor Positioning System (IPS) - Edge AI Implementation

This module implements a WiFi-based indoor positioning system using machine learning
to estimate real-time location of people or assets inside buildings.

NOT FOR SAFETY-CRITICAL USE - Research and educational purposes only.
"""

import os
import random
from typing import Tuple, Dict, List, Optional, Any
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import yaml
from pathlib import Path
import logging

# Set up deterministic behavior
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class IndoorPositioningSystem:
    """
    WiFi-based Indoor Positioning System using neural networks.
    
    This class implements a complete IPS pipeline including data generation,
    model training, evaluation, and edge optimization.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Indoor Positioning System.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        self.model = None
        self.quantized_model = None
        
        # Set seeds for reproducibility
        set_seed(self.config.get('seed', 42))
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults."""
        default_config = {
            'seed': 42,
            'n_samples': 2000,
            'n_access_points': 4,
            'area_size': (20, 20),  # meters
            'rssi_range': (-90, -30),  # dBm
            'noise_std': 5,
            'test_size': 0.2,
            'model': {
                'hidden_layers': [64, 32],
                'activation': 'relu',
                'dropout': 0.2,
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.001
            },
            'quantization': {
                'enabled': True,
                'representative_dataset_size': 100
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
                
        return default_config
    
    def generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic WiFi RSSI data for indoor positioning.
        
        Returns:
            Tuple of (features, labels) where features are RSSI values
            and labels are (x, y) coordinates
        """
        logger.info("Generating synthetic WiFi RSSI data...")
        
        n_samples = self.config['n_samples']
        n_aps = self.config['n_access_points']
        area_size = self.config['area_size']
        rssi_range = self.config['rssi_range']
        noise_std = self.config['noise_std']
        
        # Generate random coordinates
        x_coords = np.random.uniform(0, area_size[0], n_samples)
        y_coords = np.random.uniform(0, area_size[1], n_samples)
        
        # Simulate access point positions (fixed locations)
        ap_positions = np.array([
            [0, 0],           # AP1: corner
            [area_size[0], 0], # AP2: corner
            [0, area_size[1]], # AP3: corner
            [area_size[0], area_size[1]]  # AP4: corner
        ])
        
        # Calculate distances and simulate RSSI based on distance
        features = np.zeros((n_samples, n_aps))
        
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            for j, (ap_x, ap_y) in enumerate(ap_positions):
                distance = np.sqrt((x - ap_x)**2 + (y - ap_y)**2)
                # RSSI decreases with distance (simplified path loss model)
                base_rssi = rssi_range[1] - 2 * distance  # -2 dBm per meter
                rssi = np.clip(base_rssi + np.random.normal(0, noise_std), 
                             rssi_range[0], rssi_range[1])
                features[i, j] = rssi
        
        labels = np.stack([x_coords, y_coords], axis=1)
        
        logger.info(f"Generated {n_samples} samples with {n_aps} access points")
        return features, labels
    
    def prepare_data(self, features: np.ndarray, labels: np.ndarray) -> Tuple:
        """
        Prepare data for training with scaling and splitting.
        
        Args:
            features: RSSI features
            labels: Position labels
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data for training...")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, 
            test_size=self.config['test_size'], 
            random_state=self.config['seed']
        )
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def build_model(self) -> keras.Model:
        """
        Build the neural network model for position prediction.
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building neural network model...")
        
        model_config = self.config['model']
        n_aps = self.config['n_access_points']
        
        model = models.Sequential([
            layers.Input(shape=(n_aps,), name='rssi_input'),
            layers.Dense(model_config['hidden_layers'][0], 
                        activation=model_config['activation'],
                        name='hidden_1'),
            layers.Dropout(model_config['dropout']),
            layers.Dense(model_config['hidden_layers'][1], 
                        activation=model_config['activation'],
                        name='hidden_2'),
            layers.Dropout(model_config['dropout']),
            layers.Dense(2, name='position_output')  # x, y coordinates
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=model_config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logger.info("Model built successfully")
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray) -> keras.callbacks.History:
        """
        Train the model with early stopping and learning rate scheduling.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Training history
        """
        logger.info("Training model...")
        
        model_config = self.config['model']
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size'],
            callbacks=callbacks_list,
            verbose=1
        )
        
        logger.info("Training completed")
        return history
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model...")
        
        # Get predictions
        y_pred = self.model.predict(X_test, verbose=0)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Calculate accuracy within different distance thresholds
        distances = np.sqrt(np.sum((y_test - y_pred)**2, axis=1))
        accuracy_1m = np.mean(distances <= 1.0) * 100
        accuracy_2m = np.mean(distances <= 2.0) * 100
        accuracy_5m = np.mean(distances <= 5.0) * 100
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'accuracy_1m': accuracy_1m,
            'accuracy_2m': accuracy_2m,
            'accuracy_5m': accuracy_5m,
            'mean_distance_error': np.mean(distances),
            'std_distance_error': np.std(distances)
        }
        
        logger.info(f"MAE: {mae:.2f}m, RMSE: {rmse:.2f}m")
        logger.info(f"Accuracy within 1m: {accuracy_1m:.1f}%")
        logger.info(f"Accuracy within 2m: {accuracy_2m:.1f}%")
        logger.info(f"Accuracy within 5m: {accuracy_5m:.1f}%")
        
        return metrics
    
    def quantize_model(self, X_train: np.ndarray) -> keras.Model:
        """
        Create a quantized version of the model for edge deployment.
        
        Args:
            X_train: Training data for representative dataset
            
        Returns:
            Quantized model
        """
        if not self.config['quantization']['enabled']:
            logger.info("Quantization disabled in config")
            return None
            
        logger.info("Creating quantized model...")
        
        # Create representative dataset
        rep_size = self.config['quantization']['representative_dataset_size']
        rep_indices = np.random.choice(len(X_train), rep_size, replace=False)
        representative_dataset = X_train[rep_indices]
        
        # Convert to TensorFlow Lite format
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        def representative_data_gen():
            for data in representative_dataset:
                yield [data.astype(np.float32)]
        
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        
        try:
            quantized_tflite_model = converter.convert()
            
            # Save quantized model
            tflite_path = Path("assets/models/quantized_model.tflite")
            tflite_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(tflite_path, 'wb') as f:
                f.write(quantized_tflite_model)
            
            logger.info(f"Quantized model saved to {tflite_path}")
            logger.info(f"Quantized model size: {len(quantized_tflite_model) / 1024:.1f} KB")
            
            return quantized_tflite_model
            
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return None
    
    def visualize_results(self, X_test: np.ndarray, y_test: np.ndarray, 
                         save_path: Optional[str] = None) -> None:
        """
        Visualize prediction results and model performance.
        
        Args:
            X_test, y_test: Test data
            save_path: Path to save plots
        """
        logger.info("Creating visualizations...")
        
        # Get predictions
        y_pred = self.model.predict(X_test, verbose=0)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Indoor Positioning System - Results Analysis', fontsize=16)
        
        # Plot 1: Actual vs Predicted positions
        ax1 = axes[0, 0]
        ax1.scatter(y_test[:, 0], y_test[:, 1], alpha=0.6, label='Actual', s=20)
        ax1.scatter(y_pred[:, 0], y_pred[:, 1], alpha=0.6, label='Predicted', s=20)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Actual vs Predicted Positions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        ax2 = axes[0, 1]
        distances = np.sqrt(np.sum((y_test - y_pred)**2, axis=1))
        ax2.hist(distances, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Distance Error (m)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Position Errors')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: RSSI vs Position correlation
        ax3 = axes[1, 0]
        scatter = ax3.scatter(X_test[:, 0], X_test[:, 1], c=y_test[:, 0], 
                            cmap='viridis', alpha=0.6)
        ax3.set_xlabel('AP1 RSSI (dBm)')
        ax3.set_ylabel('AP2 RSSI (dBm)')
        ax3.set_title('RSSI Correlation with X Position')
        plt.colorbar(scatter, ax=ax3, label='X Position (m)')
        
        # Plot 4: Error heatmap
        ax4 = axes[1, 1]
        # Create grid for error heatmap
        x_bins = np.linspace(0, 20, 10)
        y_bins = np.linspace(0, 20, 10)
        error_grid = np.zeros((9, 9))
        
        for i in range(len(y_test)):
            x_idx = np.digitize(y_test[i, 0], x_bins) - 1
            y_idx = np.digitize(y_test[i, 1], y_bins) - 1
            if 0 <= x_idx < 9 and 0 <= y_idx < 9:
                error_grid[y_idx, x_idx] += distances[i]
        
        # Count samples per grid cell
        count_grid = np.zeros((9, 9))
        for i in range(len(y_test)):
            x_idx = np.digitize(y_test[i, 0], x_bins) - 1
            y_idx = np.digitize(y_test[i, 1], y_bins) - 1
            if 0 <= x_idx < 9 and 0 <= y_idx < 9:
                count_grid[y_idx, x_idx] += 1
        
        # Average error per grid cell
        error_grid = np.divide(error_grid, count_grid, 
                              out=np.zeros_like(error_grid), 
                              where=count_grid!=0)
        
        im = ax4.imshow(error_grid, cmap='Reds', aspect='auto')
        ax4.set_xlabel('X Position Bin')
        ax4.set_ylabel('Y Position Bin')
        ax4.set_title('Average Error Heatmap')
        plt.colorbar(im, ax=ax4, label='Avg Error (m)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plots saved to {save_path}")
        
        plt.show()
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete indoor positioning pipeline.
        
        Returns:
            Dictionary containing results and metrics
        """
        logger.info("Starting Indoor Positioning System pipeline...")
        
        # Generate data
        features, labels = self.generate_synthetic_data()
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(features, labels)
        
        # Build and train model
        model = self.build_model()
        history = self.train_model(X_train, y_train, X_test, y_test)
        
        # Evaluate model
        metrics = self.evaluate_model(X_test, y_test)
        
        # Create quantized model
        quantized_model = self.quantize_model(X_train)
        
        # Visualize results
        self.visualize_results(X_test, y_test, "assets/plots/results_analysis.png")
        
        # Compile results
        results = {
            'metrics': metrics,
            'model_size_mb': self.model.count_params() * 4 / (1024 * 1024),  # 4 bytes per float32
            'quantized_model_size_kb': len(quantized_model) / 1024 if quantized_model else None,
            'training_history': history.history,
            'config': self.config
        }
        
        logger.info("Pipeline completed successfully!")
        return results


def main():
    """Main function to run the Indoor Positioning System."""
    # Create output directories
    Path("assets/models").mkdir(parents=True, exist_ok=True)
    Path("assets/plots").mkdir(parents=True, exist_ok=True)
    
    # Initialize and run the system
    ips = IndoorPositioningSystem()
    results = ips.run_full_pipeline()
    
    # Print summary
    print("\n" + "="*60)
    print("INDOOR POSITIONING SYSTEM - RESULTS SUMMARY")
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
    print("NOT FOR SAFETY-CRITICAL USE - Research and educational purposes only")
    print("="*60)


if __name__ == "__main__":
    main()

