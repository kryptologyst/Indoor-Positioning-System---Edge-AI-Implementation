"""
Tests for Indoor Positioning System.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.indoor_positioning import IndoorPositioningSystem


class TestIndoorPositioningSystem:
    """Test cases for IndoorPositioningSystem class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ips = IndoorPositioningSystem()
    
    def test_initialization(self):
        """Test system initialization."""
        assert self.ips is not None
        assert self.ips.config is not None
        assert self.ips.scaler is not None
        assert self.ips.model is None
        assert self.ips.quantized_model is None
    
    def test_config_loading(self):
        """Test configuration loading."""
        config = self.ips.config
        assert 'seed' in config
        assert 'n_samples' in config
        assert 'n_access_points' in config
        assert 'model' in config
        assert config['n_access_points'] == 4
    
    def test_data_generation(self):
        """Test synthetic data generation."""
        features, labels = self.ips.generate_synthetic_data()
        
        assert features.shape[1] == 4  # 4 access points
        assert labels.shape[1] == 2   # x, y coordinates
        assert features.shape[0] == labels.shape[0]
        assert features.shape[0] == self.ips.config['n_samples']
        
        # Check RSSI range
        assert np.all(features >= -90)
        assert np.all(features <= -30)
        
        # Check coordinate range
        area_size = self.ips.config['area_size']
        assert np.all(labels[:, 0] >= 0)
        assert np.all(labels[:, 0] <= area_size[0])
        assert np.all(labels[:, 1] >= 0)
        assert np.all(labels[:, 1] <= area_size[1])
    
    def test_data_preparation(self):
        """Test data preparation and scaling."""
        features, labels = self.ips.generate_synthetic_data()
        X_train, X_test, y_train, y_test = self.ips.prepare_data(features, labels)
        
        assert len(X_train) + len(X_test) == len(features)
        assert len(y_train) + len(y_test) == len(labels)
        
        # Check scaling
        assert np.allclose(np.mean(X_train), 0, atol=1e-10)
        assert np.allclose(np.std(X_train), 1, atol=1e-10)
    
    def test_model_building(self):
        """Test model building."""
        model = self.ips.build_model()
        
        assert model is not None
        assert self.ips.model is not None
        assert model.input_shape == (None, 4)
        assert model.output_shape == (None, 2)
    
    def test_model_training(self):
        """Test model training."""
        # Generate small dataset for quick training
        self.ips.config['n_samples'] = 100
        self.ips.config['model']['epochs'] = 2
        
        features, labels = self.ips.generate_synthetic_data()
        X_train, X_test, y_train, y_test = self.ips.prepare_data(features, labels)
        
        model = self.ips.build_model()
        history = self.ips.train_model(X_train, y_train, X_test, y_test)
        
        assert history is not None
        assert 'loss' in history.history
        assert 'val_loss' in history.history
        assert len(history.history['loss']) <= 2  # Max 2 epochs
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        # Generate small dataset
        self.ips.config['n_samples'] = 100
        self.ips.config['model']['epochs'] = 2
        
        features, labels = self.ips.generate_synthetic_data()
        X_train, X_test, y_train, y_test = self.ips.prepare_data(features, labels)
        
        model = self.ips.train_model(X_train, y_train, X_test, y_test)
        metrics = self.ips.evaluate_model(X_test, y_test)
        
        assert 'mae' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'accuracy_1m' in metrics
        assert 'accuracy_2m' in metrics
        assert 'accuracy_5m' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy_1m'] <= 100
        assert 0 <= metrics['accuracy_2m'] <= 100
        assert 0 <= metrics['accuracy_5m'] <= 100
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
    
    def test_quantization(self):
        """Test model quantization."""
        # Generate small dataset
        self.ips.config['n_samples'] = 100
        self.ips.config['model']['epochs'] = 2
        
        features, labels = self.ips.generate_synthetic_data()
        X_train, X_test, y_train, y_test = self.ips.prepare_data(features, labels)
        
        model = self.ips.train_model(X_train, y_train, X_test, y_test)
        
        # Test quantization
        quantized_model = self.ips.quantize_model(X_train)
        
        if quantized_model is not None:
            assert isinstance(quantized_model, bytes)
            assert len(quantized_model) > 0
    
    def test_full_pipeline(self):
        """Test complete pipeline."""
        # Use small dataset for quick testing
        self.ips.config['n_samples'] = 100
        self.ips.config['model']['epochs'] = 2
        
        results = self.ips.run_full_pipeline()
        
        assert 'metrics' in results
        assert 'model_size_mb' in results
        assert 'training_history' in results
        assert 'config' in results
        
        # Check metrics
        metrics = results['metrics']
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'accuracy_1m' in metrics
        
        # Check model size
        assert results['model_size_mb'] > 0


def test_deterministic_behavior():
    """Test that the system produces deterministic results."""
    ips1 = IndoorPositioningSystem()
    ips2 = IndoorPositioningSystem()
    
    # Generate data with same seed
    features1, labels1 = ips1.generate_synthetic_data()
    features2, labels2 = ips2.generate_synthetic_data()
    
    # Should be identical due to same seed
    np.testing.assert_array_equal(features1, features2)
    np.testing.assert_array_equal(labels1, labels2)


if __name__ == "__main__":
    pytest.main([__file__])
