"""
Robust Deep Learning Training System for F1 Car Health Monitoring
=================================================================

This module provides a comprehensive solution for training deep learning models
on F1 telemetry data with built-in fallbacks for TensorFlow 2.18.0 issues.

Author: MadhaV-73
Version: 1.0.0
"""

import os
import sys
import warnings
import logging
import pickle
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

# Robust TensorFlow import with version checking
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    TF_VERSION = tf.__version__
    print(f"✅ TensorFlow {TF_VERSION} loaded successfully")
except ImportError:
    TF_AVAILABLE = False
    TF_VERSION = None
    print("❌ TensorFlow not available")

# Alternative ML libraries for fallback
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
    print("✅ Scikit-learn available for fallback")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("❌ Scikit-learn not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for robust training system"""
    
    # Model architecture
    model_type: str = "informer_tcn"  # informer_tcn, lstm, cnn_lstm, mlp
    sequence_length: int = 50
    prediction_horizon: int = 10
    feature_dim: int = 46
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 15
    
    # Robustness settings
    enable_fallback: bool = True
    max_retries: int = 3
    memory_limit_gb: float = 4.0
    
    # Environment settings
    use_mixed_precision: bool = True
    enable_xla: bool = False  # Disabled due to TF 2.18.0 issues
    distributed_strategy: bool = False  # Disabled due to transfer manager issues
    
    # Paths
    model_save_path: str = "./models"
    log_path: str = "./logs"
    checkpoint_path: str = "./checkpoints"


class RobustF1Trainer:
    """
    Robust training system for F1 car health monitoring with multiple fallback strategies
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.training_history = []
        self.current_strategy = None
        
        # Create directories
        os.makedirs(config.model_save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)
        os.makedirs(config.checkpoint_path, exist_ok=True)
        
        # Initialize strategies
        self.strategies = self._initialize_strategies()
        self.logger.info(f"Initialized {len(self.strategies)} training strategies")
    
    def _initialize_strategies(self) -> List[Dict]:
        """Initialize available training strategies"""
        strategies = []
        
        # Strategy 1: Native TensorFlow with custom training loop
        if TF_AVAILABLE:
            strategies.append({
                'name': 'custom_training_loop',
                'type': 'tensorflow',
                'description': 'Custom training loop bypassing model.fit()',
                'priority': 1
            })
        
        # Strategy 2: TensorFlow with eager execution
        if TF_AVAILABLE:
            strategies.append({
                'name': 'eager_execution',
                'type': 'tensorflow',
                'description': 'TensorFlow eager execution mode',
                'priority': 2
            })
        
        # Strategy 3: TensorFlow with reduced functionality
        if TF_AVAILABLE:
            strategies.append({
                'name': 'simple_tensorflow',
                'type': 'tensorflow',
                'description': 'Simplified TensorFlow without advanced features',
                'priority': 3
            })
        
        # Strategy 4: Scikit-learn fallback
        if SKLEARN_AVAILABLE:
            strategies.append({
                'name': 'sklearn_ensemble',
                'type': 'sklearn',
                'description': 'Ensemble of scikit-learn models',
                'priority': 4
            })
        
        return strategies
    
    def detect_environment(self) -> Dict[str, Any]:
        """Detect current environment and capabilities"""
        env_info = {
            'platform': sys.platform,
            'python_version': sys.version,
            'tensorflow_version': TF_VERSION,
            'is_kaggle': os.path.exists('/kaggle'),
            'is_colab': 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False,
            'gpu_available': False,
            'memory_available_gb': 0
        }
        
        if TF_AVAILABLE:
            env_info['gpu_available'] = len(tf.config.list_physical_devices('GPU')) > 0
            if env_info['gpu_available']:
                env_info['gpu_details'] = tf.config.list_physical_devices('GPU')
        
        # Estimate available memory
        try:
            import psutil
            env_info['memory_available_gb'] = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            env_info['memory_available_gb'] = 4.0  # Conservative estimate
        
        self.logger.info(f"Environment detected: {env_info}")
        return env_info
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training with robust preprocessing"""
        self.logger.info("Preparing data for training...")
        
        try:
            # Select numerical features
            numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove identifier columns
            feature_cols = [col for col in numerical_cols 
                          if not any(x in col.lower() for x in ['id', 'number', 'index'])]
            
            # Handle missing values
            data_clean = data[feature_cols].ffill().fillna(0)
            
            # Create sequences for time series prediction
            X, y = self._create_sequences(data_clean)
            
            self.logger.info(f"Data prepared: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise
    
    def _create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        seq_len = self.config.sequence_length
        pred_horizon = self.config.prediction_horizon
        
        # Sort by time-related columns if available
        if 'lap_number' in data.columns:
            data = data.sort_values('lap_number')
        
        values = data.values.astype(np.float32)
        
        X, y = [], []
        for i in range(len(values) - seq_len - pred_horizon + 1):
            X.append(values[i:(i + seq_len)])
            y.append(values[i + seq_len:i + seq_len + pred_horizon, 0])  # Predict first feature
        
        return np.array(X), np.array(y)
    
    def train_with_strategy(self, X: np.ndarray, y: np.ndarray, strategy: Dict) -> Dict[str, Any]:
        """Train model using specific strategy"""
        self.logger.info(f"Training with strategy: {strategy['name']}")
        self.current_strategy = strategy
        
        try:
            if strategy['type'] == 'tensorflow':
                return self._train_tensorflow_strategy(X, y, strategy)
            elif strategy['type'] == 'sklearn':
                return self._train_sklearn_strategy(X, y, strategy)
            else:
                raise ValueError(f"Unknown strategy type: {strategy['type']}")
                
        except Exception as e:
            self.logger.error(f"Strategy {strategy['name']} failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _train_tensorflow_strategy(self, X: np.ndarray, y: np.ndarray, strategy: Dict) -> Dict[str, Any]:
        """Train using TensorFlow strategies"""
        
        if strategy['name'] == 'custom_training_loop':
            return self._custom_training_loop(X, y)
        elif strategy['name'] == 'eager_execution':
            return self._eager_execution_training(X, y)
        elif strategy['name'] == 'simple_tensorflow':
            return self._simple_tensorflow_training(X, y)
        else:
            raise ValueError(f"Unknown TensorFlow strategy: {strategy['name']}")
    
    def _custom_training_loop(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Custom training loop that bypasses model.fit() to avoid transfer manager issues"""
        self.logger.info("Using custom training loop strategy")
        
        try:
            # Enable eager execution to avoid graph issues
            tf.config.run_functions_eagerly(True)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.validation_split, random_state=42
            )
            
            # Create model
            model = self._create_informer_tcn_model(X.shape[1:])
            
            # Create optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
            loss_fn = tf.keras.losses.MeanSquaredError()
            
            # Training metrics
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            patience_counter = 0
            
            # Custom training loop
            for epoch in range(self.config.epochs):
                epoch_train_loss = 0
                num_batches = 0
                
                # Training step
                for i in range(0, len(X_train), self.config.batch_size):
                    batch_X = X_train[i:i + self.config.batch_size]
                    batch_y = y_train[i:i + self.config.batch_size]
                    
                    with tf.GradientTape() as tape:
                        predictions = model(batch_X, training=True)
                        loss = loss_fn(batch_y, predictions)
                    
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    
                    epoch_train_loss += loss.numpy()
                    num_batches += 1
                
                # Validation step
                val_predictions = model(X_val, training=False)
                val_loss = loss_fn(y_val, val_predictions).numpy()
                
                # Record metrics
                avg_train_loss = epoch_train_loss / num_batches
                train_losses.append(avg_train_loss)
                val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    model.save_weights(f"{self.config.checkpoint_path}/best_model.h5")
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Load best model
            model.load_weights(f"{self.config.checkpoint_path}/best_model.h5")
            
            return {
                'success': True,
                'model': model,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'strategy': 'custom_training_loop'
            }
            
        except Exception as e:
            self.logger.error(f"Custom training loop failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _eager_execution_training(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Training with eager execution enabled"""
        self.logger.info("Using eager execution strategy")
        
        try:
            # Force eager execution
            tf.config.run_functions_eagerly(True)
            
            # Disable XLA compilation
            tf.config.optimizer.set_jit(False)
            
            # Simple model compilation and training
            model = self._create_simple_model(X.shape[1:])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss='mse',
                metrics=['mae'],
                run_eagerly=True  # Force eager execution
            )
            
            # Custom callback to handle issues
            class RobustCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if logs and logs.get('loss', float('inf')) > 1e6:
                        self.model.stop_training = True
                        logger.warning("Training stopped due to loss explosion")
            
            callbacks = [
                RobustCallback(),
                tf.keras.callbacks.EarlyStopping(
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                )
            ]
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.validation_split, random_state=42
            )
            
            # Train with reduced batch size to avoid memory issues
            batch_size = min(self.config.batch_size, 16)
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=self.config.epochs,
                callbacks=callbacks,
                verbose=1
            )
            
            return {
                'success': True,
                'model': model,
                'history': history.history,
                'strategy': 'eager_execution'
            }
            
        except Exception as e:
            self.logger.error(f"Eager execution training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _simple_tensorflow_training(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Simple TensorFlow training with minimal features"""
        self.logger.info("Using simple TensorFlow strategy")
        
        try:
            # Reset TensorFlow state
            tf.keras.backend.clear_session()
            
            # Create simple sequential model
            model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=X.shape[1:]),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(y.shape[1] if len(y.shape) > 1 else 1)
            ])
            
            # Simple compilation
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            # Train with minimal callbacks
            history = model.fit(
                X, y,
                validation_split=self.config.validation_split,
                batch_size=min(self.config.batch_size, 32),
                epochs=min(self.config.epochs, 50),
                verbose=1
            )
            
            return {
                'success': True,
                'model': model,
                'history': history.history,
                'strategy': 'simple_tensorflow'
            }
            
        except Exception as e:
            self.logger.error(f"Simple TensorFlow training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _train_sklearn_strategy(self, X: np.ndarray, y: np.ndarray, strategy: Dict) -> Dict[str, Any]:
        """Train using scikit-learn as fallback"""
        self.logger.info("Using scikit-learn fallback strategy")
        
        try:
            # Check for empty data
            if X.size == 0 or y.size == 0:
                raise ValueError("Empty dataset provided")
            
            # Reshape data for sklearn
            X_flat = X.reshape(X.shape[0], -1)
            
            # Handle y dimension mismatch
            if len(y.shape) > 1 and y.shape[1] > 1:
                # For multi-output, take only the first output for simplicity
                y_flat = y[:, 0] if y.shape[0] == X.shape[0] else y.ravel()
            else:
                y_flat = y.ravel() if len(y.shape) > 1 else y
            
            # Ensure X and y have same number of samples
            min_samples = min(len(X_flat), len(y_flat))
            X_flat = X_flat[:min_samples]
            y_flat = y_flat[:min_samples]
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_flat, y_flat, test_size=self.config.validation_split, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Ensemble of models
            models = {
                'mlp': MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42
                ),
                'rf': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'gbm': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                )
            }
            
            trained_models = {}
            scores = {}
            
            for name, model in models.items():
                self.logger.info(f"Training {name}...")
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                train_score = model.score(X_train_scaled, y_train)
                val_score = model.score(X_val_scaled, y_val)
                
                trained_models[name] = model
                scores[name] = {'train': train_score, 'val': val_score}
                
                self.logger.info(f"{name} - Train R²: {train_score:.4f}, Val R²: {val_score:.4f}")
            
            # Select best model
            best_model_name = max(scores.keys(), key=lambda k: scores[k]['val'])
            best_model = trained_models[best_model_name]
            
            return {
                'success': True,
                'models': trained_models,
                'best_model': best_model,
                'best_model_name': best_model_name,
                'scores': scores,
                'scaler': scaler,
                'strategy': 'sklearn_ensemble'
            }
            
        except Exception as e:
            self.logger.error(f"Scikit-learn training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_informer_tcn_model(self, input_shape: Tuple):
        """Create Informer-TCN model architecture"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available for model creation")
        
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # TCN layers
        x = tf.keras.layers.Conv1D(64, 3, padding='causal', activation='relu')(inputs)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Conv1D(64, 3, padding='causal', activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Attention mechanism (simplified Informer)
        attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = tf.keras.layers.Add()([x, attention])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Global pooling and dense layers
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.config.prediction_horizon)(x)
        
        return tf.keras.Model(inputs, outputs, name='informer_tcn')
    
    def _create_simple_model(self, input_shape: Tuple):
        """Create simple LSTM model"""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available for model creation")
        
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.config.prediction_horizon)
        ])
        return model
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Main training method with automatic fallback"""
        self.logger.info("Starting robust training process...")
        
        # Detect environment
        env_info = self.detect_environment()
        
        # Prepare data
        X, y = self.prepare_data(data)
        
        # Try strategies in order of priority
        for strategy in self.strategies:
            self.logger.info(f"Attempting strategy: {strategy['name']}")
            
            for attempt in range(self.config.max_retries):
                self.logger.info(f"Attempt {attempt + 1}/{self.config.max_retries}")
                
                result = self.train_with_strategy(X, y, strategy)
                
                if result.get('success', False):
                    self.logger.info(f"Training successful with strategy: {strategy['name']}")
                    
                    # Save results
                    self._save_training_results(result, env_info)
                    
                    return {
                        'success': True,
                        'strategy': strategy['name'],
                        'result': result,
                        'environment': env_info
                    }
                else:
                    self.logger.warning(f"Strategy {strategy['name']} attempt {attempt + 1} failed: {result.get('error', 'Unknown error')}")
                    time.sleep(1)  # Brief pause before retry
        
        # All strategies failed
        self.logger.error("All training strategies failed")
        return {
            'success': False,
            'error': 'All training strategies exhausted',
            'environment': env_info
        }
    
    def _save_training_results(self, result: Dict, env_info: Dict):
        """Save training results and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        if 'model' in result:
            model_path = f"{self.config.model_save_path}/model_{timestamp}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
            self.logger.info(f"Model saved to {model_path}")
        
        # Save training metadata
        metadata = {
            'timestamp': timestamp,
            'strategy': result.get('strategy'),
            'environment': env_info,
            'config': self.config.__dict__,
            'success': result.get('success', False)
        }
        
        metadata_path = f"{self.config.log_path}/training_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Training metadata saved to {metadata_path}")


# Example usage function
def run_robust_training_example():
    """Example of how to use the robust training system"""
    print("🚀 F1 Robust Training System - Example Usage")
    print("=" * 60)
    
    # Create configuration
    config = TrainingConfig(
        model_type="informer_tcn",
        batch_size=16,  # Smaller for stability
        epochs=50,      # Reduced for demo
        early_stopping_patience=10
    )
    
    # Initialize trainer
    trainer = RobustF1Trainer(config)
    
    # Create sample data (replace with actual F1 data)
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'speed_mean': np.random.normal(200, 50, 1000),
        'rpm_mean': np.random.normal(12000, 2000, 1000),
        'throttle_mean': np.random.normal(70, 20, 1000),
        'brake_mean': np.random.normal(30, 15, 1000),
        'engine_stress': np.random.uniform(0, 1, 1000),
        'lap_number': range(1000)
    })
    
    # Run training
    result = trainer.train(sample_data)
    
    if result['success']:
        print(f"✅ Training completed successfully using strategy: {result['strategy']}")
        print(f"📊 Environment: {result['environment']['platform']}")
        print(f"🔧 TensorFlow version: {result['environment']['tensorflow_version']}")
    else:
        print(f"❌ Training failed: {result['error']}")
    
    return result


if __name__ == "__main__":
    # Run example
    result = run_robust_training_example()