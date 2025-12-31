"""
Advanced ML Models Module
Multi-model ensemble system for manufacturing demand forecasting
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
try:
    from sklearn.metrics import mean_absolute_percentage_error
except ImportError:
    # For older sklearn versions
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from datetime import datetime, timedelta
import joblib
import json

class AdvancedMLEnsemble:
    """
    Advanced ML ensemble system with multiple algorithms
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        self.is_trained = False
        
        # Model availability
        self.available_models = {
            'random_forest': True,
            'xgboost': XGBOOST_AVAILABLE,
            'prophet': PROPHET_AVAILABLE,
            'lstm': TENSORFLOW_AVAILABLE,
            'ensemble': True
        }
        
        print("🤖 Advanced ML Models Available:")
        for model, available in self.available_models.items():
            status = "✅" if available else "❌"
            print(f"   {status} {model.upper()}")
    
    def prepare_features(self, data):
        """
        Advanced feature engineering for time series forecasting
        """
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['plant_id', 'product_category', 'date'])
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        
        # Cyclical encoding for time features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Lag features (grouped by plant and category)
        for group_cols in [['plant_id', 'product_category']]:
            for lag in [1, 2, 3, 7, 14, 30]:
                col_name = f'demand_lag_{lag}'
                df[col_name] = df.groupby(group_cols)['demand_units'].shift(lag)
        
        # Rolling statistics
        for window in [3, 7, 14, 30]:
            for stat in ['mean', 'std', 'min', 'max']:
                col_name = f'demand_rolling_{stat}_{window}'
                df[col_name] = df.groupby(['plant_id', 'product_category'])['demand_units'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).agg(stat)
                )
        
        # Exponential weighted moving average
        for alpha in [0.1, 0.3, 0.5]:
            col_name = f'demand_ewm_{alpha}'
            df[col_name] = df.groupby(['plant_id', 'product_category'])['demand_units'].transform(
                lambda x: x.ewm(alpha=alpha).mean()
            )
        
        # Difference features (trend indicators)
        df['demand_diff_1'] = df.groupby(['plant_id', 'product_category'])['demand_units'].diff(1)
        df['demand_diff_7'] = df.groupby(['plant_id', 'product_category'])['demand_units'].diff(7)
        
        # Categorical encoding
        if 'plant_id' not in self.encoders:
            self.encoders['plant_id'] = LabelEncoder()
            df['plant_encoded'] = self.encoders['plant_id'].fit_transform(df['plant_id'])
        else:
            df['plant_encoded'] = self.encoders['plant_id'].transform(df['plant_id'])
        
        if 'product_category' not in self.encoders:
            self.encoders['product_category'] = LabelEncoder()
            df['category_encoded'] = self.encoders['product_category'].fit_transform(df['product_category'])
        else:
            df['category_encoded'] = self.encoders['product_category'].transform(df['product_category'])
        
        # Interaction features
        df['plant_category_interaction'] = df['plant_encoded'] * df['category_encoded']
        df['month_category_interaction'] = df['month'] * df['category_encoded']
        
        return df
    
    def create_lstm_model(self, n_features, n_timesteps=30):
        """
        Create LSTM model for time series forecasting
        """
        if not TENSORFLOW_AVAILABLE:
            return None
        
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(n_timesteps, n_features)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae'])
        
        return model
    
    def prepare_lstm_data(self, data, n_timesteps=30):
        """
        Prepare data for LSTM model
        """
        sequences = []
        targets = []
        
        # Group by plant and category for sequence creation
        for (plant, category), group in data.groupby(['plant_id', 'product_category']):
            group = group.sort_values('date')
            values = group['demand_units'].values
            
            if len(values) >= n_timesteps + 1:
                for i in range(n_timesteps, len(values)):
                    sequences.append(values[i-n_timesteps:i])
                    targets.append(values[i])
        
        return np.array(sequences), np.array(targets)
    
    def train_random_forest(self, X_train, y_train, X_val, y_val, optimize_hyperparams=True):
        """
        Train Random Forest with optional hyperparameter tuning
        """
        print("🌲 Training Random Forest...")
        
        if not optimize_hyperparams:
            # Quick mode: use default good parameters
            best_params = {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        else:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            
            # Use TimeSeriesSplit for cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            if OPTUNA_AVAILABLE:
                # Use Optuna for optimization if available
                study = optuna.create_study(direction='minimize', 
                                           sampler=optuna.samplers.TPESampler(seed=42))
                
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                        'max_depth': trial.suggest_int('max_depth', 10, 30),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
                    }
                    
                    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_val)
                    return mean_absolute_error(y_val, predictions)
                
                study.optimize(objective, n_trials=10, timeout=60)  # 1 minute max
                best_params = study.best_params
            else:
                # Use GridSearchCV as fallback
                grid_search = GridSearchCV(rf, param_grid, cv=tscv, 
                                         scoring='neg_mean_absolute_error', 
                                         n_jobs=-1, verbose=0)
                grid_search.fit(X_train, y_train)
                best_params = grid_search.best_params_
        
        # Train final model
        best_rf = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        best_rf.fit(X_train, y_train)
        
        self.models['random_forest'] = best_rf
        
        # Feature importance
        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
        self.feature_importance['random_forest'] = dict(zip(feature_names, best_rf.feature_importances_))
        
        return best_rf
    
    def train_xgboost(self, X_train, y_train, X_val, y_val, optimize_hyperparams=True):
        """
        Train XGBoost with optional hyperparameter tuning
        """
        if not XGBOOST_AVAILABLE:
            print("❌ XGBoost not available")
            return None
        
        print("🚀 Training XGBoost...")
        
        if not optimize_hyperparams:
            # Quick mode: use default good parameters
            best_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        elif OPTUNA_AVAILABLE:
            study = optuna.create_study(direction='minimize')
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
                }
                
                model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
                try:
                    # Try new XGBoost API (fit with callbacks)
                    model.fit(X_train, y_train, 
                             eval_set=[(X_val, y_val)], 
                             verbose=False)
                except Exception:
                    # Fallback to basic fit without early stopping
                    model.fit(X_train, y_train, verbose=False)
                
                predictions = model.predict(X_val)
                return mean_absolute_error(y_val, predictions)
            
            study.optimize(objective, n_trials=15, timeout=120)  # 2 minutes max
            best_params = study.best_params
        else:
            # Default parameters
            best_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        
        # Train final model
        best_xgb = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        try:
            # Try new XGBoost API (fit with callbacks)
            best_xgb.fit(X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False)
        except Exception:
            # Fallback to basic fit without early stopping
            best_xgb.fit(X_train, y_train, verbose=False)
        
        self.models['xgboost'] = best_xgb
        
        # Feature importance
        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
        self.feature_importance['xgboost'] = dict(zip(feature_names, best_xgb.feature_importances_))
        
        return best_xgb
    
    def train_prophet(self, data):
        """
        Train Prophet model for time series forecasting
        """
        if not PROPHET_AVAILABLE:
            print("❌ Prophet not available")
            return None
        
        print("📈 Training Prophet...")
        
        prophet_models = {}
        
        # Train separate model for each plant-category combination
        for (plant, category), group in data.groupby(['plant_id', 'product_category']):
            if len(group) < 30:  # Need sufficient data for Prophet
                continue
            
            # Prepare Prophet data format
            prophet_data = group[['date', 'demand_units']].copy()
            prophet_data.columns = ['ds', 'y']
            prophet_data = prophet_data.sort_values('ds')
            
            # Create and fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                interval_width=0.95
            )
            
            try:
                model.fit(prophet_data)
                prophet_models[f"{plant}_{category}"] = model
            except Exception as e:
                print(f"⚠️ Prophet training failed for {plant}_{category}: {e}")
                continue
        
        self.models['prophet'] = prophet_models
        return prophet_models
    
    def train_lstm(self, data):
        """
        Train LSTM model
        """
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow/LSTM not available")
            return None
        
        print("🧠 Training LSTM...")
        
        # Prepare sequences for LSTM
        X_seq, y_seq = self.prepare_lstm_data(data)
        
        if len(X_seq) < 100:  # Need sufficient data
            print("⚠️ Insufficient data for LSTM training")
            return None
        
        # Reshape for LSTM (samples, timesteps, features)
        X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42
        )
        
        # Create and train LSTM model
        lstm_model = self.create_lstm_model(n_features=1)
        
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        history = lstm_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.models['lstm'] = lstm_model
        return lstm_model
    
    def train_ensemble(self, X_train, y_train, X_val, y_val, data, optimize_hyperparams=True):
        """
        Train all available models and create ensemble
        """
        print("🎯 Training Advanced ML Ensemble...")
        
        individual_predictions = {}
        
        # Train Random Forest
        rf_model = self.train_random_forest(X_train, y_train, X_val, y_val, optimize_hyperparams)
        if rf_model:
            rf_pred = rf_model.predict(X_val)
            individual_predictions['random_forest'] = rf_pred
            self.model_performance['random_forest'] = {
                'mae': mean_absolute_error(y_val, rf_pred),
                'mse': mean_squared_error(y_val, rf_pred),
                'mape': mean_absolute_percentage_error(y_val, rf_pred) * 100
            }
        
        # Train XGBoost
        if XGBOOST_AVAILABLE:
            xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val, optimize_hyperparams)
            if xgb_model:
                xgb_pred = xgb_model.predict(X_val)
                individual_predictions['xgboost'] = xgb_pred
                self.model_performance['xgboost'] = {
                    'mae': mean_absolute_error(y_val, xgb_pred),
                    'mse': mean_squared_error(y_val, xgb_pred),
                    'mape': mean_absolute_percentage_error(y_val, xgb_pred) * 100
                }
        
        # Train Prophet
        if PROPHET_AVAILABLE:
            prophet_models = self.train_prophet(data)
            # Prophet predictions would need special handling for validation
        
        # Train LSTM
        if TENSORFLOW_AVAILABLE:
            lstm_model = self.train_lstm(data)
            # LSTM predictions would need special handling for validation
        
        # Create ensemble weights based on performance
        if len(individual_predictions) > 1:
            # Calculate weights inversely proportional to MAE
            weights = {}
            total_weight = 0
            
            for model_name, pred in individual_predictions.items():
                mae = self.model_performance[model_name]['mae']
                weight = 1 / (mae + 1e-8)  # Avoid division by zero
                weights[model_name] = weight
                total_weight += weight
            
            # Normalize weights
            for model_name in weights:
                weights[model_name] /= total_weight
            
            # Create ensemble prediction
            ensemble_pred = np.zeros_like(list(individual_predictions.values())[0])
            for model_name, pred in individual_predictions.items():
                ensemble_pred += weights[model_name] * pred
            
            self.model_performance['ensemble'] = {
                'mae': mean_absolute_error(y_val, ensemble_pred),
                'mse': mean_squared_error(y_val, ensemble_pred),
                'mape': mean_absolute_percentage_error(y_val, ensemble_pred) * 100,
                'weights': weights
            }
            
            print("✅ Ensemble training completed!")
            print("📊 Model Performance Summary:")
            for model_name, metrics in self.model_performance.items():
                if model_name != 'ensemble':
                    print(f"   {model_name}: MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.1f}%")
            
            if 'ensemble' in self.model_performance:
                ens_metrics = self.model_performance['ensemble']
                print(f"   🎯 ENSEMBLE: MAE={ens_metrics['mae']:.2f}, MAPE={ens_metrics['mape']:.1f}%")
        
        self.is_trained = True
        return True
    
    def fit(self, data, optimize_hyperparams=True):
        """
        Main training method
        """
        mode = "🚀 Fast Mode" if not optimize_hyperparams else "🔬 Full Optimization Mode"
        print(f"{mode} - Starting Advanced ML Training Pipeline...")
        
        # Prepare features
        processed_data = self.prepare_features(data)
        
        # Remove rows with NaN values (from lag features)
        processed_data = processed_data.dropna()
        
        if len(processed_data) < 100:
            raise ValueError("Insufficient data for training. Need at least 100 samples.")
        
        # Prepare features for traditional ML models
        feature_columns = [col for col in processed_data.columns 
                          if col not in ['date', 'plant_id', 'product_category', 'product_line', 'demand_units']]
        
        X = processed_data[feature_columns]
        y = processed_data['demand_units']
        
        # Split data (time series aware)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scalers['features'].fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            self.scalers['features'].transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        # Train ensemble
        self.train_ensemble(X_train_scaled, y_train, X_val_scaled, y_val, processed_data, optimize_hyperparams)
        
        return self
    
    def predict(self, data, model_type='ensemble'):
        """
        Make predictions using specified model or ensemble
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Prepare features
        processed_data = self.prepare_features(data)
        
        # Get feature columns
        feature_columns = [col for col in processed_data.columns 
                          if col not in ['date', 'plant_id', 'product_category', 'product_line', 'demand_units']]
        
        X = processed_data[feature_columns]
        
        # Handle missing features gracefully
        for col in self.scalers['features'].feature_names_in_:
            if col not in X.columns:
                X[col] = 0  # Default value for missing features
        
        # Reorder columns to match training
        X = X.reindex(columns=self.scalers['features'].feature_names_in_, fill_value=0)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scalers['features'].transform(X),
            columns=X.columns,
            index=X.index
        )
        
        if model_type == 'ensemble' and len(self.model_performance) > 1:
            # Ensemble prediction
            ensemble_pred = np.zeros(len(X_scaled))
            
            if 'ensemble' in self.model_performance and 'weights' in self.model_performance['ensemble']:
                weights = self.model_performance['ensemble']['weights']
                
                for model_name, weight in weights.items():
                    if model_name in self.models and self.models[model_name] is not None:
                        model_pred = self.models[model_name].predict(X_scaled)
                        ensemble_pred += weight * model_pred
            else:
                # Simple average if weights not available
                model_count = 0
                for model_name in ['random_forest', 'xgboost']:
                    if model_name in self.models and self.models[model_name] is not None:
                        model_pred = self.models[model_name].predict(X_scaled)
                        ensemble_pred += model_pred
                        model_count += 1
                
                if model_count > 0:
                    ensemble_pred /= model_count
            
            return ensemble_pred
        
        elif model_type in self.models and self.models[model_type] is not None:
            return self.models[model_type].predict(X_scaled)
        
        else:
            raise ValueError(f"Model {model_type} not available or not trained")
    
    def get_feature_importance(self, model_type='random_forest', top_n=15):
        """
        Get feature importance for specified model
        """
        if model_type not in self.feature_importance:
            return {}
        
        importance = self.feature_importance[model_type]
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_importance[:top_n])
    
    def save_models(self, filepath):
        """
        Save trained models
        """
        model_data = {
            'models': {},
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_importance': self.feature_importance,
            'model_performance': self.model_performance,
            'is_trained': self.is_trained
        }
        
        # Save sklearn models
        for model_name in ['random_forest', 'xgboost']:
            if model_name in self.models and self.models[model_name] is not None:
                joblib.dump(self.models[model_name], f"{filepath}_{model_name}.pkl")
                model_data['models'][model_name] = f"{filepath}_{model_name}.pkl"
        
        # Save TensorFlow models
        if 'lstm' in self.models and self.models['lstm'] is not None:
            self.models['lstm'].save(f"{filepath}_lstm.h5")
            model_data['models']['lstm'] = f"{filepath}_lstm.h5"
        
        # Save metadata
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(model_data, f, default=str)
        
        print(f"✅ Models saved to {filepath}")
    
    def load_models(self, filepath):
        """
        Load trained models
        """
        # Load metadata
        with open(f"{filepath}_metadata.json", 'r') as f:
            model_data = json.load(f)
        
        self.scalers = model_data['scalers']
        self.encoders = model_data['encoders']
        self.feature_importance = model_data['feature_importance']
        self.model_performance = model_data['model_performance']
        self.is_trained = model_data['is_trained']
        
        # Load models
        for model_name, model_path in model_data['models'].items():
            if model_name in ['random_forest', 'xgboost']:
                self.models[model_name] = joblib.load(model_path)
            elif model_name == 'lstm' and TENSORFLOW_AVAILABLE:
                from tensorflow.keras.models import load_model
                self.models[model_name] = load_model(model_path)
        
        print(f"✅ Models loaded from {filepath}")
