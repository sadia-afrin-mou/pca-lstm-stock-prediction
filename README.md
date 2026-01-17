# PCA-LSTM Stock Price Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive implementation of **Principal Component Analysis (PCA)** combined with **Long Short-Term Memory (LSTM)** networks for stock price prediction. This project demonstrates how dimensionality reduction can significantly improve model performance, training efficiency, and resource utilization.

---

## 📊 Key Results

| Metric | PCA-LSTM | Standard LSTM | Improvement |
|--------|----------|---------------|-------------|
| **MAE** | 120.8 | 339.6 | **64.4% ↓** |
| **Training Time** | 114.9s | 237.7s | **51.7% ↓** |
| **Memory Usage** | 4.9 MB | 6.3 MB | **22.2% ↓** |
| **Input Features** | 7 PCs | 10 features | **30% ↓** |

---

## 🎯 Project Overview

This project explores the effectiveness of PCA for dimensionality reduction in time series forecasting using LSTM networks. By reducing 10 correlated technical indicators to 7 principal components, we achieve:

- **Better prediction accuracy** (64% MAE improvement)
- **Faster training** (52% time reduction)
- **Lower memory footprint** (22% reduction)
- **Reduced overfitting risk** through feature compression

### Dataset
- **Stock**: Apple Inc. (AAPL)
- **Period**: 2015-01-01 to 2023-01-01 (8 years)
- **Features**: 10 technical indicators
- **Samples**: ~2000 trading days

---

## 📁 Repository Structure

```
├── pca_lstm_stock_forecast.ipynb    # Main notebook
├── technical-indicators.ipynb       # Technical indicators notebook
├── CML-presentation.pdf             # Project presentation slides
├── pca_lstm_model.keras             # Trained PCA-LSTM model
├── lstm_model.keras                 # Trained baseline model
├── scaler.pkl                       # Fitted scaler
├── pca.pkl                          # Fitted PCA transformer
└── README.md                        
```

---

## 📓 Notebooks

### 1. **pca_lstm_stock_forecast.ipynb** 
Main analysis notebook with complete pipeline:

- **Data Pipeline**: Reusable functions for data processing and LSTM training
- **Feature Engineering**: 10 technical indicators (MA50, SMA5, RSI, MACD, Bollinger Bands, ATR, ADX, STD10)
- **PCA Analysis**: Dimensionality reduction from 10 to 7 components
- **Model Training**: Both PCA-enhanced and standard LSTM models
- **Comprehensive Evaluation**: 
  - Training/validation loss curves
  - Prediction accuracy plots
  - PCA variance analysis
  - Component visualization
  - Feature loading heatmaps
  - Performance comparison

**Sections:**
1. Import Libraries
2. Functions & Data Pipeline
3. PCA-LSTM Training Results
4. PCA Explained Variance Analysis
5. PCA Component Visualization
6. Model Comparison: Without PCA
7. PCA Components Heatmap
8. Comprehensive PCA Analysis & Results
9. Summary & Conclusions

### 2. **technical-indicators.ipynb**
Educational notebook demonstrating technical indicators:

- **Moving Averages**: SMA, EMA
- **Momentum Indicators**: RSI, Stochastic Oscillator
- **Trend Indicators**: MACD, ADX
- **Volatility Indicators**: Bollinger Bands, ATR
- **Volume Indicators**: OBV, MFI

Each indicator includes:
- Python implementation
- Visualization



## 🔧 Installation

### Prerequisites
```bash
Python 3.9+
pip or conda package manager
```

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/sadia-afrin-mou/pca-lstm-stock-prediction.git
cd pca-lstm-stock-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Packages
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
yfinance>=0.1.70
ta>=0.10.0
joblib>=1.1.0
```

---

## 🚀 Quick Start

### Load Trained Model and Make Predictions

```python
import joblib
from tensorflow import keras
import numpy as np

# Load model and preprocessors
model = keras.models.load_model('pca_lstm_model.keras')
pca = joblib.load('pca.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare new data
new_data = load_stock_data('AAPL', '2023-01-01', '2024-01-01')
new_data = add_technical_indicators(new_data)
new_scaled = scaler.transform(new_data[FEATURES])
new_pca = pca.transform(new_scaled)

# Create sequences and predict
X_new, _, _ = create_sequences(new_pca, sequence_length=60)
predictions = model.predict(X_new)
```

### Train From Scratch

```python
# Open pca_lstm_stock_forecast.ipynb and run all cells
# Or use the command line:
jupyter notebook pca_lstm_stock_forecast.ipynb
```

---

## 🧠 Model Architecture

```
Input Layer: (60, 7) for PCA / (60, 10) for Standard
│
├─ LSTM Layer 1: 128 units
│  ├─ BatchNormalization
│  ├─ Dropout(0.2)
│  └─ L2 Regularization(0.001)
│
├─ LSTM Layer 2: 64 units
│  ├─ BatchNormalization
│  ├─ Dropout(0.2)
│  └─ L2 Regularization(0.001)
│
├─ LSTM Layer 3: 32 units
│  ├─ BatchNormalization
│  ├─ Dropout(0.2)
│  └─ L2 Regularization(0.001)
│
├─ Dense Layer: 16 units (ReLU)
│  └─ L2 Regularization(0.001)
│
└─ Output Layer: 1 unit (linear)

Loss: Huber (delta=1.0)
Optimizer: Adam (learning_rate=0.0005)
```

### Training Configuration
- **Sequence Length**: 60 days
- **Batch Size**: 64
- **Max Epochs**: 150
- **Early Stopping**: patience=20, min_delta=0.0001
- **Learning Rate Reduction**: factor=0.5, patience=10

 ---

## 📊 Visualizations

The project includes comprehensive visualizations:

1. **Training Loss Curves**: Monitor convergence and overfitting
2. **Prediction Plots**: Actual vs predicted prices
3. **Explained Variance**: Individual and cumulative variance by PC
4. **Scree Plot**: Optimal number of components
5. **PCA Biplot**: Data distribution in PC1-PC2 space
6. **Component Loadings Heatmap**: Feature contributions to PCs
7. **Reconstruction Error Distribution**: PCA quality assessment
8. **Performance Comparison**: PCA vs No-PCA metrics

---

## 🎯 Key Insights

### Why PCA Works Well

1. **High Correlation**: Technical indicators are inherently correlated
   - Trend indicators move together (MA50, Close, Bollinger Bands)
   - Momentum indicators align (RSI, MACD)

2. **Noise Reduction**: PCA filters redundant information
   - Retains 96% variance with 4 components
   - Removes multicollinearity

3. **Regularization Effect**: Fewer features reduce overfitting
   - 30% feature reduction
   - Better generalization

4. **Computational Efficiency**
   - 52% faster training
   - 22% less memory

### When to Use PCA-LSTM

✅ **Recommended for:**
- High-dimensional feature spaces (>10 features)
- Correlated feature sets
- Resource-constrained environments
- Time series with multiple correlated indicators

❌ **Not recommended for:**
- Uncorrelated features
- Small feature spaces (<5 features)
- Applications requiring feature interpretability
- Strong non-linear relationships (use Kernel PCA)

---

## 🔬 Reproducibility

All experiments use fixed random seeds for reproducibility:

```python
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'
```

---

## 📚 Technical Indicators Used

| Indicator | Type | Formula/Description |
|-----------|------|---------------------|
| **Close** | Price | Daily closing price |
| **MA50** | Trend | 50-day moving average |
| **SMA5** | Trend | 5-day simple moving average |
| **RSI** | Momentum | Relative Strength Index (14-period) |
| **MACD** | Momentum | Moving Average Convergence Divergence |
| **Bollinger_Upper** | Volatility | MA10 + 2×STD10 |
| **Bollinger_Lower** | Volatility | MA10 - 2×STD10 |
| **ATR** | Volatility | Average True Range (14-period) |
| **ADX** | Trend Strength | Average Directional Index (14-period) |
| **STD10** | Volatility | 10-day standard deviation |

---

## 🚧 Future Work

### Model Enhancements
- [ ] Hyperparameter optimization (Bayesian/Grid Search)
- [ ] Alternative architectures (GRU, Bidirectional LSTM, Attention)
- [ ] Ensemble methods (stacking, voting)
- [ ] Transfer learning across different stocks

### Feature Engineering
- [ ] Additional technical indicators
- [ ] Sentiment analysis from news/social media
- [ ] Market regime indicators (VIX, economic calendars)
- [ ] Multi-timeframe analysis

### Dimensionality Reduction
- [ ] Kernel PCA for non-linear relationships
- [ ] Autoencoders for deep feature learning
- [ ] Feature selection via importance ranking
- [ ] t-SNE/UMAP for visualization

---

## 📖 References

1. Jolliffe, I. T. (2002). *Principal Component Analysis*. Springer Series in Statistics.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.
3. Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*. New York Institute of Finance.
4. Box, G. E. P., & Jenkins, G. M. (2015). *Time Series Analysis: Forecasting and Control*. Wiley.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.


## 🌟 Citation

If you use this code in your research, please cite:

```bibtex
@misc{pca_lstm_stock_prediction,
  title={PCA-Enhanced LSTM for Stock Price Prediction},
  author={Sadia Afrin Mou},
  year={2025},
  url={https://github.com/sadia-afrin-mou/pca-lstm-stock-prediction.git}
}
```

