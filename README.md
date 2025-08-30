# Health-vitals-pipeline
Neural network pipeline for healthcare anomaly detection. Uses autoencoder architecture to learn normal vital sign patterns and detect medical emergencies from patient monitoring data.


##  Overview

This pipeline uses an autoencoder architecture to:
- **Learn normal patterns** from patient vital signs (heart rate, blood pressure, oxygen saturation)
- **Detect anomalies** by measuring reconstruction errors
- **Flag potential emergencies** when vital signs deviate significantly from learned normal patterns

##  Features

- **Autoencoder-based anomaly detection** using PyTorch
- **Prefect workflow orchestration** for reliable pipeline execution
- **Standardized data preprocessing** with scikit-learn StandardScaler
- **Configurable training parameters** (epochs, learning rate, model architecture)
- **Comprehensive evaluation metrics** (precision, recall, F1-score)
- **Batch scoring capabilities** for new patient data

##  Architecture
Input Data → Preprocessing → Autoencoder Training → Threshold Setting → Anomaly Detection

CSV Files StandardScaler Neural Network Percentile-based Scored Results


## 📁 Project Structure

Health-vitals-pipeline/
├── src/
│ ├── data.py # Data loading and preprocessing
│ ├── model.py # Autoencoder model and training
│ ├── eval.py # Evaluation and scoring
│ └── pipeline.py # Prefect workflow orchestration
├── data/
│ ├── .zip # Dataset archives (see Data section)
│ └── scored/ # Auto-generated results
├── models/ # Saved model artifacts
│ ├── autoencoder.pt # Trained model weights
│ ├── scaler.joblib # Fitted StandardScaler
│ └── threshold.json # Anomaly detection threshold
├── requirements.txt # Python dependencies
└── README.md # This file


## Configuration Information

Training Parameters
Edit `src/pipeline.py` to modify:
```python
cfg = TrainingConfig(
    epochs=5,           # Training epochs
    batch_size=32,      # Batch size
    lr=0.001,          # Learning rate
    hidden_dim=8,      # Hidden layer dimension
    bottleneck=2       # Bottleneck dimension
)
```

Threshold Tuning
Edit `src/model.py` to adjust anomaly sensitivity:
```python
th = percentile_threshold(errs, 90.5)  # 90.5th percentile
```
## Acknowledgments

- PyTorch for deep learning framework
- Prefect for workflow orchestration
- Scikit-learn for data preprocessing
- Healthcare data providers for datasets


## Note : This pipeline is designed for educational purpose only. 