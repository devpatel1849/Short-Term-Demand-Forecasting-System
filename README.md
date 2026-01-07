# Advanced Manufacturing Plant Demand Forecasting System

## 🎯 Project Overview

**Objective**: AI-driven short-term demand forecasting for manufacturing plants to optimize raw material procurement, inventory management, production planning, and working capital.

**Business Impact**: 
- Reduce inventory holding costs by 15-25%
- Minimize stockouts and production delays
- Optimize raw material procurement timing
- Improve working capital efficiency

## 🏭 Manufacturing Context

This system addresses critical challenges in manufacturing demand forecasting:

1. **Inventory Optimization**: Accurate forecasts prevent excess inventory and stockouts
2. **Raw Material Planning**: Predict material requirements with lead time considerations
3. **Production Scheduling**: Align production capacity with demand forecasts
4. **Supply Chain Management**: Account for disruptions and supplier variability
5. **Financial Planning**: Optimize working capital and procurement budgets

## 📊 System Architecture

### Data Sources
- **Manufacturing Demand Data**: Historical production demand by plant/product
- **Raw Material Master**: Material specifications, costs, lead times
- **Plant Capacity**: Production capabilities and constraints
- **Economic Indicators**: Market conditions affecting demand
- **Supply Chain Events**: Disruptions and operational factors

### Key Features
- **Multi-Plant Forecasting**: 4 manufacturing plants across India
- **Multi-Product Support**: 12 product lines across 4 categories
- **Raw Material Planning**: Automatic requirement calculation
- **Economic Integration**: Market indicators impact modeling
- **Supply Chain Risk**: Disruption event modeling
- **Interactive Dashboard**: Real-time insights and alerts

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git LFS installed

### ⚠️ Important: Large Model Files
Due to GitHub's file size limitations, some large model files (>50MB) are stored separately:
- Download complete models from: [Google Drive](https://drive.google.com/drive/folders/1vjQb8DJ7Ze4BCeWl9arMJAG67twUrnVt?usp=drive_link)
- Or let the app automatically train models on first run

### Installation
```bash
# Clone the repository
git clone https://github.com/ShivamPatel145/manufacturing-forecasting.git
cd manufacturing-forecasting

# Install requirements
pip install -r requirements.txt

# Run the dashboard
streamlit run manufacturing_app.py
```

Access the dashboard at: `http://localhost:8501`

## 📊 Features

- **Multi-Plant Forecasting**: Demand prediction for 4 manufacturing plants
- **Raw Material Planning**: Automatic procurement requirements calculation
- **Interactive Dashboard**: Real-time insights and visualizations
- **Model Ensemble**: Advanced ML algorithms (Random Forest, Gradient Boosting)
- **Performance Analytics**: Plant and product category analysis

## 📁 Project Structure

```
├── data/                          # Sample manufacturing data
├── models/                        # Trained ML models
├── notebooks/                     # Jupyter notebook for analysis
├── outputs/                       # Generated reports and forecasts
├── manufacturing_app.py           # Main Streamlit dashboard
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 📈 Using the Dashboard

The dashboard contains several tabs:

1. **Demand Trends**: View historical and forecasted demand
2. **Raw Materials**: Procurement planning and cost analysis
3. **Plant Performance**: Individual plant metrics and capacity utilization
4. **Product Analysis**: Product category performance comparison
5. **Alerts & Insights**: Business recommendations and alerts

## 🔍 Advanced Analysis

For detailed analysis, explore the Jupyter notebook:
```bash
# Open the advanced forecasting notebook
jupyter notebook notebooks/02_advanced_manufacturing_forecasting.ipynb
```

## 📋 Requirements

The system requires the following Python packages:
- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- plotly
- seaborn

## 🎯 Expected Results

- **Forecast Accuracy**: 85-92% overall accuracy
- **Plant-level Performance**: <15% MAPE per plant
- **Business Impact**: 15-25% inventory cost reduction potential

---

*Built for manufacturing demand forecasting and inventory optimization*

---

*Built with advanced machine learning techniques, this system represents the cutting edge of manufacturing demand forecasting technology.*

PROJECT URL:- https://short-term-demand-forecasting-system.streamlit.app/
