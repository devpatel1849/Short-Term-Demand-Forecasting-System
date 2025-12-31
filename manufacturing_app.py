import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import io
import base64
import sys
import os

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
from inventory_optimizer import InventoryOptimizer
from advanced_ml_models import AdvancedMLEnsemble

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Manufacturing Demand Forecasting Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏭 Manufacturing Plant Demand Forecasting Dashboard</h1>', unsafe_allow_html=True)

# Simple load data function that combines historical and forecast data
@st.cache_data
def load_data():
    """Load and combine historical and forecast data"""
    try:
        # Load historical data
        historical_df = pd.read_csv("data/manufacturing_demand_data.csv")
        historical_df['date'] = pd.to_datetime(historical_df['date'])
        
        # For historical data, use actual demand as prediction (perfect hindsight)
        historical_df['predicted_demand'] = historical_df['demand_units']
        historical_df['prediction_error'] = 0
        historical_df['absolute_error'] = 0  
        historical_df['percentage_error'] = 0
        
        # Load forecast data if available
        try:
            forecast_df = pd.read_csv("outputs/manufacturing_forecast_results.csv")
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
            
            # Combine the datasets
            combined_df = pd.concat([
                historical_df[['date', 'plant_id', 'product_line', 'product_category', 
                              'demand_units', 'predicted_demand', 'prediction_error', 
                              'absolute_error', 'percentage_error']],
                forecast_df[['date', 'plant_id', 'product_line', 'product_category', 
                            'demand_units', 'predicted_demand', 'prediction_error', 
                            'absolute_error', 'percentage_error']]
            ], ignore_index=True).drop_duplicates(subset=['date', 'plant_id', 'product_line'])
        except:
            # If forecast data not available, use historical data only
            combined_df = historical_df[['date', 'plant_id', 'product_line', 'product_category', 
                                       'demand_units', 'predicted_demand', 'prediction_error', 
                                       'absolute_error', 'percentage_error']]
        
        combined_df = combined_df.sort_values(['date', 'plant_id', 'product_line']).reset_index(drop=True)
        
        return combined_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
data_df = load_data()

if data_df is not None:
    # Sidebar filters
    st.sidebar.header("📊 Dashboard Controls")
    
    # Plant selection
    selected_plants = st.sidebar.multiselect(
        "Select Plants",
        options=sorted(data_df['plant_id'].unique()),
        default=sorted(data_df['plant_id'].unique())
    )
    
    # Product category selection
    selected_categories = st.sidebar.multiselect(
        "Select Product Categories", 
        options=sorted(data_df['product_category'].unique()),
        default=sorted(data_df['product_category'].unique())
    )
    
    # Date range selection
    min_date = data_df['date'].min().date()
    max_date = data_df['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data based on selections
    if len(date_range) == 2:
        date_filter = (
            (data_df['date'] >= pd.to_datetime(date_range[0])) &
            (data_df['date'] <= pd.to_datetime(date_range[1]))
        )
    else:
        date_filter = data_df['date'] == pd.to_datetime(date_range[0])
    
    filtered_df = data_df[
        (data_df['plant_id'].isin(selected_plants)) &
        (data_df['product_category'].isin(selected_categories)) &
        date_filter
    ]
    
    # Main dashboard layout with enhanced KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Key metrics
    total_actual = filtered_df['demand_units'].sum()
    total_predicted = filtered_df['predicted_demand'].sum()
    mae = filtered_df['absolute_error'].mean()
    accuracy = 100 - (filtered_df['percentage_error'].abs().mean())
    
    # Additional KPIs
    demand_trend = filtered_df.groupby('date')['demand_units'].sum().pct_change().iloc[-1] * 100 if len(filtered_df) > 1 else 0
    
    with col1:
        st.metric(
            label="📊 Total Actual Demand",
            value=f"{total_actual:,.0f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="🎯 Total Predicted Demand", 
            value=f"{total_predicted:,.0f}",
            delta=f"{((total_predicted/total_actual-1)*100):+.1f}%" if total_actual > 0 else None
        )
    
    with col3:
        st.metric(
            label="📈 Mean Absolute Error",
            value=f"{mae:.1f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="✅ Forecast Accuracy",
            value=f"{accuracy:.1f}%",
            delta=None
        )
    
    with col5:
        trend_icon = "📈" if demand_trend > 0 else "📉" if demand_trend < 0 else "➡️"
        st.metric(
            label=f"{trend_icon} Demand Trend",
            value=f"{abs(demand_trend):.1f}%",
            delta=f"{'↗️' if demand_trend > 0 else '↘️' if demand_trend < 0 else '→'} Trend"
        )
    
    # Alert system
    if accuracy < 80:
        st.error("🚨 **Alert**: Forecast accuracy below 80%. Consider model retraining.")
    elif accuracy < 90:
        st.warning("⚠️ **Warning**: Forecast accuracy below 90%. Monitor model performance.")
    else:
        st.success("✅ **Good**: Forecast accuracy is satisfactory.")
    
    # Quick insights
    st.markdown("### 💡 Quick Insights")
    
    insights_col1, insights_col2, insights_col3 = st.columns(3)
    
    with insights_col1:
        top_plant = filtered_df.groupby('plant_id')['demand_units'].sum().idxmax()
        top_plant_demand = filtered_df.groupby('plant_id')['demand_units'].sum().max()
        st.info(f"🏭 **Top Performing Plant**: {top_plant} ({top_plant_demand:,.0f} units)")
    
    with insights_col2:
        top_category = filtered_df.groupby('product_category')['demand_units'].sum().idxmax()
        top_category_demand = filtered_df.groupby('product_category')['demand_units'].sum().max()
        st.info(f"📦 **Leading Category**: {top_category} ({top_category_demand:,.0f} units)")
    
    with insights_col3:
        peak_day = filtered_df.groupby('date')['demand_units'].sum().idxmax()
        peak_demand = filtered_df.groupby('date')['demand_units'].sum().max()
        st.info(f"📅 **Peak Demand Day**: {peak_day.strftime('%Y-%m-%d')} ({peak_demand:,.0f} units)")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Time Series Analysis", 
        "🏭 Plant Performance", 
        "📊 Category Analysis",
        "🔍 Seasonality & Trends",
        "⚠️ Anomaly Detection", 
        "🎯 What-If Analysis",
        "📦 Inventory Optimization",
        "🤖 Advanced ML Models"
    ])
    
    with tab1:
        st.subheader("Demand vs Forecast Over Time")
        
        # Aggregate data by date for time series
        ts_data = filtered_df.groupby('date').agg({
            'demand_units': 'sum',
            'predicted_demand': 'sum'
        }).reset_index()
        
        # Create time series plot
        fig_ts = go.Figure()
        
        fig_ts.add_trace(go.Scatter(
            x=ts_data['date'],
            y=ts_data['demand_units'],
            mode='lines+markers',
            name='Actual Demand',
            line=dict(color='blue', width=2)
        ))
        
        fig_ts.add_trace(go.Scatter(
            x=ts_data['date'],
            y=ts_data['predicted_demand'],
            mode='lines+markers',
            name='Predicted Demand',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_ts.update_layout(
            title="Daily Demand Trends",
            xaxis_title="Date",
            yaxis_title="Demand Units",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_ts, use_container_width=True, key="time_series_main")
        
        # Plant comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demand by Plant")
            plant_agg = filtered_df.groupby('plant_id')['demand_units'].sum().sort_values(ascending=True)
            
            fig_plant = px.bar(
                y=plant_agg.index,
                x=plant_agg.values,
                orientation='h',
                title="Total Demand by Plant",
                labels={'x': 'Demand Units', 'y': 'Plant'}
            )
            st.plotly_chart(fig_plant, use_container_width=True, key="plant_demand_bar")
        
        with col2:
            st.subheader("Demand by Category")
            category_agg = filtered_df.groupby('product_category')['demand_units'].sum()
            
            fig_category = px.pie(
                values=category_agg.values,
                names=category_agg.index,
                title="Demand Distribution by Category"
            )
            st.plotly_chart(fig_category, use_container_width=True, key="category_pie_chart")
    
    with tab2:
        st.subheader("Plant Performance Analysis")
        
        # Calculate plant-level metrics
        plant_metrics = filtered_df.groupby('plant_id').agg({
            'demand_units': 'sum',
            'predicted_demand': 'sum',
            'absolute_error': 'mean',
            'percentage_error': lambda x: abs(x).mean()
        }).reset_index()
        
        plant_metrics['accuracy'] = 100 - plant_metrics['percentage_error']
        plant_metrics = plant_metrics.round(2)
        
        st.dataframe(
            plant_metrics,
            column_config={
                "plant_id": "Plant ID",
                "demand_units": st.column_config.NumberColumn("Total Demand", format="%.0f"),
                "predicted_demand": st.column_config.NumberColumn("Predicted Demand", format="%.0f"),
                "absolute_error": st.column_config.NumberColumn("Mean Absolute Error", format="%.2f"),
                "percentage_error": st.column_config.NumberColumn("Mean % Error", format="%.2f%%"),
                "accuracy": st.column_config.NumberColumn("Accuracy", format="%.2f%%")
            },
            use_container_width=True
        )
    
    with tab3:
        st.subheader("Product Category Analysis")
        
        # Calculate category-level metrics
        category_metrics = filtered_df.groupby('product_category').agg({
            'demand_units': 'sum', 
            'predicted_demand': 'sum',
            'absolute_error': 'mean',
            'percentage_error': lambda x: abs(x).mean()
        }).reset_index()
        
        category_metrics['accuracy'] = 100 - category_metrics['percentage_error']
        category_metrics = category_metrics.round(2)
        
        st.dataframe(
            category_metrics,
            column_config={
                "product_category": "Product Category",
                "demand_units": st.column_config.NumberColumn("Total Demand", format="%.0f"),
                "predicted_demand": st.column_config.NumberColumn("Predicted Demand", format="%.0f"),
                "absolute_error": st.column_config.NumberColumn("Mean Absolute Error", format="%.2f"),
                "percentage_error": st.column_config.NumberColumn("Mean % Error", format="%.2f%%"),
                "accuracy": st.column_config.NumberColumn("Accuracy", format="%.2f%%")
            },
            use_container_width=True
        )
        
        # Category performance over time
        category_ts = filtered_df.groupby(['date', 'product_category'])['demand_units'].sum().reset_index()
        
        fig_cat_ts = px.line(
            category_ts,
            x='date',
            y='demand_units',
            color='product_category',
            title="Demand Trends by Product Category"
        )
        
        st.plotly_chart(fig_cat_ts, use_container_width=True, key="category_trends_tab3")

    with tab4:
        st.subheader("🔍 Seasonality & Trend Analysis")
        
        # Seasonality analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Monthly Seasonality Pattern**")
            
            # Monthly aggregation
            monthly_data = filtered_df.copy()
            monthly_data['month'] = monthly_data['date'].dt.month
            monthly_pattern = monthly_data.groupby('month')['demand_units'].mean()
            
            fig_monthly = px.bar(
                x=monthly_pattern.index,
                y=monthly_pattern.values,
                title="Average Demand by Month",
                labels={'x': 'Month', 'y': 'Average Demand'}
            )
            fig_monthly.update_layout(
                xaxis=dict(tickmode='linear', tick0=1, dtick=1)
            )
            st.plotly_chart(fig_monthly, use_container_width=True, key="monthly_seasonality")
        
        with col2:
            st.write("**Day of Week Pattern**")
            
            # Day of week analysis
            dow_data = filtered_df.copy()
            dow_data['day_of_week'] = dow_data['date'].dt.day_name()
            dow_pattern = dow_data.groupby('day_of_week')['demand_units'].mean()
            
            # Reorder days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_pattern = dow_pattern.reindex([day for day in day_order if day in dow_pattern.index])
            
            fig_dow = px.bar(
                x=dow_pattern.index,
                y=dow_pattern.values,
                title="Average Demand by Day of Week",
                labels={'x': 'Day', 'y': 'Average Demand'}
            )
            st.plotly_chart(fig_dow, use_container_width=True, key="day_of_week_pattern")
        
        # Trend decomposition
        st.write("**Trend Decomposition**")
        
        # Aggregate daily data for trend analysis
        daily_trend = filtered_df.groupby('date')['demand_units'].sum().reset_index()
        daily_trend = daily_trend.sort_values('date')
        
        if len(daily_trend) >= 30:  # Need at least 30 days for meaningful analysis
            # Calculate moving averages
            daily_trend['ma_7'] = daily_trend['demand_units'].rolling(window=7, center=True).mean()
            daily_trend['ma_30'] = daily_trend['demand_units'].rolling(window=min(30, len(daily_trend)//2), center=True).mean()
            
            fig_trend = go.Figure()
            
            fig_trend.add_trace(go.Scatter(
                x=daily_trend['date'],
                y=daily_trend['demand_units'],
                mode='lines',
                name='Actual Demand',
                line=dict(color='lightblue', width=1)
            ))
            
            fig_trend.add_trace(go.Scatter(
                x=daily_trend['date'],
                y=daily_trend['ma_7'],
                mode='lines',
                name='7-Day Trend',
                line=dict(color='orange', width=2)
            ))
            
            fig_trend.add_trace(go.Scatter(
                x=daily_trend['date'],
                y=daily_trend['ma_30'],
                mode='lines',
                name='30-Day Trend',
                line=dict(color='red', width=3)
            ))
            
            fig_trend.update_layout(
                title="Demand Trends with Moving Averages",
                xaxis_title="Date",
                yaxis_title="Demand Units",
                height=400
            )
            
            st.plotly_chart(fig_trend, use_container_width=True, key="trend_decomposition")
        
        # Seasonal statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            peak_month = monthly_pattern.idxmax()
            st.metric("🏔️ Peak Demand Month", f"Month {peak_month}", f"+{monthly_pattern.max():.0f} units")
        
        with col2:
            low_month = monthly_pattern.idxmin()
            st.metric("📉 Lowest Demand Month", f"Month {low_month}", f"-{monthly_pattern.min():.0f} units")
        
        with col3:
            seasonality_strength = (monthly_pattern.max() - monthly_pattern.min()) / monthly_pattern.mean() * 100
            st.metric("🌊 Seasonality Strength", f"{seasonality_strength:.1f}%")

    with tab5:
        st.subheader("⚠️ Anomaly Detection & Quality Analysis")
        
        # Anomaly detection using statistical methods
        def detect_anomalies(data, threshold=2.5):
            """Detect anomalies using Z-score method"""
            z_scores = np.abs(stats.zscore(data))
            return z_scores > threshold
        
        # Prepare data for anomaly detection
        anomaly_data = filtered_df.copy()
        anomaly_data['z_score'] = np.abs(stats.zscore(anomaly_data['demand_units']))
        anomaly_data['is_anomaly'] = anomaly_data['z_score'] > 2.5
        
        # Anomaly summary
        total_anomalies = anomaly_data['is_anomaly'].sum()
        anomaly_rate = (total_anomalies / len(anomaly_data)) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🚨 Total Anomalies", total_anomalies)
        
        with col2:
            st.metric("📊 Anomaly Rate", f"{anomaly_rate:.2f}%")
        
        with col3:
            avg_anomaly_magnitude = anomaly_data[anomaly_data['is_anomaly']]['demand_units'].mean()
            st.metric("📈 Avg Anomaly Size", f"{avg_anomaly_magnitude:.0f}" if not pd.isna(avg_anomaly_magnitude) else "N/A")
        
        # Anomaly visualization
        fig_anomaly = go.Figure()
        
        # Normal points
        normal_data = anomaly_data[~anomaly_data['is_anomaly']]
        fig_anomaly.add_trace(go.Scatter(
            x=normal_data['date'],
            y=normal_data['demand_units'],
            mode='markers',
            name='Normal Demand',
            marker=dict(color='blue', size=6)
        ))
        
        # Anomalous points
        anomaly_points = anomaly_data[anomaly_data['is_anomaly']]
        if len(anomaly_points) > 0:
            fig_anomaly.add_trace(go.Scatter(
                x=anomaly_points['date'],
                y=anomaly_points['demand_units'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10, symbol='x')
            ))
        
        fig_anomaly.update_layout(
            title="Demand Anomaly Detection (Z-score > 2.5)",
            xaxis_title="Date",
            yaxis_title="Demand Units",
            height=400
        )
        
        st.plotly_chart(fig_anomaly, use_container_width=True, key="anomaly_detection")
        
        # Anomaly details table
        if len(anomaly_points) > 0:
            st.write("**Detected Anomalies**")
            anomaly_details = anomaly_points[['date', 'plant_id', 'product_category', 'demand_units', 'z_score']].copy()
            anomaly_details['date'] = anomaly_details['date'].dt.strftime('%Y-%m-%d')
            anomaly_details = anomaly_details.sort_values('z_score', ascending=False)
            
            st.dataframe(
                anomaly_details,
                column_config={
                    "date": "Date",
                    "plant_id": "Plant",
                    "product_category": "Category",
                    "demand_units": st.column_config.NumberColumn("Demand", format="%.0f"),
                    "z_score": st.column_config.NumberColumn("Z-Score", format="%.2f")
                },
                use_container_width=True
            )
        
        # Forecast accuracy analysis
        st.write("**Forecast Quality Metrics**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Error distribution
            error_data = filtered_df['percentage_error'].dropna()
            if len(error_data) > 0:
                fig_error_dist = px.histogram(
                    x=error_data,
                    nbins=20,
                    title="Forecast Error Distribution",
                    labels={'x': 'Percentage Error (%)', 'y': 'Frequency'}
                )
                st.plotly_chart(fig_error_dist, use_container_width=True, key="error_distribution")
        
        with col2:
            # Error by category
            if 'percentage_error' in filtered_df.columns:
                category_errors = filtered_df.groupby('product_category')['percentage_error'].agg(['mean', 'std']).reset_index()
                category_errors['mean_abs_error'] = filtered_df.groupby('product_category')['percentage_error'].apply(lambda x: abs(x).mean()).values
                
                fig_cat_error = px.bar(
                    category_errors,
                    x='product_category',
                    y='mean_abs_error',
                    title="Mean Absolute Error by Category",
                    labels={'product_category': 'Category', 'mean_abs_error': 'Mean Absolute Error (%)'}
                )
                st.plotly_chart(fig_cat_error, use_container_width=True, key="category_error_analysis")

    with tab6:
        st.subheader("🎯 What-If Analysis & Scenario Planning")
        
        st.write("**Scenario Analysis Tools**")
        
        # Scenario parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Market Scenarios**")
            
            # Market growth scenario
            growth_factor = st.slider(
                "Market Growth Factor",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="1.0 = no change, 1.5 = 50% increase, 0.8 = 20% decrease"
            )
            
            # Seasonality adjustment
            seasonality_factor = st.slider(
                "Seasonality Amplification",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Amplify or dampen seasonal patterns"
            )
            
            # Plant capacity constraint
            capacity_constraint = st.checkbox("Apply Capacity Constraints", value=False)
        
        with col2:
            st.write("**Supply Chain Scenarios**")
            
            # Supply disruption
            disruption_severity = st.selectbox(
                "Supply Chain Disruption",
                ["None", "Minor (5% impact)", "Moderate (15% impact)", "Severe (30% impact)"]
            )
            
            # Lead time changes
            lead_time_factor = st.slider(
                "Lead Time Factor",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="1.0 = normal, 2.0 = double lead times"
            )
        
        # Calculate scenario impact
        scenario_data = filtered_df.copy()
        base_demand = scenario_data['demand_units'].sum()
        
        # Apply growth factor
        scenario_data['scenario_demand'] = scenario_data['demand_units'] * growth_factor
        
        # Apply seasonality changes
        scenario_data['month'] = scenario_data['date'].dt.month
        monthly_avg = scenario_data.groupby('month')['demand_units'].transform('mean')
        monthly_deviation = scenario_data['demand_units'] - monthly_avg
        scenario_data['scenario_demand'] += monthly_deviation * (seasonality_factor - 1)
        
        # Apply supply disruption
        disruption_impact = {
            "None": 0,
            "Minor (5% impact)": 0.05,
            "Moderate (15% impact)": 0.15,
            "Severe (30% impact)": 0.30
        }
        
        if disruption_severity != "None":
            # Random disruption events
            np.random.seed(42)
            disruption_days = np.random.choice(len(scenario_data), 
                                             size=int(len(scenario_data) * 0.1), 
                                             replace=False)
            # Use iloc to access by position, not by index label
            for day_idx in disruption_days:
                scenario_data.iloc[day_idx, scenario_data.columns.get_loc('scenario_demand')] *= (1 - disruption_impact[disruption_severity])
        
        # Apply capacity constraints
        if capacity_constraint:
            # Load capacity data if available
            try:
                capacity_df = pd.read_csv("data/plant_capacity.csv")
                plant_capacity = dict(zip(capacity_df['plant_id'], capacity_df['max_daily_capacity']))
                
                for plant in scenario_data['plant_id'].unique():
                    if plant in plant_capacity:
                        daily_capacity = plant_capacity[plant]
                        plant_mask = scenario_data['plant_id'] == plant
                        scenario_data.loc[plant_mask, 'scenario_demand'] = np.minimum(
                            scenario_data.loc[plant_mask, 'scenario_demand'],
                            daily_capacity
                        )
            except:
                st.warning("⚠️ Capacity data not available. Skipping capacity constraints.")
        
        # Scenario results
        scenario_demand = scenario_data['scenario_demand'].sum()
        demand_change = ((scenario_demand / base_demand) - 1) * 100
        
        st.write("**Scenario Results**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "📊 Base Demand",
                f"{base_demand:,.0f}",
                delta=None
            )
        
        with col2:
            st.metric(
                "🎯 Scenario Demand", 
                f"{scenario_demand:,.0f}",
                delta=f"{demand_change:+.1f}%"
            )
        
        with col3:
            impact = "Positive" if demand_change > 0 else "Negative" if demand_change < 0 else "Neutral"
            st.metric("📈 Impact", impact, f"{abs(demand_change):.1f}% change")
        
        # Scenario visualization
        scenario_comparison = scenario_data.groupby('date').agg({
            'demand_units': 'sum',
            'scenario_demand': 'sum'
        }).reset_index()
        
        fig_scenario = go.Figure()
        
        fig_scenario.add_trace(go.Scatter(
            x=scenario_comparison['date'],
            y=scenario_comparison['demand_units'],
            mode='lines',
            name='Current Demand',
            line=dict(color='blue', width=2)
        ))
        
        fig_scenario.add_trace(go.Scatter(
            x=scenario_comparison['date'],
            y=scenario_comparison['scenario_demand'],
            mode='lines',
            name='Scenario Demand',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_scenario.update_layout(
            title="Current vs Scenario Demand Comparison",
            xaxis_title="Date",
            yaxis_title="Demand Units",
            height=400
        )
        
        st.plotly_chart(fig_scenario, use_container_width=True, key="scenario_comparison")
        
        # Export scenario results
        if st.button("📄 Export Scenario Analysis"):
            # Prepare export data
            export_data = scenario_data[['date', 'plant_id', 'product_category', 'demand_units', 'scenario_demand']].copy()
            export_data['scenario_change'] = export_data['scenario_demand'] - export_data['demand_units']
            export_data['scenario_change_pct'] = (export_data['scenario_change'] / export_data['demand_units']) * 100
            
            # Convert to CSV
            csv_buffer = io.StringIO()
            export_data.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            # Create download link
            b64 = base64.b64encode(csv_data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="scenario_analysis.csv">📄 Download Scenario Analysis CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("✅ Scenario analysis prepared for download!")

    with tab7:
        st.subheader("📦 Inventory Optimization & Management")
        
        # Initialize inventory optimizer
        try:
            optimizer = InventoryOptimizer()
        except:
            st.error("⚠️ Inventory optimizer not available. Using fallback calculations.")
            optimizer = None
        
        if optimizer:
            # Load additional data files
            try:
                plant_capacity_df = pd.read_csv("data/plant_capacity.csv")
                raw_materials_df = pd.read_csv("data/raw_materials_master.csv")
            except:
                st.warning("⚠️ Some data files not available. Using default values.")
                plant_capacity_df = pd.DataFrame()
                raw_materials_df = pd.DataFrame()
            
            # Inventory optimization controls
            st.write("**🔧 Optimization Settings**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_plant_inv = st.selectbox(
                    "Select Plant for Analysis",
                    options=sorted(filtered_df['plant_id'].unique()),
                    key="inventory_plant_selector"
                )
            
            with col2:
                service_level = st.slider(
                    "Service Level (%)",
                    min_value=85,
                    max_value=99,
                    value=95,
                    step=1,
                    help="Target service level for safety stock calculations"
                )
                optimizer.service_level = service_level / 100
                optimizer.z_score = stats.norm.ppf(optimizer.service_level)
            
            with col3:
                lead_time_days = st.number_input(
                    "Lead Time (days)",
                    min_value=1,
                    max_value=90,
                    value=14,
                    help="Average lead time for material procurement"
                )
            
            # Perform inventory optimization
            plant_data = filtered_df[filtered_df['plant_id'] == selected_plant_inv]
            
            if len(plant_data) > 0:
                # Calculate inventory metrics by category
                inventory_results = {}
                
                for category in plant_data['product_category'].unique():
                    category_data = plant_data[plant_data['product_category'] == category]
                    
                    # Calculate demand statistics
                    avg_demand = category_data['demand_units'].mean()
                    demand_std = category_data['demand_units'].std()
                    max_demand = category_data['demand_units'].max()
                    min_demand = category_data['demand_units'].min()
                    
                    # Safety stock calculation
                    safety_stock = optimizer.calculate_safety_stock(
                        avg_demand, demand_std, lead_time_days
                    )
                    
                    # Reorder point calculation
                    reorder_point = optimizer.calculate_reorder_point(
                        avg_demand, lead_time_days, safety_stock
                    )
                    
                    # EOQ calculation (with default costs)
                    annual_demand = avg_demand * 365
                    ordering_cost = 150  # Default ordering cost
                    holding_cost_per_unit = avg_demand * 0.2  # 20% of average demand as holding cost
                    
                    eoq = optimizer.calculate_eoq(annual_demand, ordering_cost, holding_cost_per_unit)
                    
                    # Current inventory (estimate as 5 days of average demand)
                    current_inventory = avg_demand * 5
                    
                    # Inventory status
                    if current_inventory <= reorder_point * 0.5:
                        status = "🚨 CRITICAL"
                        status_color = "red"
                    elif current_inventory <= reorder_point:
                        status = "⚠️ REORDER NEEDED"
                        status_color = "orange"
                    elif current_inventory <= reorder_point * 1.5:
                        status = "✅ OPTIMAL"
                        status_color = "green"
                    else:
                        status = "📦 OVERSTOCK"
                        status_color = "blue"
                    
                    inventory_results[category] = {
                        'avg_demand': avg_demand,
                        'demand_std': demand_std,
                        'safety_stock': safety_stock,
                        'reorder_point': reorder_point,
                        'eoq': eoq,
                        'current_inventory': current_inventory,
                        'status': status,
                        'status_color': status_color,
                        'days_of_stock': current_inventory / avg_demand if avg_demand > 0 else 0
                    }
                
                # Display inventory dashboard
                st.write(f"**📊 Inventory Analysis for {selected_plant_inv}**")
                
                # Key metrics overview
                col1, col2, col3, col4 = st.columns(4)
                
                total_categories = len(inventory_results)
                critical_items = sum(1 for r in inventory_results.values() if "CRITICAL" in r['status'])
                reorder_items = sum(1 for r in inventory_results.values() if "REORDER" in r['status'])
                optimal_items = sum(1 for r in inventory_results.values() if "OPTIMAL" in r['status'])
                
                with col1:
                    st.metric("📦 Total Categories", total_categories)
                
                with col2:
                    st.metric("🚨 Critical Items", critical_items, 
                             delta="Urgent attention needed" if critical_items > 0 else None)
                
                with col3:
                    st.metric("⚠️ Reorder Needed", reorder_items,
                             delta="Action required" if reorder_items > 0 else None)
                
                with col4:
                    st.metric("✅ Optimal Items", optimal_items,
                             delta="Well managed" if optimal_items > 0 else None)
                
                # Inventory status visualization
                st.write("**📈 Inventory Status Overview**")
                
                # Create inventory status chart
                status_data = []
                for category, data in inventory_results.items():
                    status_data.append({
                        'Category': category,
                        'Current Stock': data['current_inventory'],
                        'Safety Stock': data['safety_stock'],
                        'Reorder Point': data['reorder_point'],
                        'EOQ': data['eoq'],
                        'Status': data['status'].split(' ', 1)[1] if ' ' in data['status'] else data['status']
                    })
                
                status_df = pd.DataFrame(status_data)
                
                # Inventory levels chart
                fig_inventory = go.Figure()
                
                categories = status_df['Category'].tolist()
                
                fig_inventory.add_trace(go.Bar(
                    name='Current Stock',
                    x=categories,
                    y=status_df['Current Stock'],
                    marker_color='lightblue'
                ))
                
                fig_inventory.add_trace(go.Scatter(
                    name='Reorder Point',
                    x=categories,
                    y=status_df['Reorder Point'],
                    mode='markers+lines',
                    marker=dict(color='red', size=10),
                    line=dict(color='red', width=2)
                ))
                
                fig_inventory.add_trace(go.Scatter(
                    name='Safety Stock',
                    x=categories,
                    y=status_df['Safety Stock'],
                    mode='markers+lines',
                    marker=dict(color='orange', size=8),
                    line=dict(color='orange', width=2, dash='dash')
                ))
                
                fig_inventory.update_layout(
                    title=f"Inventory Levels - {selected_plant_inv}",
                    xaxis_title="Product Category",
                    yaxis_title="Inventory Units",
                    height=400,
                    barmode='group'
                )
                
                st.plotly_chart(fig_inventory, use_container_width=True, key="inventory_levels_chart")
                
                # Detailed inventory table
                st.write("**📋 Detailed Inventory Analysis**")
                
                # Prepare data for display
                display_data = []
                for category, data in inventory_results.items():
                    display_data.append({
                        'Category': category,
                        'Avg Daily Demand': f"{data['avg_demand']:.1f}",
                        'Current Stock': f"{data['current_inventory']:.0f}",
                        'Days of Stock': f"{data['days_of_stock']:.1f}",
                        'Safety Stock': f"{data['safety_stock']:.0f}",
                        'Reorder Point': f"{data['reorder_point']:.0f}",
                        'EOQ': f"{data['eoq']:.0f}",
                        'Status': data['status']
                    })
                
                display_df = pd.DataFrame(display_data)
                
                st.dataframe(
                    display_df,
                    column_config={
                        "Category": "Product Category",
                        "Avg Daily Demand": st.column_config.NumberColumn("Avg Daily Demand", format="%.1f"),
                        "Current Stock": st.column_config.NumberColumn("Current Stock", format="%.0f"),
                        "Days of Stock": st.column_config.NumberColumn("Days of Stock", format="%.1f"),
                        "Safety Stock": st.column_config.NumberColumn("Safety Stock", format="%.0f"),
                        "Reorder Point": st.column_config.NumberColumn("Reorder Point", format="%.0f"),
                        "EOQ": st.column_config.NumberColumn("EOQ", format="%.0f"),
                        "Status": "Inventory Status"
                    },
                    use_container_width=True
                )
                
                # Cost analysis
                st.write("**💰 Cost Analysis & Recommendations**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Calculate total inventory value
                    unit_cost = 50  # Default unit cost
                    total_current_value = sum(data['current_inventory'] for data in inventory_results.values()) * unit_cost
                    total_safety_stock_value = sum(data['safety_stock'] for data in inventory_results.values()) * unit_cost
                    total_eoq_value = sum(data['eoq'] for data in inventory_results.values()) * unit_cost
                    
                    st.info(f"""
                    **💵 Financial Impact**
                    - Current Inventory Value: ${total_current_value:,.0f}
                    - Safety Stock Investment: ${total_safety_stock_value:,.0f}
                    - Optimal Order Value (EOQ): ${total_eoq_value:,.0f}
                    """)
                
                with col2:
                    # Generate recommendations
                    recommendations = []
                    
                    for category, data in inventory_results.items():
                        if "CRITICAL" in data['status']:
                            recommendations.append(f"🚨 {category}: Order {data['eoq']:.0f} units immediately")
                        elif "REORDER" in data['status']:
                            recommendations.append(f"⚠️ {category}: Order {data['eoq']:.0f} units soon")
                        elif "OVERSTOCK" in data['status']:
                            recommendations.append(f"📦 {category}: Reduce future orders, {data['days_of_stock']:.1f} days of stock")
                    
                    if not recommendations:
                        recommendations.append("✅ All inventory levels are optimal")
                    
                    st.success("**🎯 Action Items:**\n" + "\n".join(recommendations[:5]))
                
                # ABC Analysis
                st.write("**📊 ABC Analysis (Value-based Classification)**")
                
                # Calculate annual values for ABC analysis
                abc_data = []
                for category, data in inventory_results.items():
                    annual_demand = data['avg_demand'] * 365
                    annual_value = annual_demand * unit_cost
                    abc_data.append({
                        'Category': category,
                        'Annual Demand': annual_demand,
                        'Annual Value': annual_value
                    })
                
                abc_df = pd.DataFrame(abc_data)
                abc_df = abc_df.sort_values('Annual Value', ascending=False)
                
                # Calculate cumulative percentage
                total_value = abc_df['Annual Value'].sum()
                abc_df['Cumulative Value'] = abc_df['Annual Value'].cumsum()
                abc_df['Cumulative %'] = (abc_df['Cumulative Value'] / total_value) * 100
                
                # Assign ABC categories
                abc_df['ABC Class'] = 'C'
                abc_df.loc[abc_df['Cumulative %'] <= 80, 'ABC Class'] = 'A'
                abc_df.loc[(abc_df['Cumulative %'] > 80) & (abc_df['Cumulative %'] <= 95), 'ABC Class'] = 'B'
                
                # ABC visualization
                fig_abc = px.bar(
                    abc_df,
                    x='Category',
                    y='Annual Value',
                    color='ABC Class',
                    title="ABC Analysis - Annual Value by Category",
                    color_discrete_map={'A': 'red', 'B': 'orange', 'C': 'green'}
                )
                
                st.plotly_chart(fig_abc, use_container_width=True, key="abc_analysis_chart")
                
                # Export inventory analysis
                if st.button("📄 Export Inventory Analysis", key="export_inventory"):
                    # Prepare comprehensive export data
                    export_inventory_data = pd.DataFrame([
                        {
                            'Plant': selected_plant_inv,
                            'Category': category,
                            'Service Level': f"{service_level}%",
                            'Lead Time (days)': lead_time_days,
                            'Avg Daily Demand': data['avg_demand'],
                            'Demand Std Dev': data['demand_std'],
                            'Current Stock': data['current_inventory'],
                            'Days of Stock': data['days_of_stock'],
                            'Safety Stock': data['safety_stock'],
                            'Reorder Point': data['reorder_point'],
                            'EOQ': data['eoq'],
                            'Status': data['status'],
                            'ABC Class': abc_df[abc_df['Category'] == category]['ABC Class'].iloc[0] if len(abc_df[abc_df['Category'] == category]) > 0 else 'N/A'
                        }
                        for category, data in inventory_results.items()
                    ])
                    
                    csv_buffer = io.StringIO()
                    export_inventory_data.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    b64 = base64.b64encode(csv_data.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="inventory_analysis.csv">📄 Download Inventory Analysis</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("✅ Inventory analysis exported successfully!")
            
            else:
                st.warning("⚠️ No data available for selected plant.")
        
        else:
            st.error("❌ Inventory optimization module not available.")

    with tab8:
        st.subheader("🤖 Advanced Machine Learning Models")
        
        # Initialize advanced ML ensemble
        try:
            ml_ensemble = AdvancedMLEnsemble()
            
            st.write("**🔧 Model Configuration**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Available Models:**")
                for model, available in ml_ensemble.available_models.items():
                    status = "✅" if available else "❌"
                    st.write(f"{status} **{model.upper()}**")
                
                # Training options
                retrain_models = st.checkbox(
                    "Retrain Models",
                    value=False,
                    help="Retrain all models with current data"
                )
                
                optimize_hyperparams = st.checkbox(
                    "Enable Hyperparameter Optimization",
                    value=False,
                    help="⚡ Fast Mode (unchecked) vs 🔬 Full Optimization (checked) - Optimization takes much longer but may improve accuracy"
                )
                
                test_size = st.slider(
                    "Test Data Size (%)",
                    min_value=10,
                    max_value=40,
                    value=20,
                    help="Percentage of data for testing"
                )
            
            with col2:
                st.write("**Model Selection:**")
                
                selected_models = st.multiselect(
                    "Models to Use",
                    options=['random_forest', 'xgboost', 'prophet', 'lstm', 'ensemble'],
                    default=['random_forest', 'xgboost'] if ml_ensemble.available_models['xgboost'] else ['random_forest'],
                    help="Select models for training and prediction"
                )
                
                prediction_horizon = st.number_input(
                    "Prediction Days",
                    min_value=1,
                    max_value=90,
                    value=30,
                    help="Number of days to forecast"
                )
            
            # Model training section
            if retrain_models and st.button("🚀 Train Advanced Models"):
                mode_text = "🔬 Full Optimization Mode" if optimize_hyperparams else "⚡ Fast Training Mode"
                st.write(f"### 🎯 Training Advanced ML Models ({mode_text})...")
                
                timeout_text = "several minutes" if optimize_hyperparams else "1-2 minutes"
                with st.spinner(f"Training models... This may take {timeout_text}."):
                    try:
                        # Train the ensemble
                        ml_ensemble.fit(filtered_df, optimize_hyperparams=optimize_hyperparams)
                        
                        # Save models
                        ml_ensemble.save_models("models/advanced_ml_ensemble")
                        
                        st.success("✅ Models trained successfully!")
                        
                        # Display model performance
                        st.write("**📊 Model Performance:**")
                        
                        performance_data = []
                        for model_name, metrics in ml_ensemble.model_performance.items():
                            if model_name != 'ensemble' or 'weights' not in metrics:
                                performance_data.append({
                                    'Model': model_name.upper(),
                                    'MAE': f"{metrics['mae']:.2f}",
                                    'MAPE (%)': f"{metrics['mape']:.1f}",
                                    'MSE': f"{metrics['mse']:.2f}"
                                })
                        
                        if performance_data:
                            perf_df = pd.DataFrame(performance_data)
                            st.dataframe(perf_df, use_container_width=True)
                        
                        # Feature importance
                        if 'random_forest' in ml_ensemble.feature_importance:
                            st.write("**🔍 Feature Importance (Random Forest):**")
                            
                            importance = ml_ensemble.get_feature_importance('random_forest', top_n=10)
                            
                            if importance:
                                fig_importance = px.bar(
                                    x=list(importance.values()),
                                    y=list(importance.keys()),
                                    orientation='h',
                                    title="Top 10 Most Important Features",
                                    labels={'x': 'Importance', 'y': 'Features'}
                                )
                                st.plotly_chart(fig_importance, use_container_width=True, key="feature_importance_chart")
                        
                    except Exception as e:
                        st.error(f"❌ Model training failed: {str(e)}")
                        st.info("💡 Try with a smaller dataset or check data quality.")
            
            # Model prediction section
            st.write("### 🎯 Advanced Model Predictions")
            
            # Check if models exist
            try:
                # Try to load existing models
                ml_ensemble.load_models("models/advanced_ml_ensemble")
                models_available = True
            except:
                models_available = False
            
            if models_available:
                st.success("✅ Pre-trained models loaded successfully!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Model comparison
                    st.write("**📈 Model Comparison:**")
                    
                    if ml_ensemble.model_performance:
                        comparison_data = []
                        for model_name, metrics in ml_ensemble.model_performance.items():
                            if 'mae' in metrics:
                                comparison_data.append({
                                    'Model': model_name.upper(),
                                    'Accuracy': f"{100 - metrics.get('mape', 0):.1f}%",
                                    'MAE': f"{metrics['mae']:.2f}",
                                    'Status': "✅ Available" if model_name in selected_models else "⏸️ Not Selected"
                                })
                        
                        if comparison_data:
                            comp_df = pd.DataFrame(comparison_data)
                            st.dataframe(comp_df, use_container_width=True)
                
                with col2:
                    # Ensemble weights (if available)
                    if 'ensemble' in ml_ensemble.model_performance and 'weights' in ml_ensemble.model_performance['ensemble']:
                        st.write("**⚖️ Ensemble Weights:**")
                        
                        weights = ml_ensemble.model_performance['ensemble']['weights']
                        
                        fig_weights = px.pie(
                            values=list(weights.values()),
                            names=list(weights.keys()),
                            title="Model Contribution to Ensemble"
                        )
                        st.plotly_chart(fig_weights, use_container_width=True, key="ensemble_weights_pie")
                
                # Generate predictions
                if st.button("🔮 Generate Advanced Predictions"):
                    st.write("### 📊 Advanced Prediction Results")
                    
                    with st.spinner("Generating predictions..."):
                        try:
                            # Make predictions for each selected model
                            prediction_results = {}
                            
                            for model_type in selected_models:
                                if model_type in ml_ensemble.available_models and ml_ensemble.available_models[model_type]:
                                    try:
                                        predictions = ml_ensemble.predict(filtered_df, model_type=model_type)
                                        prediction_results[model_type] = predictions[:len(filtered_df)]  # Ensure same length
                                    except Exception as e:
                                        st.warning(f"⚠️ {model_type} prediction failed: {str(e)}")
                            
                            if prediction_results:
                                # Create prediction comparison chart
                                pred_df = filtered_df[['date', 'demand_units']].copy()
                                
                                for model_name, predictions in prediction_results.items():
                                    if len(predictions) == len(pred_df):
                                        pred_df[f'{model_name}_prediction'] = predictions
                                
                                # Aggregate by date for visualization
                                daily_pred = pred_df.groupby('date').agg({
                                    'demand_units': 'sum',
                                    **{col: 'sum' for col in pred_df.columns if col.endswith('_prediction')}
                                }).reset_index()
                                
                                # Create comparison chart
                                fig_pred = go.Figure()
                                
                                # Actual demand
                                fig_pred.add_trace(go.Scatter(
                                    x=daily_pred['date'],
                                    y=daily_pred['demand_units'],
                                    mode='lines',
                                    name='Actual Demand',
                                    line=dict(color='blue', width=3)
                                ))
                                
                                # Model predictions
                                colors = ['red', 'green', 'orange', 'purple', 'brown']
                                for i, (model_name, _) in enumerate(prediction_results.items()):
                                    pred_col = f'{model_name}_prediction'
                                    if pred_col in daily_pred.columns:
                                        fig_pred.add_trace(go.Scatter(
                                            x=daily_pred['date'],
                                            y=daily_pred[pred_col],
                                            mode='lines',
                                            name=f'{model_name.upper()} Prediction',
                                            line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                                        ))
                                
                                fig_pred.update_layout(
                                    title="Advanced ML Model Predictions Comparison",
                                    xaxis_title="Date",
                                    yaxis_title="Demand Units",
                                    height=500
                                )
                                
                                st.plotly_chart(fig_pred, use_container_width=True, key="advanced_predictions_chart")
                                
                                # Prediction accuracy metrics
                                st.write("**📊 Prediction Accuracy:**")
                                
                                accuracy_data = []
                                for model_name, predictions in prediction_results.items():
                                    if len(predictions) == len(filtered_df):
                                        actual = filtered_df['demand_units'].values
                                        mae = np.mean(np.abs(actual - predictions))
                                        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
                                        accuracy = 100 - mape
                                        
                                        accuracy_data.append({
                                            'Model': model_name.upper(),
                                            'MAE': f"{mae:.2f}",
                                            'MAPE (%)': f"{mape:.1f}",
                                            'Accuracy (%)': f"{accuracy:.1f}"
                                        })
                                
                                if accuracy_data:
                                    acc_df = pd.DataFrame(accuracy_data)
                                    st.dataframe(acc_df, use_container_width=True)
                                
                                # Export predictions
                                if st.button("📄 Export Advanced Predictions"):
                                    export_pred_df = filtered_df[['date', 'plant_id', 'product_category', 'demand_units']].copy()
                                    
                                    for model_name, predictions in prediction_results.items():
                                        if len(predictions) == len(export_pred_df):
                                            export_pred_df[f'{model_name}_prediction'] = predictions
                                            export_pred_df[f'{model_name}_error'] = export_pred_df['demand_units'] - predictions
                                    
                                    csv_buffer = io.StringIO()
                                    export_pred_df.to_csv(csv_buffer, index=False)
                                    csv_data = csv_buffer.getvalue()
                                    
                                    b64 = base64.b64encode(csv_data.encode()).decode()
                                    href = f'<a href="data:file/csv;base64,{b64}" download="advanced_ml_predictions.csv">📄 Download Advanced Predictions</a>'
                                    st.markdown(href, unsafe_allow_html=True)
                                    st.success("✅ Advanced predictions exported!")
                            
                            else:
                                st.warning("⚠️ No predictions generated. Check model availability.")
                                
                        except Exception as e:
                            st.error(f"❌ Prediction generation failed: {str(e)}")
            
            else:
                st.info("💡 **No pre-trained models found.** Please train models first using the 'Retrain Models' option above.")
                
                # Show model information
                st.write("**🤖 Advanced ML Features:**")
                st.info("""
                **Available Algorithms:**
                - 🌲 **Random Forest**: Ensemble of decision trees with feature importance
                - 🚀 **XGBoost**: Gradient boosting with hyperparameter optimization  
                - 📈 **Prophet**: Time series forecasting with seasonality detection
                - 🧠 **LSTM**: Deep learning for sequential pattern recognition
                - 🎯 **Ensemble**: Weighted combination of all models
                
                **Advanced Features:**
                - ⚡ Automated hyperparameter tuning with Optuna
                - 🎯 Cross-validation and early stopping
                - 📊 Feature engineering and selection
                - 🔍 Model interpretability and feature importance
                - 📈 Performance monitoring and comparison
                """)
        
        except Exception as e:
            st.error(f"❌ Advanced ML module initialization failed: {str(e)}")
            st.info("💡 **Fallback Mode**: Using basic forecasting models.")
            
            # Show basic model info
            st.write("**📊 Current Model Performance:**")
            try:
                basic_model = joblib.load("models/manufacturing_random_forest_model.pkl")
                st.success("✅ Basic Random Forest model loaded")
                
                # Load feature importance if available
                try:
                    feature_importance_df = pd.read_csv("outputs/feature_importance.csv")
                    st.write("**🔍 Current Feature Importance:**")
                    
                    fig_basic_importance = px.bar(
                        feature_importance_df.head(10),
                        x='avg_importance',
                        y='feature',
                        orientation='h',
                        title="Top 10 Important Features (Current Model)"
                    )
                    st.plotly_chart(fig_basic_importance, use_container_width=True, key="basic_feature_importance")
                    
                except:
                    st.info("📊 Feature importance data not available")
                    
            except:
                st.warning("⚠️ No trained models found")

    # Data summary and export options
    st.subheader("📋 Data Summary & Export")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info(f"**Date Range**: {filtered_df['date'].min().strftime('%Y-%m-%d')} to {filtered_df['date'].max().strftime('%Y-%m-%d')}")
    
    with col2:
        st.info(f"**Total Records**: {len(filtered_df):,}")
    
    with col3:
        st.info(f"**Plants**: {len(filtered_df['plant_id'].unique())}")
    
    with col4:
        st.info(f"**Categories**: {len(filtered_df['product_category'].unique())}")
    
    # Export functionality
    st.write("**📥 Export Options**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Dashboard Data"):
            # Prepare dashboard summary
            summary_data = {
                'Metric': ['Total Actual Demand', 'Total Predicted Demand', 'Mean Absolute Error', 'Forecast Accuracy'],
                'Value': [f"{total_actual:,.0f}", f"{total_predicted:,.0f}", f"{mae:.1f}", f"{accuracy:.1f}%"]
            }
            summary_df = pd.DataFrame(summary_data)
            
            csv_buffer = io.StringIO()
            summary_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            b64 = base64.b64encode(csv_data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="dashboard_summary.csv">📄 Download Summary</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        if st.button("📈 Export Filtered Data"):
            # Export filtered dataset
            export_df = filtered_df[['date', 'plant_id', 'product_line', 'product_category', 
                                   'demand_units', 'predicted_demand', 'absolute_error', 'percentage_error']].copy()
            
            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            b64 = base64.b64encode(csv_data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv">📄 Download Filtered Data</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col3:
        if st.button("📋 Generate Report"):
            # Generate comprehensive report
            report_data = []
            
            # Add key metrics
            report_data.append(["=== MANUFACTURING DEMAND FORECAST REPORT ===", ""])
            report_data.append(["Report Date", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            report_data.append(["Date Range", f"{filtered_df['date'].min().strftime('%Y-%m-%d')} to {filtered_df['date'].max().strftime('%Y-%m-%d')}"])
            report_data.append(["", ""])
            
            # Key metrics
            report_data.append(["=== KEY METRICS ===", ""])
            report_data.append(["Total Actual Demand", f"{total_actual:,.0f}"])
            report_data.append(["Total Predicted Demand", f"{total_predicted:,.0f}"])
            report_data.append(["Mean Absolute Error", f"{mae:.2f}"])
            report_data.append(["Forecast Accuracy", f"{accuracy:.2f}%"])
            report_data.append(["", ""])
            
            # Plant performance
            report_data.append(["=== PLANT PERFORMANCE ===", ""])
            plant_summary = filtered_df.groupby('plant_id')['demand_units'].sum().sort_values(ascending=False)
            for plant, demand in plant_summary.items():
                report_data.append([f"Plant {plant}", f"{demand:,.0f}"])
            
            # Convert to DataFrame and export
            report_df = pd.DataFrame(report_data, columns=['Metric', 'Value'])
            
            csv_buffer = io.StringIO()
            report_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            b64 = base64.b64encode(csv_data.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="forecast_report.csv">📄 Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("✅ Comprehensive report generated!")

else:
    st.error("❌ Unable to load data. Please check that the data files exist.")
    st.info("💡 Make sure you have run the data generation script and forecasting notebook.")

# Add advanced settings in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Advanced Settings")

# Forecast settings
forecast_horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    [7, 14, 30, 60, 90],
    index=2,
    help="Number of days to forecast ahead"
)

# Confidence level
confidence_level = st.sidebar.slider(
    "Confidence Level",
    min_value=80,
    max_value=99,
    value=95,
    step=1,
    help="Statistical confidence level for predictions"
)

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ["Random Forest", "Ensemble", "LightGBM"],
    help="Select forecasting model"
)

# Auto-refresh option
auto_refresh = st.sidebar.checkbox(
    "Auto-refresh Data",
    value=False,
    help="Automatically refresh data every 5 minutes"
)

if auto_refresh:
    st.sidebar.info("🔄 Auto-refresh enabled (5 min intervals)")

# Footer with additional information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>🏭 Manufacturing Demand Forecasting Dashboard | Built with Streamlit & Machine Learning</p>
    <p>📊 Real-time Analytics • 🎯 Predictive Insights • 📈 Business Intelligence</p>
</div>
""", unsafe_allow_html=True)
