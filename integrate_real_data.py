"""
Example Real Data Integration Script
This script demonstrates how to integrate real manufacturing data
Replace the sample data loading with your actual data sources
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.real_data_integration import RealDataIntegrator
from data.erp_config import SAP_COLUMN_MAPPING, ORACLE_COLUMN_MAPPING, VALIDATION_RULES

def create_sample_real_data():
    """
    Create a more realistic sample dataset that mimics real manufacturing data
    This simulates data you might get from ERP systems
    """
    print("🔄 Creating realistic sample manufacturing data...")
    
    # Date range - 2 years of historical data
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 7, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Real plant configurations
    plants = {
        'Mumbai_Electronics': {
            'category': 'Electronics',
            'base_capacity': 1500,
            'products': ['PCB_Assembly_001', 'PCB_Assembly_002', 'Component_001']
        },
        'Chennai_Automotive': {
            'category': 'Automotive', 
            'base_capacity': 1200,
            'products': ['Engine_Part_A01', 'Body_Part_B02', 'Interior_C03']
        },
        'Pune_Textiles': {
            'category': 'Textiles',
            'base_capacity': 2000,
            'products': ['Cotton_Fabric_T01', 'Synthetic_T02', 'Blended_T03']
        },
        'Bangalore_Pharma': {
            'category': 'Pharmaceuticals',
            'base_capacity': 800,
            'products': ['Tablet_P01', 'Liquid_P02', 'API_P03']
        }
    }
    
    # Raw materials mapping
    raw_materials = {
        'Electronics': ['Silicon_Wafers', 'Copper_Wire', 'Plastic_Resin', 'Gold_Wire'],
        'Automotive': ['Steel_Sheet', 'Aluminum_Block', 'Rubber_Compound', 'Electronics_Module'],
        'Textiles': ['Cotton_Fiber', 'Polyester_Yarn', 'Dye_Chemical', 'Finishing_Agent'],
        'Pharmaceuticals': ['Active_Ingredient', 'Lactose', 'Starch', 'Cellulose']
    }
    
    data = []
    
    for date in dates:
        # Economic factors affecting all plants
        economic_base = 100
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
        economic_index = economic_base * seasonal_factor * (0.95 + 0.1 * np.random.random())
        
        # Global factors
        is_holiday = date.weekday() >= 5 or date.month == 12 and date.day >= 25
        fuel_cost = 50 + 10 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365) + np.random.normal(0, 2)
        
        for plant_id, plant_config in plants.items():
            for product in plant_config['products']:
                
                # Plant-specific factors
                base_demand = plant_config['base_capacity'] * (0.7 + 0.3 * np.random.random())
                
                # Weekly patterns (lower on weekends)
                weekly_factor = 0.3 if date.weekday() >= 5 else 1.0
                
                # Holiday effects
                holiday_factor = 0.1 if is_holiday else 1.0
                
                # Seasonal patterns (different for each category)
                if plant_config['category'] == 'Electronics':
                    seasonal = 1 + 0.3 * np.sin(2 * np.pi * (date.timetuple().tm_yday + 90) / 365)
                elif plant_config['category'] == 'Automotive':
                    seasonal = 1 + 0.2 * np.sin(2 * np.pi * (date.timetuple().tm_yday + 45) / 365)
                elif plant_config['category'] == 'Textiles':
                    seasonal = 1 + 0.25 * np.sin(2 * np.pi * (date.timetuple().tm_yday + 180) / 365)
                else:  # Pharmaceuticals
                    seasonal = 1 + 0.1 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                
                # Calculate final demand
                demand = int(base_demand * weekly_factor * holiday_factor * seasonal)
                demand = max(0, demand)  # No negative demand
                
                # Capacity utilization
                max_capacity = plant_config['base_capacity']
                capacity_util = min(1.0, demand / max_capacity)
                
                # Supply chain disruptions (5% chance)
                disruption = 1 if np.random.random() < 0.05 else 0
                if disruption:
                    demand = int(demand * 0.7)  # Reduce by 30% during disruption
                
                # Labor availability (varies with holidays and disruptions)
                labor_avail = 0.95 if not is_holiday and not disruption else 0.7
                labor_avail *= (0.9 + 0.2 * np.random.random())
                
                # Inventory and lead times
                inventory = int(1000 + 500 * np.random.random())
                lead_time = max(1, int(7 + 5 * np.random.random()))
                
                # Order backlog
                backlog = max(0, int(100 * np.random.random()))
                
                # Working days in month
                working_days = 22 if date.month != 2 else 20
                
                record = {
                    'date': date.strftime('%Y-%m-%d'),
                    'plant_id': plant_id,
                    'product_line': product,
                    'product_category': plant_config['category'],
                    'demand_units': demand,
                    'raw_material_1': raw_materials[plant_config['category']][0],
                    'raw_material_2': raw_materials[plant_config['category']][1],
                    'raw_material_3': raw_materials[plant_config['category']][2],
                    'raw_material_4': raw_materials[plant_config['category']][3],
                    'economic_index': round(economic_index, 2),
                    'capacity_utilization': round(capacity_util, 3),
                    'supply_chain_disruption': disruption,
                    'working_days_month': working_days,
                    'is_holiday': int(is_holiday),
                    'fuel_cost_index': round(fuel_cost, 2),
                    'labor_availability': round(labor_avail, 3),
                    'inventory_level': inventory,
                    'lead_time_days': lead_time,
                    'order_backlog': backlog
                }
                
                data.append(record)
    
    df = pd.DataFrame(data)
    
    # Save realistic sample data
    df.to_csv('data/realistic_manufacturing_data.csv', index=False)
    print(f"✅ Created realistic sample data: {len(df)} records")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"🏭 Plants: {df['plant_id'].unique().tolist()}")
    print(f"📦 Categories: {df['product_category'].unique().tolist()}")
    
    return df

def demonstrate_erp_integration():
    """
    Demonstrate how to integrate data from different ERP systems
    """
    print("\n" + "="*60)
    print("🔧 ERP SYSTEM INTEGRATION EXAMPLES")
    print("="*60)
    
    integrator = RealDataIntegrator()
    
    # Example 1: SAP Integration
    print("\n📊 SAP ERP Integration Example:")
    print("Column mapping from SAP fields to our format:")
    for sap_field, our_field in list(SAP_COLUMN_MAPPING.items())[:5]:
        print(f"   {sap_field} → {our_field}")
    
    # Example 2: Oracle Integration  
    print("\n📊 Oracle ERP Integration Example:")
    print("Column mapping from Oracle fields to our format:")
    for oracle_field, our_field in list(ORACLE_COLUMN_MAPPING.items())[:5]:
        print(f"   {oracle_field} → {our_field}")
    
    # Example 3: Excel/CSV Integration
    print("\n📊 Excel/CSV Integration Example:")
    print("For Excel files, map your columns like this:")
    excel_mapping = {
        'Production Date': 'date',
        'Plant': 'plant_id', 
        'Product Code': 'product_line',
        'Category': 'product_category',
        'Quantity': 'demand_units'
    }
    for excel_col, our_col in excel_mapping.items():
        print(f"   '{excel_col}' → '{our_col}'")

def validate_and_process_real_data(file_path):
    """
    Validate and process real manufacturing data
    """
    print(f"\n🔍 Processing real data from: {file_path}")
    
    integrator = RealDataIntegrator()
    
    # Load data
    if file_path.endswith('.xlsx'):
        df = integrator.load_from_excel(file_path)
    else:
        df = integrator.load_from_csv(file_path)
    
    if df is not None:
        print(f"📊 Loaded {len(df)} records")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"🏭 Unique plants: {df['plant_id'].nunique()}")
        print(f"📦 Unique products: {df['product_line'].nunique()}")
        
        # Validate format
        if integrator.validate_data_format(df, 'demand_data'):
            # Clean and process
            cleaned_df = integrator.clean_and_process_demand_data(df)
            
            # Replace the original synthetic data
            cleaned_df.to_csv('data/manufacturing_demand_data.csv', index=False)
            print("✅ Real data successfully integrated!")
            print("📂 Saved to: data/manufacturing_demand_data.csv")
            
            return True
        else:
            print("❌ Data validation failed")
            return False
    else:
        return False

def main():
    """
    Main integration workflow
    """
    print("🏭 REAL MANUFACTURING DATA INTEGRATION")
    print("="*50)
    
    # Step 1: Create realistic sample data (replace this with your real data loading)
    sample_df = create_sample_real_data()
    
    # Step 2: Demonstrate ERP integration approaches
    demonstrate_erp_integration()
    
    # Step 3: Process the sample data (replace 'realistic_manufacturing_data.csv' with your real file)
    print("\n" + "="*60)
    print("🔄 PROCESSING SAMPLE DATA")
    print("="*60)
    
    if validate_and_process_real_data('data/realistic_manufacturing_data.csv'):
        print("\n✅ Integration completed successfully!")
        print("\n📋 Next steps:")
        print("1. Replace 'realistic_manufacturing_data.csv' with your actual data file")
        print("2. Modify the column mapping based on your data structure")
        print("3. Run the dashboard: streamlit run manufacturing_app.py")
        print("4. Retrain models with real data using the Jupyter notebook")
    else:
        print("\n❌ Integration failed. Please check your data format.")
    
    print("\n" + "="*60)
    print("📚 INTEGRATION GUIDE")
    print("="*60)
    print("1. 📄 See REAL_DATA_GUIDE.md for detailed instructions")
    print("2. 🔧 Modify data/erp_config.py for your ERP system")
    print("3. 🔄 Use this script as a template for your data")
    print("4. ✅ Validate data format before processing")
    print("5. 🏭 Test with small dataset first")

if __name__ == "__main__":
    main()
