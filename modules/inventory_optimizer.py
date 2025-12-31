"""
Inventory Optimization Module
Advanced inventory management calculations for manufacturing demand forecasting
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class InventoryOptimizer:
    """
    Advanced inventory optimization calculations for manufacturing plants
    """
    
    def __init__(self):
        self.service_level = 0.95  # Default 95% service level
        self.z_score = stats.norm.ppf(self.service_level)  # Z-score for service level
        
    def calculate_demand_statistics(self, demand_data, plant_id=None, product_category=None):
        """
        Calculate demand statistics for inventory optimization
        
        Args:
            demand_data: DataFrame with demand history
            plant_id: Optional plant filter
            product_category: Optional category filter
            
        Returns:
            Dict with demand statistics
        """
        # Filter data if specified
        filtered_data = demand_data.copy()
        if plant_id:
            filtered_data = filtered_data[filtered_data['plant_id'] == plant_id]
        if product_category:
            filtered_data = filtered_data[filtered_data['product_category'] == product_category]
        
        # Calculate statistics
        stats_dict = {
            'avg_daily_demand': filtered_data['demand_units'].mean(),
            'demand_std': filtered_data['demand_units'].std(),
            'demand_variance': filtered_data['demand_units'].var(),
            'coefficient_of_variation': filtered_data['demand_units'].std() / filtered_data['demand_units'].mean(),
            'max_demand': filtered_data['demand_units'].max(),
            'min_demand': filtered_data['demand_units'].min(),
            'demand_trend': self._calculate_trend(filtered_data)
        }
        
        return stats_dict
    
    def _calculate_trend(self, data):
        """Calculate demand trend over time"""
        if len(data) < 2:
            return 0
        
        # Sort by date and calculate linear trend
        data_sorted = data.sort_values('date')
        x = np.arange(len(data_sorted))
        y = data_sorted['demand_units'].values
        
        # Linear regression slope
        if len(x) > 1:
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope
        return 0
    
    def calculate_safety_stock(self, avg_demand, demand_std, lead_time_days, service_level=None):
        """
        Calculate safety stock using statistical method
        
        Args:
            avg_demand: Average daily demand
            demand_std: Standard deviation of demand
            lead_time_days: Lead time in days
            service_level: Service level (default uses class setting)
            
        Returns:
            Safety stock quantity
        """
        if service_level:
            z_score = stats.norm.ppf(service_level)
        else:
            z_score = self.z_score
        
        # Safety stock = Z-score * sqrt(lead_time) * demand_std
        safety_stock = z_score * np.sqrt(lead_time_days) * demand_std
        
        return max(0, safety_stock)  # Ensure non-negative
    
    def calculate_reorder_point(self, avg_demand, lead_time_days, safety_stock):
        """
        Calculate reorder point
        
        Args:
            avg_demand: Average daily demand
            lead_time_days: Lead time in days
            safety_stock: Safety stock quantity
            
        Returns:
            Reorder point quantity
        """
        # Reorder Point = (Average daily demand * Lead time) + Safety stock
        reorder_point = (avg_demand * lead_time_days) + safety_stock
        
        return max(0, reorder_point)
    
    def calculate_eoq(self, annual_demand, ordering_cost, holding_cost_per_unit):
        """
        Calculate Economic Order Quantity (EOQ)
        
        Args:
            annual_demand: Annual demand quantity
            ordering_cost: Cost per order
            holding_cost_per_unit: Annual holding cost per unit
            
        Returns:
            EOQ quantity
        """
        if holding_cost_per_unit <= 0:
            return 0
        
        # EOQ = sqrt((2 * Annual Demand * Ordering Cost) / Holding Cost per Unit)
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
        
        return max(1, eoq)  # Minimum order of 1
    
    def calculate_total_inventory_cost(self, order_quantity, annual_demand, 
                                     ordering_cost, holding_cost_per_unit):
        """
        Calculate total inventory cost for given order quantity
        
        Returns:
            Dictionary with cost breakdown
        """
        if order_quantity <= 0:
            return {'total_cost': float('inf'), 'ordering_cost': 0, 'holding_cost': 0}
        
        # Annual ordering cost = (Annual demand / Order quantity) * Ordering cost
        annual_ordering_cost = (annual_demand / order_quantity) * ordering_cost
        
        # Annual holding cost = (Order quantity / 2) * Holding cost per unit
        annual_holding_cost = (order_quantity / 2) * holding_cost_per_unit
        
        total_cost = annual_ordering_cost + annual_holding_cost
        
        return {
            'total_cost': total_cost,
            'ordering_cost': annual_ordering_cost,
            'holding_cost': annual_holding_cost,
            'order_quantity': order_quantity
        }
    
    def optimize_inventory_for_plant(self, demand_data, plant_capacity_data, 
                                   raw_materials_data, plant_id):
        """
        Complete inventory optimization for a specific plant
        
        Args:
            demand_data: Historical demand data
            plant_capacity_data: Plant capacity information
            raw_materials_data: Raw materials master data
            plant_id: Plant to optimize
            
        Returns:
            Dictionary with optimization results
        """
        plant_data = demand_data[demand_data['plant_id'] == plant_id]
        
        if len(plant_data) == 0:
            return {'error': f'No data found for plant {plant_id}'}
        
        optimization_results = {}
        
        # Group by product category for optimization
        for category in plant_data['product_category'].unique():
            category_data = plant_data[plant_data['product_category'] == category]
            
            # Calculate demand statistics
            demand_stats = self.calculate_demand_statistics(category_data)
            
            # Get lead time from raw materials data (default 7 days if not available)
            lead_time = 7  # Default
            if len(raw_materials_data) > 0:
                material_lead_times = raw_materials_data['lead_time_days'].mean()
                if not pd.isna(material_lead_times):
                    lead_time = material_lead_times
            
            # Calculate inventory metrics
            safety_stock = self.calculate_safety_stock(
                demand_stats['avg_daily_demand'],
                demand_stats['demand_std'],
                lead_time
            )
            
            reorder_point = self.calculate_reorder_point(
                demand_stats['avg_daily_demand'],
                lead_time,
                safety_stock
            )
            
            # Estimate costs (these could be made configurable)
            annual_demand = demand_stats['avg_daily_demand'] * 365
            ordering_cost = 100  # Default ordering cost
            holding_cost_per_unit = 5  # Default holding cost
            
            eoq = self.calculate_eoq(annual_demand, ordering_cost, holding_cost_per_unit)
            
            # Calculate cost analysis
            eoq_cost = self.calculate_total_inventory_cost(
                eoq, annual_demand, ordering_cost, holding_cost_per_unit
            )
            
            # Current inventory level (use latest demand as proxy)
            current_inventory = category_data['demand_units'].iloc[-1] * 3  # Assume 3 days stock
            
            optimization_results[category] = {
                'demand_stats': demand_stats,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'eoq': eoq,
                'lead_time_days': lead_time,
                'current_inventory': current_inventory,
                'cost_analysis': eoq_cost,
                'inventory_status': self._get_inventory_status(current_inventory, reorder_point),
                'recommendations': self._generate_recommendations(
                    current_inventory, reorder_point, safety_stock, eoq
                )
            }
        
        return optimization_results
    
    def _get_inventory_status(self, current_inventory, reorder_point):
        """Determine inventory status"""
        if current_inventory <= reorder_point * 0.5:
            return 'CRITICAL'
        elif current_inventory <= reorder_point:
            return 'REORDER_NEEDED'
        elif current_inventory <= reorder_point * 1.5:
            return 'OPTIMAL'
        else:
            return 'OVERSTOCK'
    
    def _generate_recommendations(self, current_inventory, reorder_point, safety_stock, eoq):
        """Generate inventory recommendations"""
        recommendations = []
        
        if current_inventory <= reorder_point:
            order_quantity = max(eoq, reorder_point + safety_stock - current_inventory)
            recommendations.append(f"🚨 ORDER NOW: {order_quantity:.0f} units")
        
        if current_inventory < safety_stock:
            recommendations.append("⚠️ Below safety stock - urgent restocking needed")
        
        if current_inventory > reorder_point * 2:
            recommendations.append("📦 Consider reducing order quantities - possible overstock")
        
        if len(recommendations) == 0:
            recommendations.append("✅ Inventory levels are optimal")
        
        return recommendations
    
    def calculate_abc_analysis(self, demand_data):
        """
        Perform ABC analysis on inventory items
        
        Returns:
            DataFrame with ABC classification
        """
        # Calculate annual demand value by product category
        annual_demand = demand_data.groupby('product_category').agg({
            'demand_units': lambda x: x.sum() * (365 / len(demand_data['date'].unique()))
        }).reset_index()
        
        # Assume unit value (could be made configurable)
        annual_demand['unit_value'] = 100  # Default unit value
        annual_demand['annual_value'] = annual_demand['demand_units'] * annual_demand['unit_value']
        
        # Sort by annual value
        annual_demand = annual_demand.sort_values('annual_value', ascending=False)
        
        # Calculate cumulative percentage
        total_value = annual_demand['annual_value'].sum()
        annual_demand['cumulative_value'] = annual_demand['annual_value'].cumsum()
        annual_demand['cumulative_percentage'] = (annual_demand['cumulative_value'] / total_value) * 100
        
        # Classify into ABC categories
        annual_demand['abc_category'] = 'C'
        annual_demand.loc[annual_demand['cumulative_percentage'] <= 80, 'abc_category'] = 'A'
        annual_demand.loc[(annual_demand['cumulative_percentage'] > 80) & 
                         (annual_demand['cumulative_percentage'] <= 95), 'abc_category'] = 'B'
        
        return annual_demand
    
    def generate_inventory_report(self, optimization_results, plant_id):
        """
        Generate comprehensive inventory optimization report
        
        Returns:
            Dictionary with report data
        """
        report = {
            'plant_id': plant_id,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_categories': len(optimization_results),
            'summary': {
                'critical_items': 0,
                'reorder_needed': 0,
                'optimal_items': 0,
                'overstock_items': 0,
                'total_safety_stock_value': 0,
                'total_eoq_value': 0
            },
            'categories': optimization_results
        }
        
        # Calculate summary statistics
        for category, data in optimization_results.items():
            status = data['inventory_status']
            if status == 'CRITICAL':
                report['summary']['critical_items'] += 1
            elif status == 'REORDER_NEEDED':
                report['summary']['reorder_needed'] += 1
            elif status == 'OPTIMAL':
                report['summary']['optimal_items'] += 1
            elif status == 'OVERSTOCK':
                report['summary']['overstock_items'] += 1
            
            # Estimate values (using default unit cost)
            unit_cost = 50  # Default unit cost
            report['summary']['total_safety_stock_value'] += data['safety_stock'] * unit_cost
            report['summary']['total_eoq_value'] += data['eoq'] * unit_cost
        
        return report
