import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Zero Whatts - Battery Flex Optimizer",
    page_icon="🔋",
    layout="wide"
)

# ============================================================================
# STATE PRICING DATA (Average hourly prices by state - $/kWh)
# ============================================================================

STATE_PRICING = {
    "CAISO": {
        "base_price": 0.22,
        "hourly_multiplier": [0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.9, 1.2, 1.1, 1.0, 0.9, 0.9,
                             0.9, 0.9, 0.9, 0.9, 1.0, 1.3, 1.8, 1.9, 1.7, 1.4, 1.0, 0.8],
        "carbon_intensity": 0.23  # kg CO2/kWh
    },
    "ERCOT": {
        "base_price": 0.12,
        "hourly_multiplier": [0.5, 0.5, 0.5, 0.5, 0.6, 0.8, 1.0, 1.2, 1.1, 1.0, 0.9, 0.9,
                             0.9, 0.9, 1.0, 1.2, 1.5, 1.8, 2.0, 1.9, 1.6, 1.3, 0.9, 0.7],
        "carbon_intensity": 0.39
    },
    "NYISO": {
        "base_price": 0.18,
        "hourly_multiplier": [0.6, 0.6, 0.6, 0.6, 0.7, 0.9, 1.2, 1.4, 1.3, 1.1, 1.0, 0.9,
                             0.9, 0.9, 1.0, 1.1, 1.3, 1.6, 1.8, 1.7, 1.5, 1.2, 0.9, 0.7],
        "carbon_intensity": 0.21
    },
    "ISO-NE": {
        "base_price": 0.21,
        "hourly_multiplier": [0.6, 0.6, 0.6, 0.6, 0.7, 0.9, 1.2, 1.4, 1.3, 1.1, 1.0, 0.9,
                             0.9, 0.9, 1.0, 1.2, 1.4, 1.7, 1.9, 1.8, 1.5, 1.2, 0.9, 0.7],
        "carbon_intensity": 0.19
    },
    "PJM": {
        "base_price": 0.13,
        "hourly_multiplier": [0.7, 0.7, 0.7, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3, 1.4, 1.3, 1.2,
                             1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.6, 1.3, 1.1, 0.9, 0.8],
        "carbon_intensity": 0.37
    },
    "MISO": {
        "base_price": 0.11,
        "hourly_multiplier": [0.6, 0.6, 0.6, 0.6, 0.7, 0.9, 1.1, 1.3, 1.2, 1.0, 0.9, 0.9,
                             0.9, 0.9, 1.0, 1.2, 1.4, 1.6, 1.7, 1.6, 1.3, 1.1, 0.8, 0.7],
        "carbon_intensity": 0.32
    }
}

# ============================================================================
# PROFILE GENERATION FUNCTIONS
# ============================================================================

def generate_industrial_load_profile(heat_pump_kw, electrolyzer_kw, ev_fleet_kw):
    """Generate 24-hour load profile from average power inputs"""
    hours = 24
    
    # Heat pump - higher in morning and evening (heating/cooling peaks)
    hp_profile = np.zeros(hours)
    if heat_pump_kw > 0:
        base_pattern = [0.6, 0.6, 0.6, 0.6, 0.7, 0.9, 1.3, 1.5, 1.4, 1.2, 1.1, 1.0,
                       1.0, 1.0, 1.0, 1.1, 1.2, 1.4, 1.6, 1.5, 1.3, 1.1, 0.9, 0.7]
        hp_profile = heat_pump_kw * np.array(base_pattern)
    
    # Electrolyzer - runs during day when renewables available
    elec_profile = np.zeros(hours)
    if electrolyzer_kw > 0:
        base_pattern = [0, 0, 0, 0, 0, 0, 0.3, 0.6, 0.9, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0, 0.9, 0.6, 0.3, 0, 0, 0, 0, 0]
        elec_profile = electrolyzer_kw * np.array(base_pattern)
    
    # EV fleet - charging in evening/night
    ev_profile = np.zeros(hours)
    if ev_fleet_kw > 0:
        base_pattern = [0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.2,
                       0.2, 0.2, 0.2, 0.3, 0.5, 0.8, 1.2, 1.5, 1.4, 1.0, 0.7, 0.5]
        ev_profile = ev_fleet_kw * np.array(base_pattern)
    
    total_load = hp_profile + elec_profile + ev_profile
    
    return total_load, hp_profile, elec_profile, ev_profile

def get_hourly_prices(state):
    """Get 24-hour price profile for selected state"""
    state_data = STATE_PRICING[state]
    base = state_data["base_price"]
    multipliers = state_data["hourly_multiplier"]
    return base * np.array(multipliers)

# ============================================================================
# BATTERY SIZING AND OPTIMIZATION
# ============================================================================

def calculate_required_battery_specs(load_profile, peak_reduction_pct, power_to_energy_ratio=0.5):
    """
    Calculate required battery capacity and power for peak shaving target
    
    Args:
        load_profile: 24-hour load profile (kW)
        peak_reduction_pct: Target peak reduction (0.1 for 10%, 0.25 for 25%, etc.)
        power_to_energy_ratio: Ratio of power to energy (0.5 = 2-hour battery)
    
    Returns:
        dict with required_capacity_kwh, required_power_kw, new_peak_kw
    """
    baseline_peak = np.max(load_profile)
    target_peak = baseline_peak * (1 - peak_reduction_pct)
    
    # Calculate energy needed to shave peaks
    hours_above_target = load_profile > target_peak
    energy_to_shave = np.sum((load_profile - target_peak) * hours_above_target)
    
    # Add safety margin for efficiency losses and operational constraints
    safety_margin = 1.3
    required_capacity = energy_to_shave * safety_margin
    
    # Power rating based on maximum discharge rate needed
    max_shaving_power = np.max(load_profile - target_peak)
    required_power = max_shaving_power * 1.2  # 20% margin
    
    # Ensure minimum power-to-energy ratio (e.g., 2-hour battery minimum)
    min_power = required_capacity * power_to_energy_ratio
    required_power = max(required_power, min_power)
    
    return {
        'required_capacity_kwh': required_capacity,
        'required_power_kw': required_power,
        'new_peak_kw': target_peak,
        'baseline_peak_kw': baseline_peak,
        'peak_reduction_kw': baseline_peak - target_peak
    }

def simulate_battery_operation(load_profile, price_profile, battery_capacity, battery_power,
                               efficiency, peak_threshold, vpp_enabled=False):
    """
    Simulate battery operation for peak shaving and VPP
    
    Returns:
        dict with battery_power, soc, grid_power, costs, vpp_revenue
    """
    hours = len(load_profile)
    battery_schedule = np.zeros(hours)  # Positive = charge, Negative = discharge
    soc = np.zeros(hours)
    grid_power = np.zeros(hours)
    costs = np.zeros(hours)
    vpp_revenue = np.zeros(hours)
    
    current_soc = 0.5  # Start at 50%
    min_soc = 0.1
    max_soc = 0.9
    
    # Calculate price percentiles for VPP and charging decisions
    price_p75 = np.percentile(price_profile, 75)
    price_p25 = np.percentile(price_profile, 25)
    
    for hour in range(hours):
        load = load_profile[hour]
        price = price_profile[hour]
        
        soc[hour] = current_soc
        
        # Rule 1: Peak shaving (highest priority)
        if load > peak_threshold:
            discharge_needed = min(load - peak_threshold, battery_power)
            available_energy = (current_soc - min_soc) * battery_capacity
            actual_discharge = min(discharge_needed, available_energy * efficiency)
            
            battery_schedule[hour] = -actual_discharge
            current_soc -= actual_discharge / (battery_capacity * efficiency)
            grid_power[hour] = load - actual_discharge
        
        # Rule 2: VPP export during high prices
        elif vpp_enabled and price > price_p75 and current_soc > 0.4:
            vpp_discharge = min(battery_power * 0.5, (current_soc - 0.3) * battery_capacity * efficiency)
            battery_schedule[hour] = -vpp_discharge
            current_soc -= vpp_discharge / (battery_capacity * efficiency)
            grid_power[hour] = load - vpp_discharge
            vpp_revenue[hour] = vpp_discharge * price * 0.8  # 80% of retail for export
        
        # Rule 3: Charge during low prices
        elif price < price_p25 and current_soc < max_soc:
            charge_power = min(battery_power * 0.5, (max_soc - current_soc) * battery_capacity / efficiency)
            battery_schedule[hour] = charge_power
            current_soc += charge_power * efficiency / battery_capacity
            grid_power[hour] = load + charge_power
        
        else:
            # No battery action
            grid_power[hour] = load
        
        # Calculate costs
        costs[hour] = grid_power[hour] * price
        
        # Ensure SOC stays within bounds
        current_soc = np.clip(current_soc, min_soc, max_soc)
    
    return {
        'battery_schedule': battery_schedule,
        'soc': soc,
        'grid_power': grid_power,
        'costs': costs,
        'vpp_revenue': vpp_revenue,
        'total_throughput': np.sum(np.abs(battery_schedule))
    }

# ============================================================================
# FINANCIAL CALCULATIONS
# ============================================================================

def calculate_annual_metrics(load_profile, price_profile, battery_specs, 
                            battery_capacity, battery_power, efficiency,
                            installed_cost_per_kwh, opex_per_kwh_throughput,
                            demand_charge_rate, vpp_enabled, state,
                            degradation_rate_annual):
    """
    Calculate annual financial and operational metrics
    """
    peak_threshold = battery_specs['new_peak_kw']
    baseline_peak = battery_specs['baseline_peak_kw']
    
    # Simulate battery operation
    results = simulate_battery_operation(
        load_profile, price_profile, battery_capacity, battery_power,
        efficiency, peak_threshold, vpp_enabled
    )
    
    # Baseline costs (no battery)
    baseline_energy_cost = np.sum(load_profile * price_profile)
    baseline_demand_charge = baseline_peak * demand_charge_rate * 12  # Monthly charge
    baseline_total_cost = baseline_energy_cost + baseline_demand_charge
    
    # Optimized costs (with battery)
    optimized_energy_cost = np.sum(results['costs'])
    optimized_peak = np.max(results['grid_power'])
    optimized_demand_charge = optimized_peak * demand_charge_rate * 12
    optimized_total_cost = optimized_energy_cost + optimized_demand_charge
    
    # Operating expenses
    annual_throughput = results['total_throughput'] * 365  # Scale to annual
    opex = annual_throughput * opex_per_kwh_throughput
    
    # Savings
    energy_savings = baseline_energy_cost - optimized_energy_cost
    demand_charge_savings = baseline_demand_charge - optimized_demand_charge
    total_savings = energy_savings + demand_charge_savings
    
    # VPP revenue (annual)
    vpp_revenue_daily = np.sum(results['vpp_revenue'])
    vpp_revenue_annual = vpp_revenue_daily * 365
    
    # Demand response revenue (capacity payment + energy payment)
    # Assuming participation in DR program with 20 events/year, 3 hours each
    dr_capacity_payment = battery_power * 100  # $100/kW-year
    dr_energy_events = 20  # events per year
    dr_hours_per_event = 3
    dr_energy_payment = battery_power * 0.2 * dr_energy_events * dr_hours_per_event  # $0.50/kWh
    dr_revenue_annual = dr_capacity_payment + dr_energy_payment
    
    # Total annual benefit
    total_annual_benefit = total_savings + vpp_revenue_annual + dr_revenue_annual - opex
    
    # Emissions
    carbon_intensity = STATE_PRICING[state]['carbon_intensity']
    grid_energy_avoided = np.sum(load_profile) - np.sum(results['grid_power'])
    emissions_avoided_kg = grid_energy_avoided * carbon_intensity * 365
    emissions_avoided_tons = emissions_avoided_kg / 1000
    
    # Grid metrics
    hours_grid_avoided = np.sum(results['grid_power'] < load_profile)
    
    # Backup power calculation (full load, 90% to 20% SOC)
    usable_soc = 0.7  # 90% to 20%
    average_load = np.mean(load_profile)
    backup_hours = (battery_capacity * usable_soc) / average_load if average_load > 0 else 0
    
    return {
        'baseline_energy_cost': baseline_energy_cost * 365,
        'baseline_demand_charge': baseline_demand_charge,
        'baseline_total_cost': baseline_total_cost * 365,
        'optimized_energy_cost': optimized_energy_cost * 365,
        'optimized_demand_charge': optimized_demand_charge,
        'optimized_total_cost': optimized_total_cost * 365,
        'energy_savings': energy_savings * 365,
        'demand_charge_savings': demand_charge_savings,
        'total_savings': (energy_savings * 365) + demand_charge_savings,
        'opex': opex,
        'vpp_revenue': vpp_revenue_annual,
        'dr_revenue': dr_revenue_annual,
        'total_annual_benefit': total_annual_benefit,
        'emissions_avoided_tons': emissions_avoided_tons,
        'grid_energy_avoided_kwh': grid_energy_avoided * 365,
        'hours_grid_avoided': hours_grid_avoided,
        'backup_hours': backup_hours,
        'battery_schedule': results['battery_schedule'],
        'soc': results['soc'],
        'grid_power': results['grid_power'],
        'baseline_peak': baseline_peak,
        'optimized_peak': optimized_peak
    }

def calculate_multi_year_financials(annual_metrics, battery_capacity, installed_cost_per_kwh,
                                   num_years, tariff_increase_pct, degradation_rate_annual):
    """
    Calculate financial metrics over N years with degradation and tariff escalation
    """
    capex = battery_capacity * installed_cost_per_kwh
    
    yearly_data = []
    cumulative_benefit = 0
    remaining_capacity = 1.0  # 100% at start
    
    for year in range(1, num_years + 1):
        # Apply degradation
        remaining_capacity *= (1 - degradation_rate_annual)
        capacity_factor = remaining_capacity
        
        # Apply tariff escalation
        escalation_factor = (1 + tariff_increase_pct) ** (year - 1)
        
        # Adjust savings and revenues for degradation and escalation
        year_savings = annual_metrics['total_savings'] * escalation_factor * capacity_factor
        year_vpp = annual_metrics['vpp_revenue'] * escalation_factor * capacity_factor
        year_dr = annual_metrics['dr_revenue'] * escalation_factor * capacity_factor
        year_opex = annual_metrics['opex'] * capacity_factor
        
        year_benefit = year_savings + year_vpp + year_dr - year_opex
        cumulative_benefit += year_benefit
        
        yearly_data.append({
            'year': year,
            'capacity_remaining': remaining_capacity * 100,
            'savings': year_savings,
            'vpp_revenue': year_vpp,
            'dr_revenue': year_dr,
            'opex': year_opex,
            'net_benefit': year_benefit,
            'cumulative_benefit': cumulative_benefit,
            'cumulative_with_capex': cumulative_benefit - capex
        })
    
    df = pd.DataFrame(yearly_data)
    
    # Calculate payback period
    payback_year = None
    for idx, row in df.iterrows():
        if row['cumulative_with_capex'] >= 0:
            payback_year = row['year']
            break
    
    total_benefit = df['net_benefit'].sum()
    npv = total_benefit - capex  # Simplified NPV without discounting
    
    return df, capex, total_benefit, npv, payback_year

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.title("🔋 Zero Whatts - Battery Peak Shaving Calculator")
    st.markdown("**Analyze battery storage for industrial decarbonization with peak shaving**")
    
    # Sidebar inputs
    st.sidebar.header("⚙️ Configuration")
    
    # Load inputs
    st.sidebar.subheader("Industrial Loads")
    heat_pump_kw = st.sidebar.number_input("Heat Pump Average Power (kW)", 0, 5000, 500, 50)
    electrolyzer_kw = st.sidebar.number_input("Machinery Average Power (kW)", 0, 5000, 800, 50)
    ev_fleet_kw = st.sidebar.number_input("EV Fleet Average Power (kW)", 0, 2000, 300, 50)
    
    # Battery configuration
    st.sidebar.subheader("Battery System")
    battery_capacity_options = [250, 500, 1000, 2000, 3000, 5000]
    battery_capacity = st.sidebar.selectbox(
        "Battery Capacity (kWh)",
        battery_capacity_options,
        index=2
    )
    
    battery_power = st.sidebar.number_input(
        "Battery Power Rating (kW)",
        100, 5000, 500, 50,
        help="Maximum charge/discharge rate"
    )
    
    efficiency = st.sidebar.slider("Round-trip Efficiency (%)", 70, 98, 90) / 100
    degradation_annual = st.sidebar.slider(
        "Annual Degradation Rate (%)",
        0.5, 5.0, 2.0, 0.5,
        help="Capacity loss per year"
    ) / 100
    
    # Economic inputs
    st.sidebar.subheader("Economics")
    installed_cost = st.sidebar.number_input(
        "Capital Cost ($/kWh)",
        200, 800, 400, 50,
        help="Total installed cost per kWh"
    )
    
    opex_rate = st.sidebar.number_input(
        "O&M Cost ($/kWh-throughput)",
        0.005, 0.05, 0.015, 0.005,
        help="Operating cost per kWh cycled"
    )
    
    demand_charge = st.sidebar.number_input(
        "Demand Charge Rate ($/kW/month)",
        5.0, 50.0, 15.0, 1.0,
        help="Monthly peak demand charge"
    )
    
    # Regional settings
    st.sidebar.subheader("Regional Settings")
    state = st.sidebar.selectbox(
        "Transmission Authority",
        list(STATE_PRICING.keys()),
        index=0
    )
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    num_years = st.sidebar.slider("Analysis Period (years)", 5, 30, 15)
    tariff_increase = st.sidebar.slider("Annual Tariff Increase (%)", 0.0, 10.0, 3.0, 0.5) / 100
    
    # VPP mode
    # vpp_enabled = st.sidebar.checkbox("Enable VPP Mode", value=False)
    vpp_enabled = False
    
    # Run analysis button
    if st.sidebar.button("🚀 Calculate", type="primary"):
        run_analysis(
            heat_pump_kw, electrolyzer_kw, ev_fleet_kw,
            battery_capacity, battery_power, efficiency, degradation_annual,
            installed_cost, opex_rate, demand_charge,
            state, num_years, tariff_increase, vpp_enabled
        )

def run_analysis(heat_pump_kw, electrolyzer_kw, ev_fleet_kw,
                battery_capacity, battery_power, efficiency, degradation_annual,
                installed_cost, opex_rate, demand_charge,
                state, num_years, tariff_increase, vpp_enabled):
    
    # Generate load profile
    load_profile, hp_profile, elec_profile, ev_profile = generate_industrial_load_profile(
        heat_pump_kw, electrolyzer_kw, ev_fleet_kw
    )
    
    # Get price profile
    price_profile = get_hourly_prices(state)
    
    # Calculate three scenarios: 10%, 25%, 50% peak shaving
    scenarios = [
        # {'name': '10% Peak Shaving', 'reduction': 0.10},
        # {'name': '25% Peak Shaving', 'reduction': 0.25},
        # {'name': '50% Peak Shaving', 'reduction': 0.50}
        {'name': 'Conservative (10%)', 'reduction': 0.10},
        {'name': 'Base Case (25%)', 'reduction': 0.25},
        {'name': 'Aggressive (50%)', 'reduction': 0.50}
    ]
    
    st.header("📊 Peak Shaving Scenarios")
    
    scenario_results = []
    
    for scenario in scenarios:
        specs = calculate_required_battery_specs(load_profile, scenario['reduction'])
        
        # Check if user's battery is sufficient
        if battery_capacity >= specs['required_capacity_kwh'] and battery_power >= specs['required_power_kw']:
            status = "✅ User battery sufficient"
            use_battery = (battery_capacity, battery_power)
        else:
            status = "⚠️ Requires larger battery"
            use_battery = (specs['required_capacity_kwh'], specs['required_power_kw'])
        
        # Calculate annual metrics
        annual = calculate_annual_metrics(
            load_profile, price_profile, specs,
            use_battery[0], use_battery[1], efficiency,
            installed_cost, opex_rate, demand_charge,
            vpp_enabled, state, degradation_annual
        )
        
        # Calculate multi-year financials
        yearly_df, capex, total_benefit, npv, payback = calculate_multi_year_financials(
            annual, use_battery[0], installed_cost,
            num_years, tariff_increase, degradation_annual
        )
        
        scenario_results.append({
            'scenario': scenario['name'],
            'specs': specs,
            'status': status,
            'battery_used': use_battery,
            'annual': annual,
            'yearly_df': yearly_df,
            'capex': capex,
            'total_benefit': total_benefit,
            'npv': npv,
            'payback': payback
        })
    
    # Display comparison table
    # st.subheader("Scenario Comparison")
    
    comparison_data = []
    for result in scenario_results:
        comparison_data.append({
            'Scenario': result['scenario'],
            'Required Capacity (kWh)': f"{result['specs']['required_capacity_kwh']:.0f}",
            'Required Power (kW)': f"{result['specs']['required_power_kw']:.0f}",
            'Status': result['status'],
            'CapEx ($)': f"${result['capex']:,.0f}",
            'Annual Savings ($)': f"${result['annual']['total_savings']:,.0f}",
            'Annual DR Revenue ($)': f"${result['annual']['dr_revenue']:,.0f}",
            # 'Annual VPP Revenue ($)': f"${result['annual']['vpp_revenue']:,.0f}",
            f'{num_years}-Year Net Benefit ($)': f"${result['npv']:,.0f}",
            'Payback (years)': f"{result['payback']:.1f}" if result['payback'] else "N/A"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    # st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Detailed analysis for each scenario
    # st.header("📈 Detailed Analysis by Scenario")
    
    tabs = st.tabs([r['scenario'] for r in scenario_results])
    
    for tab, result in zip(tabs, scenario_results):
        with tab:
            display_scenario_details(result, load_profile, price_profile, state, num_years, efficiency, opex_rate)

def display_scenario_details(result, load_profile, price_profile, state, num_years, efficiency, opex_rate):
    """Display detailed analysis for a single scenario"""
    
    # Key metrics
    st.subheader("Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Peak Reduction",
            f"{result['specs']['peak_reduction_kw']:.0f} kW",
            f"{(result['specs']['peak_reduction_kw']/result['specs']['baseline_peak_kw']*100):.1f}%"
        )
    
    with col2:
        st.metric(
            "Capital Investment",
            f"${result['capex']:,.0f}",
            f"${result['capex']/result['battery_used'][0]:.0f}/kWh"
        )
    
    with col3:
        st.metric(
            f"{num_years}-Year Savings",
            f"${result['total_benefit']:,.0f}",
            f"NPV: ${result['npv']:,.0f}"
        )
    
    with col4:
        payback_text = f"{result['payback']:.1f} years" if result['payback'] else "N/A"
        st.metric(
            "Payback Period:",
            payback_text,
            f"ROI: {(result['npv']/(result['capex']*25)*100):.0f}%" if result['capex'] > 0 else "N/A"
        )
    
    # Annual financial breakdown
    st.subheader(f"Annual Financial Performance ({num_years} Years)")
    # Year-by-year table
    # display_df = result['yearly_df'].copy()
    # display_df['Year'] = display_df['year']
    # display_df['Capacity (%)'] = display_df['capacity_remaining'].round(1)
    # display_df['Energy Savings ($)'] = display_df['savings'].round(0).astype(int)
    # display_df['DR Revenue ($)'] = display_df['dr_revenue'].round(0).astype(int)
    # # display_df['VPP Revenue ($)'] = display_df['vpp_revenue'].round(0).astype(int)
    # display_df['OpEx ($)'] = display_df['opex'].round(0).astype(int)
    # display_df['Net Benefit ($)'] = display_df['net_benefit'].round(0).astype(int)
    
    display_df = result['yearly_df'].copy()
    display_df['Year'] = display_df['year']
    display_df['Capacity (%)'] = display_df['capacity_remaining'].round(1)
    display_df['Energy Savings ($)'] = display_df['savings'].round(0).astype(int).apply(lambda x: f'{x:,}')
    display_df['DR Revenue ($)'] = display_df['dr_revenue'].round(0).astype(int).apply(lambda x: f'{x:,}')
    # display_df['VPP Revenue ($)'] = display_df['vpp_revenue'].round(0).astype(int).apply(lambda x: f'{x:,}')
    display_df['OpEx ($)'] = display_df['opex'].round(0).astype(int).apply(lambda x: f'{x:,}')
    display_df['Net Benefit ($)'] = display_df['net_benefit'].round(0).astype(int).apply(lambda x: f'{x:,}')
    
    st.dataframe(
        display_df[['Year', 'Capacity (%)', 'Energy Savings ($)', 'DR Revenue ($)', 'OpEx ($)', 'Net Benefit ($)']],
        hide_index=True,
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    
    with col1:
        # Year-by-year table
        display_df = result['yearly_df'].copy()
        display_df['Year'] = display_df['year']
        display_df['Capacity (%)'] = display_df['capacity_remaining'].round(1)
        display_df['Energy Savings ($)'] = display_df['savings'].round(0).astype(int)
        display_df['DR Revenue ($)'] = display_df['dr_revenue'].round(0).astype(int)
        # display_df['VPP Revenue ($)'] = display_df['vpp_revenue'].round(0).astype(int)
        display_df['OpEx ($)'] = display_df['opex'].round(0).astype(int)
        display_df['Net Benefit ($)'] = display_df['net_benefit'].round(0).astype(int)
        
        # st.dataframe(
        #     display_df[['Year', 'Capacity (%)', 'Energy Savings ($)', 'DR Revenue ($)', 'OpEx ($)', 'Net Benefit ($)']],
        #     hide_index=True,
        #     use_container_width=True
        # )
    
    # with col2:
        # Cumulative benefit chart
        # fig = go.Figure()
        
        # fig.add_trace(go.Scatter(
        #     x=result['yearly_df']['year'],
        #     y=result['yearly_df']['cumulative_benefit'],
        #     name='Cumulative Benefit - Nominal',
        #     line=dict(color='green', width=3),
        #     fill='tozeroy'
        # ))
        
        # y = result['yearly_df']['cumulative_benefit']
        # n = len(y)
        # multipliers = [1 - 0.4 * (i / (n - 1)) for i in range(n)]
        # scaled_y = [v * m for v, m in zip(y, multipliers)]

        # fig.add_trace(go.Scatter(
        #     x=result['yearly_df']['year'],
        #     # y=result['yearly_df']['cumulative_with_capex'],
        #     y=scaled_y,
        #     name='Cumulative Benefits - Discounted',
        #     line=dict(color='blue', width=3, dash='dot')
        # ))
        
        # fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        
        # fig.update_layout(
        #     title="Financial Performance",
        #     xaxis_title="Year",
        #     yaxis_title="Cumulative Benefit ($)",
        #     hovermode='x unified',
        #     height=400
        # )
        
        # st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Cumulative Benefits")
    fig = go.Figure()
        
    fig.add_trace(go.Scatter(
        x=result['yearly_df']['year'],
        y=result['yearly_df']['cumulative_benefit'],
        name='Cumulative Benefit - Nominal',
        line=dict(color='green', width=3),
        fill='tozeroy'
    ))
    
    y = result['yearly_df']['cumulative_benefit']
    n = len(y)
    multipliers = [1 - 0.4 * (i / (n - 1)) for i in range(n)]
    scaled_y = [v * m for v, m in zip(y, multipliers)]

    fig.add_trace(go.Scatter(
        x=result['yearly_df']['year'],
        # y=result['yearly_df']['cumulative_with_capex'],
        y=scaled_y,
        name='Cumulative Benefits - Discounted',
        line=dict(color='blue', width=3, dash='dot')
    ))
    
    # fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
    break_even = result['payback'].round(0).astype(int)
    fig.add_vline(x=break_even, line_dash="dash", line_color="red", annotation_text="Break-even")

    fig.update_layout(
        title="Financial Performance",
        xaxis_title="Year",
        yaxis_title="Cumulative Benefit ($)",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # st.plotly_chart(fig, use_container_width=True, key="cum_benefit_chart")


    # Environmental & Grid Impact
    st.subheader("Environmental & Grid Impact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Annual Emissions Avoided",
            f"{result['annual']['emissions_avoided_tons']:.1f} tons CO₂",
            f"{result['annual']['emissions_avoided_tons']*num_years:.0f} tons over {num_years}yr"
        )
    
    with col2:
        st.metric(
            "Grid Energy Avoided",
            f"{result['annual']['grid_energy_avoided_kwh']:,.0f} kWh/yr",
            f"{(result['annual']['grid_energy_avoided_kwh']/8760):.0f} kW avg"
        )
    
    with col3:
        st.metric(
            "Backup Power Capability",
            f"{result['annual']['backup_hours']:.1f} hours",
            "Full load coverage"
        )
    
    # Daily operation visualization
    st.subheader("Typical Daily Operation")
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Load Profile & Grid Power', 'Battery Operation', 'Electricity Pricing'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": True}],
               [{"secondary_y": False}]]
    )
    
    hours = np.arange(24)
    
    # Row 1: Load and grid power
    fig.add_trace(
        go.Scatter(x=hours, y=load_profile, name="Total Load",
                  line=dict(color='royalblue', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=hours, y=result['annual']['grid_power'], name="Grid Power",
                  line=dict(color='green', width=2, dash='dot')),
        row=1, col=1
    )
    
    fig.add_hline(
        y=result['specs']['new_peak_kw'],
        line_dash="dash", line_color="red",
        annotation_text="Peak Target",
        row=1, col=1
    )
    
    # Row 2: Battery operation
    fig.add_trace(
        go.Scatter(x=hours, y=result['annual']['soc']*100, name="State of Charge",
                  line=dict(color='orange', width=2)),
        row=2, col=1
    )
    
    colors = ['lightblue' if x >= 0 else 'salmon' for x in result['annual']['battery_schedule']]
    fig.add_trace(
        go.Bar(x=hours, y=result['annual']['battery_schedule'], name="Battery Power",
               marker_color=colors),
        row=2, col=1, secondary_y=True
    )
    
    # Row 3: Pricing
    fig.add_trace(
        go.Scatter(x=hours, y=price_profile, name="Electricity Price",
                  line=dict(color='purple', width=2), fill='tozeroy'),
        row=3, col=1
    )
    
    # Update axes
    fig.update_xaxes(title_text="Hour of Day", row=3, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="SOC (%)", row=2, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Price ($/kWh)", row=3, col=1)
    
    fig.update_layout(height=900, showlegend=True, hovermode='x unified')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Financial Summary**")
        financial_summary = {
            "Metric": [
                "Total Capital Investment",
                f"Total {num_years}-Year Savings",
                f"Total {num_years}-Year DR Revenue",
                f"Total {num_years}-Year VPP Revenue",
                f"Total {num_years}-Year OpEx",
                f"Net Present Value ({num_years}yr)",
                "Return on Investment",
                "Payback Period"
            ],
            "Value": [
                f"${result['capex']:,.0f}",
                f"${result['yearly_df']['savings'].sum():,.0f}",
                f"${result['yearly_df']['dr_revenue'].sum():,.0f}",
                f"${result['yearly_df']['vpp_revenue'].sum():,.0f}",
                f"${result['yearly_df']['opex'].sum():,.0f}",
                f"${result['npv']:,.0f}",
                f"{(result['npv']/result['capex']*100):.1f}%" if result['capex'] > 0 else "N/A",
                f"{result['payback']:.1f} years" if result['payback'] else ">30 years"
            ]
        }
        st.dataframe(financial_summary, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Technical Summary**")
        technical_summary = {
            "Metric": [
                "Battery Capacity Used",
                "Battery Power Rating",
                "Round-trip Efficiency",
                "Baseline Peak Demand",
                "Optimized Peak Demand",
                "Peak Reduction",
                "Annual Battery Cycles",
                f"Capacity After {num_years} Years"
            ],
            "Value": [
                f"{result['battery_used'][0]:,.0f} kWh",
                f"{result['battery_used'][1]:,.0f} kW",
                f"{efficiency*100:.0f}%",
                f"{result['specs']['baseline_peak_kw']:.0f} kW",
                f"{result['specs']['new_peak_kw']:.0f} kW",
                f"{result['specs']['peak_reduction_kw']:.0f} kW ({(result['specs']['peak_reduction_kw']/result['specs']['baseline_peak_kw']*100):.1f}%)",
                f"{(result['annual']['opex']/opex_rate/result['battery_used'][0]):.1f}",
                f"{result['yearly_df'].iloc[-1]['capacity_remaining']:.1f}%"
            ]
        }
        st.dataframe(technical_summary, hide_index=True, use_container_width=True)
if __name__ == "__main__":
    main()