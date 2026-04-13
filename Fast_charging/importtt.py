import pandas as pd
import numpy as np
import os

# 1. SERIAL CONFIGURATION
# Ordered based on experimental progression
input_files = [
    'B3_FAST_RPT_0.csv',
    'B3_FAST_1_10.csv',
    'B3_FAST_RPT_11-20.csv',
    'B3_FAST_RPT_21-35.csv',
    'B3_FAST_RPT_35_45.csv',
    'B3_FAST_RPT_46_53.xlsx',
    'B3_FAST_RPT_54_74.xlsx'
]

FINAL_COLS = [
    'terminal_voltage', 'terminal_current', 'temp', 
    'charge_current', 'charge_voltage', 'time', 
    'capacity', 'cycle', 'SoH'
]

RATED_CAPACITY = 2.0  
all_processed_cycles = []
global_cycle_count = 0

def find_col_fuzzy(df_cols, keywords):
    """Finds a column name even if software headers change (e.g. T1(?) vs Temperature)."""
    for col in df_cols:
        for key in keywords:
            if key.lower() in str(col).lower():
                return col
    return None

print("Starting Serial Processing for B3_FAST datasets...")

# 2. EXTRACTION LOOP
for file_name in input_files:
    if not os.path.exists(file_name):
        print(f"Skipping {file_name}: File not found.")
        continue
        
    print(f"Processing: {file_name}...")
    
    # Handle CSV or XLSX with specific sheet
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_name, low_memory=False)
    else:
        # Assuming openpyxl is installed for excel reading
        df = pd.read_excel(file_name, sheet_name='record')

    # Fuzzy Dynamic Mapping
    mapping = {
        find_col_fuzzy(df.columns, ['Voltage(V)', 'Voltage']): 'terminal_voltage',
        find_col_fuzzy(df.columns, ['Current(A)', 'Current']): 'terminal_current',
        find_col_fuzzy(df.columns, ['T1', 'temperature', 'temp']): 'temp',
        find_col_fuzzy(df.columns, ['Capacity', 'Cap(Ah)']): 'capacity',
        find_col_fuzzy(df.columns, ['Time']): 'time',
        find_col_fuzzy(df.columns, ['V1(V)', 'V1']): 'charge_voltage'
    }
    mapping = {k: v for k, v in mapping.items() if k is not None}
    
    # Filter for CC Discharge steps
    step_col = find_col_fuzzy(df.columns, ['Step Type', 'StepName'])
    if not step_col:
        print(f"Error: Step Type column not found in {file_name}")
        continue
    
    discharge_df = df[df[step_col].str.contains('DChg', na=False)].copy()
    
    # Identify Cycles
    dp_col = find_col_fuzzy(df.columns, ['DataPoint', 'Step Index'])
    discharge_df['new_cycle_flag'] = (discharge_df[dp_col].diff() != 1) 
    discharge_df['local_cycle'] = discharge_df['new_cycle_flag'].cumsum()
    
    processed_df = discharge_df.rename(columns=mapping).copy()
    
    # Fix the 'charge_current' issue by using magnitude of discharge current
    if 'terminal_current' in processed_df.columns:
        processed_df['charge_current'] = processed_df['terminal_current'].abs()
    
    # Process exactly 10 cycles per file (if available)
    local_cycle_ids = processed_df['local_cycle'].unique()[:10]
    
    for local_id in local_cycle_ids:
        global_cycle_count += 1
        cycle_data = processed_df[processed_df['local_cycle'] == local_id].copy()
        
        # Calculate SoH
        cycle_max_cap = cycle_data['capacity'].max()
        cycle_data['SoH'] = cycle_max_cap / RATED_CAPACITY
        cycle_data['cycle'] = global_cycle_count
        
        # Ensure all required columns are present
        for col in FINAL_COLS:
            if col not in cycle_data.columns:
                cycle_data[col] = 0.0
                
        all_processed_cycles.append(cycle_data[FINAL_COLS])

# 3. MERGE & SAVE
if all_processed_cycles:
    master_df = pd.concat(all_processed_cycles, ignore_index=True)
    master_df.to_csv('B3_FAST_merged_70_cycles.csv', index=False)
    print(f"\nSUCCESS: Generated 'B3_FAST_merged_70_cycles.csv' with {global_cycle_count} cycles.")
else:
    print("No data was processed. Check file names and Step Types.")