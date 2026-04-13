import pandas as pd
import numpy as np
import os

# 1. SERIAL CONFIGURATION
input_files = [
    'E:\B0__ageing\BO_CAL_0-10.csv', 
    'E:\B0__ageing\BO_CAL_RPT_11_20.csv', 
    'E:\B0__ageing\B0_CAL_RPT_20_30.csv', 
    'E:\B0__ageing\BO_CAL_RPT_30_40.csv', 
    'E:\B0__ageing\BO_CAL_RPT_40_50.csv', 
    'E:\B0__ageing\BO_CAL_RPT_50_70.csv', # Adjust filename if slightly different
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
    """Finds a column if ANY keyword is a substring of the column name."""
    for col in df_cols:
        for key in keywords:
            if key.lower() in col.lower():
                return col
    return None

print("Starting Fuzzy-Logic Serial Extraction...")

for file_name in input_files:
    if not os.path.exists(file_name):
        print(f"Skipping {file_name}: File not found.")
        continue
        
    print(f"Processing: {file_name}...")
    # Use low_memory=False to handle large mixed-type files
    df = pd.read_csv(file_name, low_memory=False)
    
    # 2. DYNAMIC FUZZY MAPPING
    # This maps your new 'Auxiliary temperature_1(℃)' and other variants
    mapping = {
        find_col_fuzzy(df.columns, ['Voltage(V)', 'Terminal_V']): 'terminal_voltage',
        find_col_fuzzy(df.columns, ['Current(A)', 'Terminal_I']): 'terminal_current',
        find_col_fuzzy(df.columns, ['temperature', 'T1', 'temp']): 'temp',
        find_col_fuzzy(df.columns, ['Capacity', 'Cap(Ah)']): 'capacity',
        find_col_fuzzy(df.columns, ['Time']): 'time',
        find_col_fuzzy(df.columns, ['V1(V)', 'V1']): 'charge_voltage'
    }
    
    # Remove any None mappings to prevent renaming errors
    mapping = {k: v for k, v in mapping.items() if k is not None}
    
    # 3. FILTER & PROCESS
    # Ensure 'Step Type' exists before filtering
    step_col = find_col_fuzzy(df.columns, ['Step Type'])
    if not step_col:
        print(f"Error: No 'Step Type' column in {file_name}. Skipping.")
        continue
        
    discharge_df = df[df[step_col] == 'CC DChg'].copy()
    
    # Detect cycle boundaries
    dp_col = find_col_fuzzy(df.columns, ['DataPoint'])
    discharge_df['new_cycle_flag'] = (discharge_df[dp_col].diff() > 1) | (discharge_df[dp_col].diff().isna())
    discharge_df['local_cycle'] = discharge_df['new_cycle_flag'].cumsum()
    
    processed_df = discharge_df.rename(columns=mapping).copy()
    
    # Handle absolute current for charge_current column
    if 'terminal_current' in processed_df.columns:
        processed_df['charge_current'] = processed_df['terminal_current'].abs()
    
    # Ensure all required columns exist; fill missing with 0
    for col in FINAL_COLS:
        if col not in processed_df.columns:
            processed_df[col] = 0.0

    # 4. SOH & SEQUENTIAL ID
    cycle_max_caps = processed_df.groupby('local_cycle')['capacity'].transform('max')
    processed_df['SoH'] = cycle_max_caps / RATED_CAPACITY
    
    for local_id in range(1, int(processed_df['local_cycle'].max()) + 1):
        global_cycle_count += 1
        cycle_data = processed_df[processed_df['local_cycle'] == local_id].copy()
        cycle_data['cycle'] = global_cycle_count
        all_processed_cycles.append(cycle_data[FINAL_COLS])

# 5. MERGE & SAVE
if all_processed_cycles:
    master_df = pd.concat(all_processed_cycles, ignore_index=True)
    master_df.to_csv('merged_70_cycles_battery_data.csv', index=False)
    print(f"\nSUCCESS: Created 'merged_70_cycles_battery_data.csv'")
    print(f"Total Cycles: {global_cycle_count}")
    print(f"Found Temperature Column as: {find_col_fuzzy(df.columns, ['temperature', 'T1', 'temp'])}")