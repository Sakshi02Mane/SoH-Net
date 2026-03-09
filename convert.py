import scipy.io as sio
import pandas as pd
import numpy as np

def process_battery(mat_file, battery_name):
    mat = sio.loadmat(mat_file)
    battery = mat[battery_name]
    cycles = battery['cycle'][0][0][0]
    
    all_data = []
    cycle_num = 0
    
    for cycle in cycles:
        cycle_type = str(cycle['type'][0])
        if cycle_type == 'discharge':
            cycle_num += 1
            data = cycle['data'][0][0]
            
            voltage   = data['Voltage_measured'][0].flatten()
            current   = data['Current_measured'][0].flatten()
            temp      = data['Temperature_measured'][0].flatten()
            curr_load = data['Current_load'][0].flatten()
            volt_load = data['Voltage_load'][0].flatten()
            time      = data['Time'][0].flatten()
            capacity  = float(data['Capacity'][0].flatten()[0])
            soh       = capacity / 2.0

            df = pd.DataFrame({
                'terminal_voltage': voltage,
                'terminal_current': current,
                'temperature':      temp,
                'charge_current':   curr_load,
                'charge_voltage':   volt_load,
                'time':             time,
                'capacity':         capacity,
                'cycle':            cycle_num,
                'SOH':              soh
            })
            all_data.append(df)
    
    final_df = pd.concat(all_data, ignore_index=True)
    output_path = f"Datasets/{battery_name}_discharge_soh.csv"
    final_df.to_csv(output_path, index=False)
    print(f"✅ Saved {output_path} — {cycle_num} cycles, {len(final_df)} rows")

process_battery("Datasets/B0005.mat", "B0005")
process_battery("Datasets/B0006.mat", "B0006")
process_battery("Datasets/B0007.mat", "B0007")
process_battery("Datasets/B0018.mat", "B0018")
