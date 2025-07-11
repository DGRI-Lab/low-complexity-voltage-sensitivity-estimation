import Config as config
import pandas as pd
import os
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import scipy.io
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import optuna

################## READ FREQUENCY FROM WAVEFORM DATA ####################

data = pd.read_parquet(config.file_path_Waveform)

data[config.columns_to_convert] = data[config.columns_to_convert].apply(pd.to_numeric, errors='coerce')

filtered_data = data[(data[config.time_column] >= config.start_time) & (data[config.time_column] <= config.end_time)]

# Extract time
time = filtered_data[config.time_column].to_numpy()  # Time in seconds

# Directory to save the graphs (if needed)
os.makedirs(config.output_directory, exist_ok=True)


############## READ PMU DATA ##############


### READING THE WHOLE SNIPPET

voltage_mat = scipy.io.loadmat(config.file_path_PMU_Voltage)
current_mat = scipy.io.loadmat(config.file_path_PMU_Current)

# VOLTAGE
PS_Mag_voltage = np.squeeze(voltage_mat['PS_Mag'])         
PS_Ang_voltage = np.squeeze(voltage_mat['PS_Ang'])         
time_vector_voltage = np.squeeze(voltage_mat['time_vector'])  

PS_Ang_D_voltage = np.squeeze(voltage_mat['PS_Ang_D'])     
PS_Mag_D_voltage = np.squeeze(voltage_mat['PS_Mag_D'])     
time_downsampled_voltage = np.squeeze(voltage_mat['time_downsampled']) 

# CURRENT
PS_Mag_current = np.squeeze(current_mat['PS_Mag'])         # Shape: (76801,)
PS_Ang_current = np.squeeze(current_mat['PS_Ang'])         # Shape: (76801,)
time_vector_current = np.squeeze(current_mat['time_vector'])  # Shape: (76801,)

PS_Ang_D_current = np.squeeze(current_mat['PS_Ang_D'])     # Shape: (2401,)
PS_Mag_D_current = np.squeeze(current_mat['PS_Mag_D'])     # Shape: (2401,)
time_downsampled_current = np.squeeze(current_mat['time_downsampled'])  # Shape: (2401,)

# FREQUENCY
Freq_Dev=np.squeeze(voltage_mat['Freq_Dev'])
Freq_Actual =   60+Freq_Dev
Freq_Dev_time= np.squeeze(voltage_mat['Freq_Time'])

# Downsampling the Frequency

downsampled_Freq_Actual = np.zeros_like(time_vector_voltage)

# Find the corresponding frequency value for each voltage time step
freq_index = 0  # Start from the beginning of Freq_Dev_time
for i, t in enumerate(time_downsampled_voltage):
    # Move forward in the frequency time array until the closest previous timestamp is found
    while freq_index < len(Freq_Dev_time) - 1 and Freq_Dev_time[freq_index + 1] <= t:
        freq_index += 1
    
    # Assign the last known frequency value
    downsampled_Freq_Actual[i] = Freq_Actual[freq_index]

### FILTERING TIME SELECTED ####################

config.start_time
config.end_time

# AFTRE DOWNSAMPLING 
valid_indices_voltage_D = np.where((time_downsampled_voltage >= config.start_time) & (time_downsampled_voltage <= config.end_time))[0]
valid_indices_current_D = np.where((time_downsampled_current >= config.start_time) & (time_downsampled_current <= config.end_time))[0]
valid_indices_Freq = np.where((Freq_Dev_time >= config.start_time) & (Freq_Dev_time <= config.end_time))[0]

PS_Mag_D_voltage_filtered = PS_Mag_D_voltage[valid_indices_voltage_D]
PS_Ang_D_voltage_filtered = PS_Ang_D_voltage[valid_indices_voltage_D]
PS_Ang_D_voltage_filtered = np.deg2rad(PS_Ang_D_voltage_filtered)
time_downsampled_voltage_filtered = time_downsampled_voltage[valid_indices_voltage_D]

PS_Mag_D_current_filtered = PS_Mag_D_current[valid_indices_current_D]
PS_Ang_D_current_filtered = PS_Ang_D_current[valid_indices_current_D]
PS_Ang_D_current_filtered = np.deg2rad(PS_Ang_D_current_filtered)
time_downsampled_current_filtered = time_downsampled_current[valid_indices_current_D]

Freq_Actual_filtered= Freq_Actual[valid_indices_Freq]
time_freq_filtered = Freq_Dev_time[valid_indices_Freq]
downsampled_Freq_Actual_filtered=downsampled_Freq_Actual[valid_indices_voltage_D]


####################################### OPTUNA ESTIMATE BOUNDARIES ############################### 

R_low=config.R_low
R_up=config.R_up
L_low=config.L_low
L_up=config.L_up
Np_low=config.Np_low
Np_up=config.Np_up
Nq_low=config.Nq_low
Nq_up=config.Nq_up


############ CALCULATE POWER BOUNDARIES #################


# Convert angle to radians if needed (assuming they're in radians)
V_pos = PS_Mag_D_voltage * np.exp(1j * PS_Ang_D_voltage)
I_pos = PS_Mag_D_current * np.exp(1j * PS_Ang_D_current)

# Complex power per phase
S_pos = V_pos * np.conj(I_pos)

# Total 3-phase power
S_total = 3 * S_pos  # Element-wise multiplication

# Active, reactive, and apparent power
P0_array = np.real(S_total)
Q0_array = np.imag(S_total)
S_array = np.abs(S_total)

# If you want average values over all samples:
P0_calc = np.abs(np.mean(P0_array))
Q0_calc = np.abs(np.mean(Q0_array))
S_calc = np.mean(S_array)

print(f"Active Power (P): {P0_calc:.2f} W")
print(f"Reactive Power (Q): {Q0_calc:.2f} VAr")
print(f"Apparent Power (S): {S_calc:.2f} VA")

# Set bounds

Q0_up = max(Q0_calc * 1.1, Q0_calc * 0.9)
Q0_low = min(Q0_calc * 1.1, Q0_calc * 0.9)

P0_up = max(P0_calc * 1.1, P0_calc * 0.9)
P0_low = min(P0_calc * 1.1, P0_calc * 0.9)

########## OPTUNA ###############################
from FunctionsPhasors import Initial_R_1_L_1_Calc_Ph   
from FunctionsPhasors import Current_Estimation_1_Ph_Opt  



V0=config.V0_sys
f=config.f_nominal

def objective(trial):
    # Suggest values for R, L, Np, and Nq
    R = trial.suggest_float('R', R_low, R_up)  
    L = trial.suggest_float('L', L_low, L_up)  
    Np = trial.suggest_float('Np', Np_low, Np_up)  
    Nq = trial.suggest_float('Nq', Nq_low, Nq_up)  
    P0=trial.suggest_float('P0', P0_low, P0_up)
    Q0=trial.suggest_float('Q0', Q0_low, Q0_up)

    R_1_init, L_1_init= Initial_R_1_L_1_Calc_Ph(V0,P0,Q0,f)

    I_estimate_mag_1, I_estimated_ang_1, R_1_calc_1, L_1_calc_1 =Current_Estimation_1_Ph_Opt(PS_Mag_D_voltage_filtered, PS_Ang_D_voltage_filtered,
                                                                                    PS_Mag_D_current_filtered, PS_Ang_D_current_filtered, 
                                                                                    R, L, 
                                                                                    Np, Nq,
                                                                                    P0, Q0, 
                                                                                    V0, downsampled_Freq_Actual_filtered,
                                                                                    R_1_init, L_1_init, trial)

                                        
                                        

    I_est_complex = I_estimate_mag_1 * np.exp(1j * I_estimated_ang_1)
    I_actual_complex = PS_Mag_D_current_filtered * np.exp(1j * PS_Ang_D_current_filtered)

    # Check for NaNs in y_true or y_pred
    if np.isnan(I_actual_complex).any() or np.isnan(I_est_complex).any():
        # Skip the trial if NaNs are detected
        raise optuna.TrialPruned()
    
    RMSE_phasor = np.sqrt(mean_squared_error(np.real(I_actual_complex), np.real(I_est_complex)) + 
                          mean_squared_error(np.imag(I_actual_complex), np.imag(I_est_complex)))

    return RMSE_phasor


# Create and run the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15000)  # Adjust n_trials as needed for more precision

# Get the best parameters
best_params = study.best_params
# print("Best parameters found: ", best_params)
# print("Best RMSE: ", study.best_value)

trials_df = study.trials_dataframe()

# Keep only completed trials (ignore pruned or failed ones)
trials_df = trials_df[trials_df['state'] == 'COMPLETE']

# Sort trials by RMSE (objective value) in ascending order (best values first)
trials_df = trials_df.sort_values(by='value', ascending=True)

# Select the top 10 best trials
top_10_trials = trials_df.head(10)


# Define the file path
from Config import output_directory
file_path = os.path.join(output_directory, "20.top_10_best_trials_Phasor.csv")
# Ensure the directory exists before saving
os.makedirs(os.path.dirname(file_path), exist_ok=True)
# Save the file
top_10_trials.to_csv(file_path, index=False)

print("Top 10 best parameter sets:")
for i, trial in enumerate(top_10_trials.itertuples(), start=1):
    print(f"Rank {i}: RMSE = {trial.value}")
    params = study.trials[trial.Index].params  # Already in dictionary format
    for key, val in params.items():
        print(f"    {key}: {val}")
    print("-" * 50)

# Display DataFrame in a readable format
print(top_10_trials)