###################### DIFFERENTIAL EQUATION METHOD ##########################################

######################## OPTUNA ESTIMATION #########################

import Config as config
import pandas as pd
import os
import plotly.graph_objs as go
from sklearn.metrics import root_mean_squared_error
import plotly.io as pio
import numpy as np
import optuna


# Load the Parquet file
data = pd.read_parquet(config.file_path_Waveform)

data[config.columns_to_convert] = data[config.columns_to_convert].apply(pd.to_numeric, errors='coerce')

filtered_data = data[(data[config.time_column] >= config.start_time) & (data[config.time_column] <= config.end_time)]

# Extract time
time = filtered_data[config.time_column].to_numpy()  # Time in seconds




V1_measured_phase1 = filtered_data[config.voltage_columns['V1_phase1']].to_numpy() * config.voltage_scale 
I_measured_phase1 = filtered_data[config.current_columns['I_phase1']].to_numpy() * config.current_scale 

V1_measured_phase2 = filtered_data[config.voltage_columns['V1_phase2']].to_numpy() * config.voltage_scale 
I_measured_phase2 = filtered_data[config.current_columns['I_phase2']].to_numpy() * config.current_scale 

V1_measured_phase3 = filtered_data[config.voltage_columns['V1_phase3']].to_numpy() * config.voltage_scale 
I_measured_phase3 = filtered_data[config.current_columns['I_phase3']].to_numpy() * config.current_scale 



V0=config.V0_sys
f=config.f_nominal

####################################### OPTUNA ESTIMATE BOUNDARIES ############################### 

R_low=config.R_low
R_up=config.R_up
L_low=config.L_low
L_up=config.L_up
Np_low=config.Np_low
Np_up=config.Np_up
Nq_low=config.Nq_low
Nq_up=config.Nq_up

######################### POWER BOUDARY CALCULATIONS ##################

# Calculate instantaneous power
instantaneous_power_phaseA = V1_measured_phase1 * I_measured_phase1
instantaneous_power_phaseB = V1_measured_phase2 * I_measured_phase2
instantaneous_power_phaseC = V1_measured_phase3 * I_measured_phase3

# Calculate Active Power (P): Average of instantaneous power
active_power = np.mean(instantaneous_power_phaseA)+np.mean(instantaneous_power_phaseB)+np.mean(instantaneous_power_phaseC)
# print(f"Active Power (P): {active_power} W")


from FunctionsDAEReal import calculate_rms_non_uniform 
voltage_rms_A = calculate_rms_non_uniform(time, V1_measured_phase1)
current_rms_A = calculate_rms_non_uniform(time,I_measured_phase1)

voltage_rms_B = calculate_rms_non_uniform(time, V1_measured_phase2)
current_rms_B = calculate_rms_non_uniform(time,I_measured_phase2)

voltage_rms_C = calculate_rms_non_uniform(time, V1_measured_phase3)
current_rms_C = calculate_rms_non_uniform(time,I_measured_phase3)


# Calculate Apparent Power (S) for each phase
apparent_power_A = voltage_rms_A * current_rms_A
apparent_power_B = voltage_rms_B * current_rms_B
apparent_power_C = voltage_rms_C * current_rms_C

apparent_power= apparent_power_A+apparent_power_B+apparent_power_C



reactive_power= np.sqrt(apparent_power**2-active_power**2)
print(f"Apparent Power (S): {apparent_power} VA")
print(f"Active Power (S): {active_power} W")
print(f"Reactive Power (Q): {reactive_power} VAr")

P0_calc=active_power
Q0_calc=reactive_power

P0_up=active_power*1.1
P0_low=active_power*0.9

Q0_up=reactive_power*1.1
Q0_low=reactive_power*0.9

print (f'Active power up:{P0_up}, Low:{P0_low}')
print (f'Reactive power up:{Q0_up}, Low:{Q0_low}')



################################### IMPORTING FUNCTIONS ####################################

from FunctionsDAEReal import find_zero_crossings           #def find_zero_crossings(signal):
from FunctionsDAEReal import find_positive_peaks           #def find_positive_peaks(signal):
from FunctionsDAEReal import Initial_R_1_L_1_Calc          #def Initial_R_1_L_1_Calc(V0, P0, Q0,freq):
from FunctionsDAEReal import Current_Estimation_1 
from FunctionsDAE import Current_Estimation_2          #def Current_Estimation_1(zero_crossings, V1,i, R, L, Np, Nq, V1_positive_peaks, P0, Q0, R_1_init, L_1_init, time_1, freq, V0):
from FunctionsDAE import calculate_rmspe               #def calculate_rmspe(y_true, y_pred):



######################################## INITIAL R' , L' VALUES #####################################################



V1_zero_crossings_phase1= find_zero_crossings(V1_measured_phase1)
V1_positive_peaks_phase1= find_positive_peaks(V1_measured_phase1)




def objective(trial):
    # Suggest values for R, L, Np, and Nq
    R = trial.suggest_float('R', R_low, R_up)  
    L = trial.suggest_float('L', L_low, L_up)  
    Np = trial.suggest_float('Np', Np_low, Np_up)  
    Nq = trial.suggest_float('Nq', Nq_low, Nq_up)  
    P0=trial.suggest_float('P0', P0_low, P0_up)
    Q0=trial.suggest_float('Q0', Q0_low, Q0_up)

    # R=config.R_calc
    # L=config.L_calc
    # P0=config.P0_calc
    # Q0=config.Q0_calc



    R_1_init, L_1_init= Initial_R_1_L_1_Calc(V0,P0,Q0,f)

    # I_estimate_phase1 =Current_Estimation_2(V1_zero_crossings_phase1,
    #                                     V1_measured_phase1, 
    #                                     I_measured_phase1, 
    #                                     R, L, 
    #                                     Np, Nq, 
    #                                     V1_positive_peaks_phase1, 
    #                                     P0, Q0, 
    #                                     R_1_init, L_1_init, 
    #                                     time, f_measured, V0)

    I_estimate_phase1 =Current_Estimation_1(V1_zero_crossings_phase1,
                                        V1_measured_phase1, 
                                        I_measured_phase1, 
                                        R, L, 
                                        Np, Nq, 
                                        V1_positive_peaks_phase1, 
                                        P0, Q0, 
                                        R_1_init, L_1_init, 
                                        time, V0)


    y_true = I_measured_phase1[(V1_zero_crossings_phase1[4]):]
    y_pred = I_estimate_phase1[(V1_zero_crossings_phase1[4]):]

    # Check for NaNs in y_true or y_pred
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        # Skip the trial if NaNs are detected
        raise optuna.TrialPruned()

    # Calculate the root mean square error
    RMSE_I = root_mean_squared_error(y_true, y_pred)

    return RMSE_I


# Create and run the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15000)  # Adjust n_trials as needed for more precision

# # Get the best parameters
best_params = study.best_params
# print("Best parameters found: ", best_params)
# print("Best RMSE: ", study.best_value)

# Convert trials to a DataFrame
trials_df = study.trials_dataframe()

# Keep only completed trials (ignore pruned or failed ones)
trials_df = trials_df[trials_df['state'] == 'COMPLETE']

# Sort trials by RMSE (objective value) in ascending order (best values first)
trials_df = trials_df.sort_values(by='value', ascending=True)

# Select the top 10 best trials
top_10_trials = trials_df.head(10)


# Define the file path
file_path = r"C:\Punsara\University of Calgary\Research-masters\From 14022025\Graphs\07032025\Results\17.top_10_best_trials_DAE.csv"
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