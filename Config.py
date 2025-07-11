############### COMMON CONSTANTS ############################

voltage_scale=1000
current_scale=1000

frequency=60
f_nominal=60 # Frequency of the system
# V0_sys= 12.47*1000   # V
V0_sys= 25*1000   # V


########### CONSTANAT ACCORDING TO THE SCENARIO #############

# P0_calc= 6 *1000000   # W
# Q0_calc= 3 *1000000   #Var
# R_calc = 0.625   # Ohm
# L_calc = 0.0020723    # H
# Np_calc = 1.8
# Nq_calc = 1.5


############## OPTUNA PARAMETER MARGINS #####################

#Optuna Ranges (Stiff V source)
# R_low=0.1
# R_up=1
# L_low=0.001
# L_up=0.01
# Np_low=0
# Np_up=2.0
# Nq_low=0
# Nq_up=2.0
# P0_low=P0_calc*0.95
# P0_up=P0_calc*1.05
# Q0_low=Q0_calc*0.95
# Q0_up=Q0_calc*1.05


R_low=0
R_up=1
L_low=0
L_up=1
Np_low=0
Np_up=2.0
Nq_low=0
Nq_up=2.0



################# DATA EXTRACTION FROM FILE ##############

time_column= 'Time'

# Define mappings for voltage and current columns
voltage_columns = {
    'V1_phase1': 'InstVDGBus_MV_phase1',
    'V1_phase2': 'InstVDGBus_MV_phase2',
    'V1_phase3': 'InstVDGBus_MV_phase3'
    # 'V2_phase1': 'InstVDGBus_TX_phase1',
    # 'V2_phase2': 'InstVDGBus_TX_phase2',
    # 'V2_phase3': 'InstVDGBus_TX_phase3'
    # 'V1_phase1_mag': 'VA_mag_Fund',
    # 'V1_phase1_ang': 'VA_ang_Fund',
    # 'V1_phase1_rms' :'V1-phase_A_RMS'

}

current_columns = {
    'I_phase1': 'InstIDGBus_MV_phase1',
    'I_phase2': 'InstIDGBus_MV_phase2',
    'I_phase3': 'InstIDGBus_MV_phase3'
    # 'I_phase1_mag': 'IA_mag_Fund',
    # 'I_phase1_ang': 'IA_ang_Fund',
    # 'IL_phase1': 'LInstILoad_phase1',
    # 'IR_phase1': 'RInstILoad_phase1',
    # 'I_phase1_2': 'InstIDGBus_TX_phase1',
    # 'I_phase1_3': 'L1InstILoad_phase1'
}

# Internal_parameters={
#     'R_measured':'R',
#     'L_measured':'L',
#     'f_measured':'Frequency'
# }


columns_to_convert = [
    'InstIDGBus_MV_phase1', 'InstIDGBus_MV_phase2', 'InstIDGBus_MV_phase3',
    'InstVDGBus_MV_phase1', 'InstVDGBus_MV_phase2', 'InstVDGBus_MV_phase3'
    # 'InstIDGBus_TX_phase1', 'InstIDGBus_TX_phase2', 'InstIDGBus_TX_phase3',
    # 'InstVDGBus_TX_phase1', 'InstVDGBus_TX_phase2', 'InstVDGBus_TX_phase3'
    # 'L1InstVLoad_phase1', 'L1InstVLoad_phase2', 'L1InstVLoad_phase3',
    # 'L1InstILoad_phase1', 'L1InstILoad_phase2', 'L1InstILoad_phase3',
    # 'L1PLoad', 'L1QLoad', 
    # 'I-phase_A_RMS', 'I-phase_B_RMS', 'I-phase_C_RMS',
    # 'V1-phase_A_RMS', 'V1-phase_B_RMS', 'V1-phase_C_RMS',
    # 'V2-phase_A_RMS', 'V2-phase_B_RMS', 'V2-phase_C_RMS',
    # 'VA_mag_Fund','VB_mag_Fund', 'VC_mag_Fund',
    # 'VA_ang_Fund', 'VB_ang_Fund', 'VC_ang_Fund',
    # 'IA_mag_Fund','IB_mag_Fund','IC_mag_Fund',
    # 'IA_ang_Fund','IB_ang_Fund','IC_ang_Fund',
    #'LInstILoad_phase1','RInstILoad_phase1'

]

from pathlib import Path
# Define the common base directory once
base_dir = Path(r"C:\Punsara\University of Calgary\Research-masters\From 14022025\Data generated\Final Data used in Thesis\StiffV_ExpL\New\1. Np1.7Nq1.2_Dr_Dr")

# Build full file paths by joining the base directory with filenames
file_path_Waveform    = base_dir / "Data.parquet"
file_path_PMU_Voltage = base_dir / "Voltage_PMU.mat"
file_path_PMU_Current = base_dir / "Current_PMU.mat"

# output_directory = r'C:\Punsara\University of Calgary\Research-masters\From 14022025\Graphs\15062025\Results'


############### Timings for each event

####### StiffV_ExpL_ML ############

# Event 1
start_time = 9.94
end_time = 10.13

# # Event 2
# start_time = 
# end_time = 







# #Time for Stiff Voltage Sourse #########
# # # Disturbance
# start_time = 24.94
# end_time = 25.15
# #3 Steady State
# start_time = 24
# end_time = 24.5

## Time of 3DG_ Scenario 1 ##########
# First event
# start_time = 19.94
# end_time = 21

# # second event
# start_time = 69.94
# end_time = 71

## Time of 3DG_ Scenario 2/3/4  ##########
# First event
# start_time = 9.94
# end_time = 10.5

# # # second event
# start_time = 49
# end_time = 52

## Time of Stiff V with ExpL and ML Scenarios ##########
# First event
# start_time = 9.94
# end_time = 10.1


# # # second event
# start_time = 
# end_time =



## Time for Stiff V with CIGRE

# DR-Inc
# # First event
# start_time = 9.94
# end_time = 10.1

# # # second event
# start_time = 
# end_time =

# Inc-Dr
# First event
# start_time = 9.94
# end_time = 10.09

# # # second event
# start_time = 
# end_time =


# # RMSE (DAE)
# R_opt=809.221224774777
# L_opt= 0.381535818269796
# Np_opt= 0.0830310609430451
# Nq_opt= 0.442431735286258
# P0_opt= 6173437.41520654
# Q0_opt= 3041766.19506101

# #RMSE (Phasor)
# R_opt_P= 0.2574347562946
# L_opt_P= 0.00102853822929884
# Np_opt_P= 1.30156344572524
# Nq_opt_P=1.24573798923296
# P0_opt_P=4019.08787853265
# Q0_opt_P=31964.0866947775



### NEw Optuna Values

### 2kV- 0.8,0.5 

# # DAE
# R_opt= 0.100382504113414
# L_opt= 0.00218254454832458
# Np_opt= 0.934677697410926
# Nq_opt=0.738168634823773
# P0_opt=6003717.07128241
# Q0_opt= 3006168.87973551

# Phasor
# R_opt_P=0.100238434813329
# L_opt_P= 0.0010025103905998
# Np_opt_P= 0.929599694998821
# Nq_opt_P=0.695020723358229
# P0_opt_P=6011024.87159207
# Q0_opt_P= 3028995.66540398

# ## 2kV-1.1, 1.7
# # DAE
# R_opt= 0.3398281128505406
# L_opt= 0.005534266364956286
# Np_opt= 1.2452145620976822
# Nq_opt=1.8249750840982752
# P0_opt=6194635.96563428
# Q0_opt= 3004106.953487727

# # Phasor
# R_opt_P= 0.7533249533826014
# L_opt_P= 0.005056804249662329
# Np_opt_P= 1.2215431822827343
# Nq_opt_P=1.6599885740155125
# P0_opt_P=6053580.1454757
# Q0_opt_P=2943294.2149025006  

# # 2kV- 1.8 ,1.5

# # DAE
# R_opt= 0.139318974
# L_opt= 0.001307109
# Np_opt= 1.81807463
# Nq_opt=1.581686332
# P0_opt=5964757.196
# Q0_opt=2997498.877

# Phasor
# R_opt_P= 0.125799347395072
# L_opt_P= 0.00157748893223485
# Np_opt_P=1.81168968511072
# Nq_opt_P=1.58487223358204
# P0_opt_P=5971103.45232578
# Q0_opt_P= 2992312.75243413

## 2kV- 2,2

# DAE
# R_opt= 0.5420111243676986
# L_opt= 0.004507186690071362
# Np_opt= 1.9689684317513212
# Nq_opt=1.9216579836328902
# P0_opt=6144521.727067255
# Q0_opt= 3005525.6783137014

# Phasor
# R_opt_P= 0.6033628564018847
# L_opt_P= 0.005944808515744126
# Np_opt_P= 1.9979943693107138
# Nq_opt_P=1.9443538865421341
# P0_opt_P=6078048.33464585
# Q0_opt_P= 2924992.4865507614
