import numpy as np
import cmath
import optuna


######################################## INITIAL R' , L' VALUES #####################################################

def Initial_R_1_L_1_Calc_Ph(V0, P0, Q0,frq):
    R_1_init= V0**2/(P0)
    L_1_init= V0**2/(Q0*2*np.pi*frq)

    return R_1_init, L_1_init


###################################### ESTIMATING R AND L #######################################################

# when only single values used
def R_1_L_1_Estimation_1(V1_mag,V1_ang, I_mag, I_ang ,Np,Nq, R,L, P0, Q0,freq,V0):
    V1_complex= cmath.rect(V1_mag,V1_ang)
    I_complex= cmath.rect(I_mag,I_ang)

    # V1_complex= V1_mag*(np.cos(V1_ang)+1j*np.sin(V1_ang))
    # I_complex= I_mag*(np.cos(I_ang)+1j*np.sin(I_ang))

    V2_complex = V1_complex-I_complex* (R+1j* 2* np.pi* freq* L)

    V2_rms=np.abs(V2_complex)             
    
    
    R_1_next= (3**(1-Np/2))/P0*(V0**Np)*(V2_rms**(2-Np))
    L_1_next= (3**(1-Nq/2))/(Q0*2*np.pi*freq)*(V0**Nq)*(V2_rms**(2-Nq))
    # print(f'V1:{V1_complex}, V2:{V2_complex}, I:{I_complex}, R:{R_1_next}, L:{L_1_next}')
    return R_1_next,L_1_next


# last cycle used
def R_1_L_1_Estimation_2(V1_mag,V1_ang, I_mag, I_ang ,Np,Nq, R,L, P0, Q0,freq,V0):
    V1_complex= V1_mag*(np.cos(V1_ang)+1j*np.sin(V1_ang))
    I_complex= I_mag*(np.cos(I_ang)+1j*np.sin(I_ang))

    V2_complex = V1_complex-I_complex* (R+1j* 2* np.pi* freq* L)

    V2_rms=np.abs(V2_complex)         
    R_1_array= (3**(1-Np/2))/P0*(V0**Np)*(V2_rms**(2-Np))
    L_1_array= (3**(1-Nq/2))/(Q0*2*np.pi*freq)*(V0**Nq)*(V2_rms**(2-Nq))
    R_1_next= np.mean(R_1_array)
    L_1_next = np.mean(L_1_array)

    # V2_rms_array=np.abs(V2_complex)
    # V2_rms= np.mean(V2_rms_array)
    # f_avg=np.mean(freq)
    # R_1_next= (3**(1-Np/2))/P0*(V0**Np)*(V2_rms**(2-Np))
    # L_1_next= (3**(1-Nq/2))/(Q0*2*np.pi*f_avg)*(V0**Nq)*(V2_rms**(2-Nq))
  
    return R_1_next,L_1_next




############# ERROR CALCULATION ####################

def Error_Calculation(y_measured_mag, y_measured_ang, y_estimated_mag, y_estimated_ang, start_index):
    y_mag_true = y_measured_mag[start_index:]
    y_ang_true =y_measured_ang[start_index:]

    y_mag_predicted = y_estimated_mag[start_index:]
    y_ang_predicted =y_estimated_ang[start_index:]

    magnitude_error = np.sqrt(np.sum((y_mag_true - y_mag_predicted)**2))
    angle_error = np.sqrt(np.sum((y_ang_true - y_ang_predicted)**2))

    w_mag = 1  # Weight for magnitude
    w_ang = 1  # Weight for angle

    total_error = w_mag * magnitude_error + w_ang * angle_error

    return total_error, magnitude_error, angle_error


############# ZERO HOLD FOR WAVEFOMR RECONSTRUCTION #############


def zero_order_hold(time_vector, time_downsampled, mag_downsampled):
    
    # Initialize output arrays
    mag_ZOH = np.zeros_like(time_vector)  
    

    index = 0  # Start at the first PMU sample
    num_samples = len(time_downsampled)  # Number of downsampled points

    for i, t in enumerate(time_vector):
        # Move to the next PMU sample if within bounds
        if index < num_samples - 1 and t >= time_downsampled[index + 1]:
            index += 1  # Update index to the next PMU sample
        
        # Assign the current PMU sample to all points within this period
        mag_ZOH[i] = mag_downsampled[index]
        

    return mag_ZOH



###################### MAIN CURRENT ESTIMATION FUNCTION ######################

# R_1 AND L_1 IS CALCULATED USING THE VOLTAGE AND THE CURRENT AT THE PREVIOUS STEP. USED THE CALCULATED PREVIOUS STEP VALUE. 

########### Used for Optuna #################
def Current_Estimation_1_Ph_Opt(V1_mag,V1_ang, I_mag, I_ang, R, L, Np, Nq, P0, Q0, V0,freq, R_1_init, L_1_init, trial):

    I_mag_calc = np.zeros(len(V1_mag))
    I_ang_calc = np.zeros(len(V1_mag))
    R_1_Array = np.zeros(len(V1_mag))
    L_1_Array = np.zeros(len(V1_mag))

    I_mag_calc[0] = I_mag[0]
    I_ang_calc[0] = I_ang[0]
    R_1_Array[0] = R_1_init
    L_1_Array[0] = L_1_init 

    
    for index in range (0,len(V1_mag)-1):
        R_1= R_1_Array[index]
        L_1= L_1_Array[index]

        # V1_complex= V1_mag[index+1]*(np.cos(V1_ang[index+1])+1j*np.sin(V1_ang[index+1]))
        V1_complex = cmath.rect(V1_mag[index+1], V1_ang[index+1])

        Z1= R+ 1j* 2* np.pi* freq[index+1]* L
        Z2_R=R_1
        Z2_L=1j*2*np.pi*freq[index+1]*L_1
        # print (f'Index:{index}, Z2_R:{Z2_R}, Z2_L:{Z2_L} ')

        epsilon = 1e-8
        if abs(R_1) < epsilon or abs(L_1) < epsilon:
            print(f"Skipping trial: R_1 or L_1 too small at index {index} -> R_1={R_1}, L_1={L_1}")
            raise optuna.TrialPruned()
        
        Y2=1/Z2_R+1/Z2_L
        Z2=1/Y2 
        I_est_complex= V1_complex/(Z1+Z2) 
        I_mag_est=np.abs(I_est_complex)
        I_ang_est=np.angle(I_est_complex)  

        I_mag_calc[index+1] = I_mag_est
        I_ang_calc[index+1] = I_ang_est

        
        # print(f'V1:{V1_mag[index+1]} , V1_a:{V1_ang[index+1]} , I:{I_mag_calc[index+1]} , I_a:{I_ang_calc[index+1]}, np:{Np}, nq:{Nq}, R:{R}, L:{L}, P0:{P0}, Q0:{Q0}, Frq:{freq[index+1]}, V0:{V0} ')
        #def R_1_L_1_Estimation(V1_mag,V1_ang, I_mag, I_ang ,Np,Nq, R,L, P0, Q0,freq,V0):
        R_1,L_1=R_1_L_1_Estimation_1(V1_mag[index+1], V1_ang[index+1], I_mag_calc[index+1], I_ang_calc[index+1], Np, Nq, R, L, P0, Q0, freq[index+1], V0)

        print()
        
        R_1_Array[index+1] = R_1
        L_1_Array[index+1] = L_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
                   
    return I_mag_calc, I_ang_calc,R_1_Array, L_1_Array 



########### Used for Optuna #################
def Current_Estimation_1_Ph_CC(V1_mag,V1_ang, I_mag, I_ang, R, L, Np, Nq, P0, Q0, V0,freq, R_1_init, L_1_init):

    I_mag_calc = np.zeros(len(V1_mag))
    I_ang_calc = np.zeros(len(V1_mag))
    R_1_Array = np.zeros(len(V1_mag))
    L_1_Array = np.zeros(len(V1_mag))

    I_mag_calc[0] = I_mag[0]
    I_ang_calc[0] = I_ang[0]
    R_1_Array[0] = R_1_init
    L_1_Array[0] = L_1_init 

    
    for index in range (0,len(V1_mag)-1):
        R_1= R_1_Array[index]
        L_1= L_1_Array[index]

        # V1_complex= V1_mag[index+1]*(np.cos(V1_ang[index+1])+1j*np.sin(V1_ang[index+1]))
        V1_complex = cmath.rect(V1_mag[index+1], V1_ang[index+1])

        Z1= R+ 1j* 2* np.pi* freq[index+1]* L
        Z2_R=R_1
        Z2_L=1j*2*np.pi*freq[index+1]*L_1
        # print (f'Index:{index}, Z2_R:{Z2_R}, Z2_L:{Z2_L} ')

        
        Y2=1/Z2_R+1/Z2_L
        Z2=1/Y2 
        I_est_complex= V1_complex/(Z1+Z2) 
        I_mag_est=np.abs(I_est_complex)
        I_ang_est=np.angle(I_est_complex)  

        I_mag_calc[index+1] = I_mag_est
        I_ang_calc[index+1] = I_ang_est

        
        # print(f'V1:{V1_mag[index+1]} , V1_a:{V1_ang[index+1]} , I:{I_mag_calc[index+1]} , I_a:{I_ang_calc[index+1]}, np:{Np}, nq:{Nq}, R:{R}, L:{L}, P0:{P0}, Q0:{Q0}, Frq:{freq[index+1]}, V0:{V0} ')
        #def R_1_L_1_Estimation(V1_mag,V1_ang, I_mag, I_ang ,Np,Nq, R,L, P0, Q0,freq,V0):
        R_1,L_1=R_1_L_1_Estimation_1(V1_mag[index+1], V1_ang[index+1], I_mag_calc[index+1], I_ang_calc[index+1], Np, Nq, R, L, P0, Q0, freq[index+1], V0)

        print()
        
        R_1_Array[index+1] = R_1
        L_1_Array[index+1] = L_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
                   
    return I_mag_calc, I_ang_calc,R_1_Array, L_1_Array 





