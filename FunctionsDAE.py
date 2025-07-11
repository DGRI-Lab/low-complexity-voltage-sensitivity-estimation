import numpy as np
from scipy.signal import find_peaks
import cmath


######################################## INITIAL R' , L' VALUES #####################################################

def Initial_R_1_L_1_Calc(V0, P0, Q0,frq):
    R_1_init= V0**2/(P0)
    L_1_init= V0**2/(Q0*2*np.pi*frq)

    return R_1_init, L_1_init


#################### FIND ZERO CROSSINGS ######################

def find_zero_crossings(signal):
 
    zero_crossings = []
    for n in range(1, len(signal)):
        if signal[n-1] * signal[n] < 0:  # Detect zero crossing (sign change)
            zero_crossings.append(n)  # Store the index of zero crossing. This is the index just after the zero crossing
    return zero_crossings


#################### FINIDNG POSITIVE PEAKS ##################

def find_positive_peaks(signal):
    positive_peaks = []
    for n in range(1, len(signal) - 1):
        # Check for a positive maximum peak
        if signal[n - 1] < signal[n] and  signal[n] > signal[n + 1] and signal[n] > 0:
            positive_peaks.append(n)  # Store the index of the positive peak (local maximum)
    return positive_peaks


################################# CALCULATING V2 IN TIME DOMAIN USING THE GIVEN TIME DOMAIN V1 AND I ###########################

def calculate_V2(V1, i, time_1, R,L):
    V2 = []
    V2.append(V1[0])
    for n in range(1,len(i)):  
        # Calculate V2 at time step n
        V2_tn = V1[n] - R * i[n] - L * (i[n] - i[n-1]) / (time_1[n]-time_1[n-1])
        V2.append(V2_tn) 
    return V2

################################ RMS CALCULATION ########################################

def calculate_rms_non_uniform(time_1, signal):
    # Ensure time and signal arrays are numpy arrays
    # time_1 = np.array(time_1)
    signal = np.array(signal)
    # print(f'in rms calc estimation time length:{len(time_1)}')  
    # print(f'in rms calc estimation signal length:{len(signal)}')
    
    # Calculate the squared signal values
    squared_signal = signal ** 2

    # Apply the trapezoidal rule for non-uniform time intervals
    integral_of_squared_signal = np.trapz(squared_signal, time_1)
    # integral_of_squared_signal = np.sum(squared_signal)   #This method gave wrong answers

    # Calculate the total time duration
    total_time = time_1[-1] - time_1[0]

    # Compute the RMS value
    rms_value = np.sqrt(integral_of_squared_signal / total_time)

    return rms_value

def calculate_rms_uniform(signal):
    
    # Compute the mean of the squared signal
    mean_square = np.mean(signal ** 2)
    # Return the RMS value
    return np.sqrt(mean_square)


###################################### ESTIMATING R AND L #######################################################

#Here the input v, and i should be the current and voltage upto now. Only the previous two zero crossings are not enough and for the digital RMS calculator, there should be atleast one cycle. 
def R_1_L_1_Estimation(V,i,time_1,Np,Nq, R,L, P0, Q0,freq,V0):
      
    # print (f'in RL estimation lenths V:{len(V)}, I:{len(i)}, t:{len(time_1)}')  
    V2_t_calc=calculate_V2(V,i,time_1, R,L)             #def calculate_V2(V1, i, time_1, R,L):
    V2_rms_calc=calculate_rms_non_uniform(time_1, V2_t_calc)            #def calculate_rms_non_uniform(time_1, signal):   
    # print(f'in RL estimation V2 signal length:{len(V2_t_calc)}')   
    R_1_next= (3**(1-Np/2))/P0*(V0**Np)*(V2_rms_calc**(2-Np))
    L_1_next= (3**(1-Nq/2))/(Q0*2*np.pi*np.average(freq))*(V0**Nq)*(V2_rms_calc**(2-Nq))
    
    return R_1_next,L_1_next

def R_1_L_1_Estimation_PMU(V2_PMU_Mag,Np,Nq, R,L, P0, Q0,freq,V0):
    
    V2_rms=V2_PMU_Mag   
    # print(f'in RL estimation V2 signal length:{len(V2_t_calc)}')   
    R_1_next= (3**(1-Np/2))/P0*(V0**Np)*(V2_rms**(2-Np))
    L_1_next= (3**(1-Nq/2))/(Q0*2*np.pi*np.average(freq))*(V0**Nq)*(V2_rms**(2-Nq))
    
    return R_1_next,L_1_next

####################### THETA CALCULATION ##################

def theta_calc(signal1, signal2, time_1, freq):

    signal1_peaks, _ = find_peaks(signal1)
    signal2_peaks, _ = find_peaks(signal2)
    # Get the time for the first peak in both signals
    if len(signal1_peaks) > 0:
        Signal1_peak_time = time_1[signal1_peaks[0]]  # First peak in V1 
    else:
        # Use the maximum value as the peak if no peak is found
        Signal1_peak_time = time_1[np.argmax(signal1)]
        
    if len(signal2_peaks) > 0:
        Signal2_peak_time = time_1[signal2_peaks[0]]  # First peak in V2
        
    else:
        # Use the maximum value as the peak if no peak is found
        Signal2_peak_time = time_1[np.argmax(signal2)]
        
    # Calculate the time shift (delta_t)
    delta_t = Signal2_peak_time - Signal1_peak_time
    # Calculate the angular frequency (omega)
    omega = 2 * np.pi * freq
    # Calculate the phase difference in radians
    theta = omega * delta_t * (-1)  # If delta_t is positive, theta should be negative (Lagging)

    return theta

 ############# ESTIMATING CURRENT SUING THE DIFFERENTIAL EQUATION #############

def Estimating_current_each_step(i_n_1, V_n, V_n_1, time_n, time_n_1,R_1,L_1, I_L_n_1,R,L):
    d_tn= time_n-time_n_1

    A=1+ R_1*d_tn/L*(1+R/R_1+L/L_1)+R*R_1*(d_tn**2)/(2*L*L_1)
    B=1+R_1*d_tn/L_1-R*R_1*(d_tn**2)/(2*L*L_1)
    C=d_tn/L+R_1*(d_tn**2)/(2*L*L_1)
    D= R_1*(d_tn**2)/(2*L*L_1)
    E=R_1*d_tn/L

    # print(f'A: {A}, B: {B}, C: {C}, D: {D}, E: {E}')
    # print(f'i_n_1: {i_n_1}, V_n: {V_n}, V_n_1: {V_n_1}, d_tn: {d_tn}, R: {R}, L: {L}, I_L_n_1: {I_L_n_1}, R_1: {R_1}, L_1: {L_1}')
    
    i_n=(i_n_1*B+V_n*C +V_n_1*D + I_L_n_1*E)/A
    # print(f'i_n: {i_n}')

    return i_n

################ ESTIMATING IL #########################################

#V1 should be given from a zero crossing
def Estimate_IL_init(V1, time_1,R,L, R_1, L_1, freq):
    
    V1_mag= calculate_rms_non_uniform(time_1,V1)             #def calculate_rms_non_uniform(time_1, signal):       
    V1_complex=V1_mag*(np.cos(0)+1j*np.sin(0))
    Z1= R+ 2*np.pi*freq*L*1j
    Z2_R=R_1
    Z2_L=1j*2*np.pi*freq*L_1
    Y2=1/Z2_R+1/Z2_L
    Z2=1/Y2
    I_complex=V1_complex/(Z1+Z2)
    IL_complex=I_complex*Z2_R/(Z2_R+Z2_L)
    IL_mag=np.abs(IL_complex)
    IL_angle=np.angle(IL_complex)

    return IL_mag,IL_angle

def Estimate_IL_init_PMU(V_Mag_PMU, V_Ang_PMU, R,L, R_1, L_1, freq):
    
    V1_complex=V_Mag_PMU*(np.cos(0)+1j*np.sin(0))
    Z1= R+ 2*np.pi*freq*L*1j
    Z2_R=R_1
    Z2_L=1j*2*np.pi*freq*L_1
    Y2=1/Z2_R+1/Z2_L
    Z2=1/Y2
    I_complex=V1_complex/(Z1+Z2)
    IL_complex=I_complex*Z2_R/(Z2_R+Z2_L)
    IL_mag=np.abs(IL_complex)
    IL_angle=np.angle(IL_complex)

    return IL_mag,IL_angle

def Estimate_IL(i_L_n_1, i_n, i_n_1, v_n, v_n_1, R, L, L_1, time_n, time_n_1):
    d_tn= time_n-time_n_1

    A= -R*d_tn/(2*L_1)-L/L_1
    B= -R*d_tn/(2*L_1)+L/L_1
    C= d_tn/(2*L_1)

    i_L_n= i_L_n_1 + A*i_n + B*i_n_1 + C*(v_n+v_n_1)

    return i_L_n

    
###################### ESTIMATONG CURRENT IN PHASOR DOMAIN ##################

def I_estimation_phasor(V1_rms, R, L, R_1_array, L_1_array, freq):
    R_1=np.average(R_1_array)
    L_1=np.average(L_1_array)
    # C_1=np.average(C_1_array)
    V1_complex=V1_rms*(np.cos(0)+1j*np.sin(0))
    Z1= R+ 2*np.pi*freq*L*1j
    Z2_R=R_1
    Z2_L=1j*2*np.pi*freq*L_1
    # Z2_C=1/(1j*2*np.pi*freq*C_1)
    Y2=1/Z2_R+1/Z2_L
    Z2=1/Y2
    I_complex=V1_complex/(Z1+Z2)
    I_mag= np.abs(I_complex)
    I_angle=np.angle(I_complex)

    return I_mag, I_angle

##################### CALCULATING AVERAGE PEAK ##############################

def Avg_Peak_Signal(positive_peaks,Signal_measured):
    Signal_peaks=np.array([])
    for x in positive_peaks:
        Signal_peaks=np.append(Signal_peaks,Signal_measured[x])
    Signal_peak_avg= np.average(Signal_peaks)

    return Signal_peak_avg


###################### MAIN CURRENT ESTIMATION FUNCTION ######################

# Here the I_L is estimated using V,R,L,R_1,L_1 at each zero crossing using last half cycle
def Current_Estimation_1(zero_crossings, V1,i, R, L, Np, Nq, V1_positive_peaks, P0, Q0, R_1_init, L_1_init, time_1, freq, V0):
    init_index=zero_crossings[0]
    start_index=zero_crossings[4]
    current=i[0:start_index+1]
    current_IL=np.array([])
    
    for index in range (start_index,len(V1)-1):
        # print(f'index:{index}')
        previous_zero_crossing = None
        previous_to_previous_zero_crossing = None

        for zc in zero_crossings:
            if zc < index:
                # Update previous_to_previous before updating previous_zero_crossing
                previous_to_previous_zero_crossing = previous_zero_crossing
                previous_zero_crossing = zc
            else:
                break 

        if index in zero_crossings:
                  
            if index==start_index:
                R_1 = R_1_init
                L_1 = L_1_init
                # current= np.append(current,i[start_index])
                
                I_L_mag,I_L_phase= Estimate_IL_init(V1[previous_zero_crossing:index+1], time_1[previous_zero_crossing:index+1], R, L, R_1, L_1, freq[index])
                I_L_est=I_L_mag* (2**0.5)* np.cos(2*np.pi*freq[index]*(time_1[index]-time_1[V1_positive_peaks[0]])+I_L_phase)   
                current_IL=np.append(current_IL,I_L_est)
            else:
                
                # print (f'length time:{len(time_1[previous_zero_crossing:index])}')
                # print (f'previous zero crossing:{previous_zero_crossing}')
                R_1,L_1=R_1_L_1_Estimation(V1[previous_zero_crossing:index+1], current[previous_zero_crossing:index+1], time_1[previous_zero_crossing:index+1], Np, Nq, R, L, P0, Q0, freq[previous_zero_crossing:index+1], V0)    #def R_1_L_1_Estimation(V,i,time_1,Np,Nq, R,L, P0, Q0,freq,V0):
                
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        #def Estimating_current_each_step(i_n_1, V_n, V_n_1, time_n, time_n_1,R_1,L_1, I_L_n_1,R,L):
        current_est=Estimating_current_each_step(current[index], V1[index+1], V1[index], time_1[index+1], time_1[index], R_1, L_1, current_IL[-1], R, L)   
        current= np.append(current,current_est)

        I_L_est= Estimate_IL(current_IL[-1], current[index+1], current[index], V1[index+1], V1[index], R, L, L_1, time_1[index+1], time_1[index])   #def Estimate_IL(i_L_n_1, i_n, i_n_1, v_n, v_n_1, R, L, L_1, time_n, time_n_1):
        current_IL=np.append(current_IL,I_L_est)

            
    return current


# Here the I_L is estimated using V,R,L,R_1,L_1 at each zero crossing using last cycle
def Current_Estimation_2(zero_crossings, V1,i, R, L, Np, Nq, V1_positive_peaks, P0, Q0, R_1_init, L_1_init, time_1, freq, V0):
    init_index=zero_crossings[0]
    start_index=zero_crossings[4]
    current=i[0:start_index+1]
    current_IL=np.array([])
    
    for index in range(start_index, len(V1) - 1):
    # Initialize variables to store past zero crossings
        previous_zero_crossing = None
        previous_to_previous_zero_crossing = None
        previous_to_previous_to_previous_zero_crossing = None
        previous_to_previous_to_previous_to_previous_zero_crossing = None

        for zc in zero_crossings:
            if zc < index:
                # Shift values before updating the latest
                previous_to_previous_to_previous_to_previous_zero_crossing = previous_to_previous_to_previous_zero_crossing
                previous_to_previous_to_previous_zero_crossing = previous_to_previous_zero_crossing
                previous_to_previous_zero_crossing = previous_zero_crossing
                previous_zero_crossing = zc

                x=previous_to_previous_zero_crossing
            else:
                break
    

        if index in zero_crossings:
                  
            if index==start_index:
                R_1 = R_1_init
                L_1 = L_1_init
                # current= np.append(current,i[start_index])
                
                I_L_mag,I_L_phase= Estimate_IL_init(V1[x:index+1], time_1[x:index+1], R, L, R_1, L_1, freq[index])
                I_L_est=I_L_mag* (2**0.5)* np.cos(2*np.pi*freq[index]*(time_1[index]-time_1[V1_positive_peaks[0]])+I_L_phase)   
                current_IL=np.append(current_IL,I_L_est)
            else:
                
                # print (f'length time:{len(time_1[previous_zero_crossing:index])}')
                # print (f'previous zero crossing:{previous_zero_crossing}')
                R_1,L_1=R_1_L_1_Estimation(V1[x:index+1], current[x:index+1], time_1[x:index+1], Np, Nq, R, L, P0, Q0, freq[x:index+1], V0)    #def R_1_L_1_Estimation(V,i,time_1,Np,Nq, R,L, P0, Q0,freq,V0):
                
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        #def Estimating_current_each_step(i_n_1, V_n, V_n_1, time_n, time_n_1,R_1,L_1, I_L_n_1,R,L):
        current_est=Estimating_current_each_step(current[index], V1[index+1], V1[index], time_1[index+1], time_1[index], R_1, L_1, current_IL[-1], R, L)   
        current= np.append(current,current_est)

        I_L_est= Estimate_IL(current_IL[-1], current[index+1], current[index], V1[index+1], V1[index], R, L, L_1, time_1[index+1], time_1[index])   #def Estimate_IL(i_L_n_1, i_n, i_n_1, v_n, v_n_1, R, L, L_1, time_n, time_n_1):
        current_IL=np.append(current_IL,I_L_est)

            
    return current


def Current_Estimation_3(zero_crossings, V1,i, R, L, Np, Nq, V1_positive_peaks, P0, Q0, R_1_init, L_1_init, time_1, freq, V0, V1_Mag_PMU, V1_Ang_PMU, V2_Mag_PMU):
    init_index=zero_crossings[0]
    start_index=zero_crossings[4]
    current=i[0:start_index+1]
    current_IL=np.array([])
    
    for index in range (start_index,len(V1)-1):
        # print(f'index:{index}')
        previous_zero_crossing = None
        previous_to_previous_zero_crossing = None

        for zc in zero_crossings:
            if zc < index:
                # Update previous_to_previous before updating previous_zero_crossing
                previous_to_previous_zero_crossing = previous_zero_crossing
                previous_zero_crossing = zc
            else:
                break 

        if index in zero_crossings:
                  
            if index==start_index:
                R_1 = R_1_init
                L_1 = L_1_init
                # current= np.append(current,i[start_index])
                
                #def Estimate_IL_init_PMU(V_Mag_PMU, V_Ang_PMU, R,L, R_1, L_1, freq):
                I_L_mag,I_L_phase= Estimate_IL_init_PMU(V1_Mag_PMU[index],V1_Ang_PMU[index], R, L, R_1, L_1, freq[index] )

                # I_L_mag,I_L_phase= Estimate_IL_init(V1[previous_to_previous_zero_crossing:index+1], time_1[previous_to_previous_zero_crossing:index+1], R, L, R_1, L_1, freq[index])
                I_L_est=I_L_mag* (2**0.5)* np.cos(2*np.pi*freq[index]*(time_1[index]-time_1[V1_positive_peaks[0]])+I_L_phase)   
                current_IL=np.append(current_IL,I_L_est)
            else:
                
                # print (f'length time:{len(time_1[previous_zero_crossing:index])}')
                # print (f'previous zero crossing:{previous_zero_crossing}')
                # R_1,L_1=R_1_L_1_Estimation(V1[previous_to_previous_zero_crossing:index+1], current[previous_to_previous_zero_crossing:index+1], time_1[previous_to_previous_zero_crossing:index+1], Np, Nq, R, L, P0, Q0, freq[previous_to_previous_zero_crossing:index+1], V0)    #def R_1_L_1_Estimation(V,i,time_1,Np,Nq, R,L, P0, Q0,freq,V0):
                ##def R_1_L_1_Estimation_PMU(V2_PMU_Mag,Np,Nq, R,L, P0, Q0,freq,V0):
                R_1,L_1=R_1_L_1_Estimation_PMU(V2_Mag_PMU[index], Np, Nq, R, L, P0, Q0, freq[previous_to_previous_zero_crossing:index+1], V0)    #def R_1_L_1_Estimation(V,i,time_1,Np,Nq, R,L, P0, Q0,freq,V0):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        #def Estimating_current_each_step(i_n_1, V_n, V_n_1, time_n, time_n_1,R_1,L_1, I_L_n_1,R,L):
        current_est=Estimating_current_each_step(current[index], V1[index+1], V1[index], time_1[index+1], time_1[index], R_1, L_1, current_IL[-1], R, L)   
        current= np.append(current,current_est)

        I_L_est= Estimate_IL(current_IL[-1], current[index+1], current[index], V1[index+1], V1[index], R, L, L_1, time_1[index+1], time_1[index])   #def Estimate_IL(i_L_n_1, i_n, i_n_1, v_n, v_n_1, R, L, L_1, time_n, time_n_1):
        current_IL=np.append(current_IL,I_L_est)

            
    return current


###################### RMSPE #####################################

def calculate_rmspe(y_true, y_pred):
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmspe = np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))
    
    return rmspe