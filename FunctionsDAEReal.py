import numpy as np
from scipy.signal import find_peaks



######################################## INITIAL R' , L' VALUES #####################################################

def Initial_R_1_L_1_Calc(V0, P0, Q0,frq):
    R_1_init= V0**2/(P0)
    L_1_init= V0**2/(Q0*2*np.pi*frq)

    return R_1_init, L_1_init


#################### FIND ZERO CROSSINGS ######################

def find_zero_crossings(signal):
    zero_crossings = []
    for n in range(1, len(signal)):
        # Detect zero crossing (sign change)
        if signal[n-1] * signal[n] < 0:  
            zero_crossings.append(n)  # Store the index just after the zero crossing
        # If the signal is exactly zero, consider it a zero crossing
        elif signal[n] == 0 and signal[n-1] != 0:  
            zero_crossings.append(n)  # Store the index where it becomes zero
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

def calculate_V2_2(V1, i, time_1, R,L):
    V2 = []
    V2.append(V1[0])
    for n in range(1,len(i)):  
        dt=(time_1[n]-time_1[n-1])
        A= dt/(2*L)
        B=R*dt/(2*L)
        # Calculate V2 at time step n
        V2_tn = (i[n]*(-B-1) + i[n-1]*(-B+1) +(V1[n]+V1[n-1])*(A) + V2[n-1]*(-A))/A
        V2.append(V2_tn) 
    return V2

################################ RMS CALCULATION ########################################

def calculate_rms_non_uniform(time_1, signal):
    # Ensure time and signal arrays are numpy arrays
    time_1 = np.array(time_1)
    signal = np.array(signal)
    
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
      
    V2_t_calc=calculate_V2(V,i,time_1, R,L)             #def calculate_V2(V1, i, time_1, R,L):
    V2_rms_calc=calculate_rms_non_uniform(time_1, V2_t_calc)            #def calculate_rms_non_uniform(time_1, signal):   
         
    R_1_next= (3**(1-Np/2))/P0*(V0**Np)*(V2_rms_calc**(2-Np))
    L_1_next= (3**(1-Nq/2))/(Q0*2*np.pi*freq)*(V0**Nq)*(V2_rms_calc**(2-Nq))
    
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

    A=( 1+ R_1*d_tn/L*(1+R/R_1+L/L_1)+R*R_1*(d_tn**2)/(2*L*L_1) )
    B=(1+R_1*d_tn/L_1-R*R_1*(d_tn**2)/(2*L*L_1) )
    C=(d_tn/L+R_1*(d_tn**2)/(2*L*L_1) )
    D= (R_1*(d_tn**2)/(2*L*L_1) )
    E=(R_1*d_tn/L )

    # A= (1 + R/R_1 + L/(d_tn*R_1) + d_tn*R/(2*L_1) + L/L_1 ) 
    # B= (L/(R_1*d_tn) - d_tn*R/(2*L_1) + L/L_1) 
    # C=(1/R_1 + d_tn/(2*L_1) ) 
    # D=( d_tn/(2*L_1) )
    # E= 1 


    
    i_n=(i_n_1*B+V_n*C +V_n_1*D + I_L_n_1*E)/A
    
    return i_n

################ ESTIMATING IL #########################################


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

def Estimate_IL(i_L_n_1, i_n, i_n_1, v_n, v_n_1, R, L, L_1, time_n, time_n_1):
    d_tn= time_n-time_n_1

    A= -R*d_tn/(2*L_1)-L/L_1
    B= -R*d_tn/(2*L_1)+L/L_1
    C= d_tn/(2*L_1)

    i_L_n= i_L_n_1 + A*i_n + B*i_n_1 + C*(v_n+v_n_1)

    return i_L_n



###################### MAIN CURRENT ESTIMATION FUNCTION ######################

# # Here the I_L is estimated using V,R,L,R_1,L_1 at each zero crossing
# def Current_Estimation_1(zero_crossings, V1,i, R, L, Np, Nq, V1_positive_peaks, P0, Q0, R_1_init, L_1_init, time_1, freq, V0):
#     init_index=zero_crossings[0]
#     start_index=zero_crossings[4]
#     current=i[0:start_index+1]
#     current_IL=np.array([])
    
#     for index in range (start_index,len(V1)-1):
#         # print(f'index:{index}')
#         previous_zero_crossing = None
#         previous_to_previous_zero_crossing = None

#         for zc in zero_crossings:
#             if zc < index:
#                 # Update previous_to_previous before updating previous_zero_crossing
#                 previous_to_previous_zero_crossing = previous_zero_crossing
#                 previous_zero_crossing = zc
#             else:
#                 break 

#         if index in zero_crossings:
                  
#             if index==start_index:
#                 R_1 = R_1_init
#                 L_1 = L_1_init
#                 # current= np.append(current,i[start_index])
                
#                 I_L_mag,I_L_phase= Estimate_IL_init(V1[previous_zero_crossing:index+1], time_1[previous_zero_crossing:index+1], R, L, R_1, L_1, freq)
#                 I_L_est=I_L_mag* (2**0.5)* np.cos(2*np.pi*freq*(time_1[index]-time_1[V1_positive_peaks[0]])+I_L_phase)   
#                 current_IL=np.append(current_IL,I_L_est)
#             else:
                
#                 # print (f'length time:{len(time_1[previous_zero_crossing:index])}')
#                 # print (f'previous zero crossing:{previous_zero_crossing}')
#                 R_1,L_1=R_1_L_1_Estimation(V1[previous_zero_crossing:index+1], current[previous_zero_crossing:index+1], time_1[previous_zero_crossing:index+1], Np, Nq, R, L, P0, Q0, freq, V0)    #def R_1_L_1_Estimation(V,i,time_1,Np,Nq, R,L, P0, Q0,freq,V0):
                
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
#         #def Estimating_current_each_step(i_n_1, V_n, V_n_1, time_n, time_n_1,R_1,L_1, I_L_n_1,R,L):
#         current_est=Estimating_current_each_step(current[index], V1[index+1], V1[index], time_1[index+1], time_1[index], R_1, L_1, current_IL[-1], R, L)   
#         current= np.append(current,current_est)

#         I_L_est= Estimate_IL(current_IL[-1], current[index+1], current[index], V1[index+1], V1[index], R, L, L_1, time_1[index+1], time_1[index])   #def Estimate_IL(i_L_n_1, i_n, i_n_1, v_n, v_n_1, R, L, L_1, time_n, time_n_1):
#         current_IL=np.append(current_IL,I_L_est)

            
#     return current

################################### DYNAMIC FREQUENCY ##############################





def find_zero_crossings_inter(waveform, time_1, voltage_zero_crossing_index):
    """
    Find the last three zero crossings in the current array up to the given voltage zero crossing index.
    Use linear interpolation for accurate zero crossing detection.
    """
    # Slice the current and time arrays up to the voltage zero crossing index
    waveform_slice = waveform[:voltage_zero_crossing_index + 1]
    time_slice = time_1[:voltage_zero_crossing_index + 1]

    # Initialize zero crossings
    zero_crossings = []

    # Iterate backward to find zero crossings
    for i in range(len(waveform_slice) - 1, 0, -1):  # Iterate backward
        if waveform_slice[i] * waveform_slice[i - 1] < 0:  # Zero crossing detected
            # Linearly interpolate to find the exact zero crossing time
            t1, t2 = time_slice[i - 1], time_slice[i]
            i1, i2 = waveform_slice[i - 1], waveform_slice[i]
            zero_crossing_time = t1 - (i1 * (t2 - t1)) / (i2 - i1)
            zero_crossings.append(zero_crossing_time)
        
        # If the signal is exactly zero, store its time
        elif waveform_slice[i] == 0 and waveform_slice[i - 1] != 0:
            zero_crossings.append(time_slice[i])
        
        # Stop if three zero crossings are found
        if len(zero_crossings) == 3:
            break

    # If fewer than two zero crossings are found, return None
    if len(zero_crossings) < 2:
        return None

    return zero_crossings[0], zero_crossings[2]



# Function to calculate frequency based on current zero crossings
def calculate_frequency_from_current(current, time_1, voltage_zero_crossing_index):
    """
    Calculate the frequency using the current zero crossings around a voltage zero crossing.
    """
    zero_crossings = find_zero_crossings_inter(current, time_1, voltage_zero_crossing_index)
    if zero_crossings is None:
        # print (f'Retured 60Hz in index:{voltage_zero_crossing_index}')
        return 60  # Default frequency if zero crossings are not found
    previous_zero_crossing_time, second_previous_zero_crossing_time = zero_crossings
    

    # Calculate the frequency (1 / period)
    period = previous_zero_crossing_time - second_previous_zero_crossing_time
    frequency = 1 /( period) if period > 0 else 0  # Avoid division by zero
    return frequency

def calculate_frequency_from_voltage(voltage, time_1, voltage_zero_crossing_index):
    """
    Calculate the frequency using the current zero crossings around a voltage zero crossing.
    """
    zero_crossings = find_zero_crossings_inter(voltage, time_1, voltage_zero_crossing_index)
    if zero_crossings is None:
        print (f'Retured 60Hz in index:{voltage_zero_crossing_index}')
        return 60  # Default frequency if zero crossings are not found
    previous_zero_crossing_time, second_previous_zero_crossing_time = zero_crossings
    

    # Calculate the frequency (1 / period)
    period = previous_zero_crossing_time - second_previous_zero_crossing_time
    frequency = 1 /( period) if period > 0 else 0  # Avoid division by zero
    return frequency

# Updated Current_Estimation_1 function
def Current_Estimation_1(zero_crossings, V1, i, R, L, Np, Nq, V1_positive_peaks, P0, Q0, R_1_init, L_1_init, time_1, V0):
    init_index = zero_crossings[0]
    start_index = zero_crossings[4]
    current = i[0:start_index + 1]
    current_IL = np.array([])
    freq = 60
    print (f'Start index:{start_index}')
    for index in range(start_index, len(V1) - 1):
        previous_zero_crossing = None
        previous_to_previous_zero_crossing = None

        for zc in zero_crossings:
            if zc < index:
                # Shift values before updating the latest
                
                previous_to_previous_zero_crossing = previous_zero_crossing
                previous_zero_crossing = zc

                x = previous_to_previous_zero_crossing 

                print 


                
            else:
                break
        
        # Find the frequency using current zero crossings around the current voltage zero crossing
        if index in zero_crossings:
            
            # if index in [zero_crossings[4], zero_crossings[5], zero_crossings[6], zero_crossings[7]]:
            #     # freq = calculate_frequency_from_current(current, time_1, index)
            #     freq = calculate_frequency_from_voltage(V1, time_1, index)
            #     # freq =60

            # else:
            #     # freq = calculate_frequency_from_current(current, time_1, index)
            #     freq = calculate_frequency_from_voltage(V1, time_1, index)
            #     # freq =60
            freq = calculate_frequency_from_voltage(V1, time_1, index)
            # freq = calculate_frequency_from_current(current, time_1, index)
            
            
        
            if index == start_index:
                # R_1 = R_1_init
                # L_1 = L_1_init
                R_1, L_1= Initial_R_1_L_1_Calc(V0,P0,Q0,freq)


                # Estimate initial I_L
                I_L_mag, I_L_phase = Estimate_IL_init(
                    V1[zero_crossings[2]:index + 1], 
                    time_1[zero_crossings[2]:index + 1], 
                    R, L, R_1, L_1, freq
                )
                I_L_est = I_L_mag * (2 ** 0.5) * np.cos(
                    2 * np.pi * freq * (time_1[index] - time_1[V1_positive_peaks[0]]) + I_L_phase
                )
                current_IL = np.append(current_IL, I_L_est)
            else:
                R_1, L_1 = R_1_L_1_Estimation(
                    V1[x:index + 1],
                    current[x:index + 1],
                    time_1[x:index + 1],
                    Np, Nq, R, L, P0, Q0, freq, V0
                )

        # Estimate current for the next step
        current_est = Estimating_current_each_step(
            current[index], V1[index + 1], V1[index], time_1[index + 1], time_1[index], 
            R_1, L_1, current_IL[-1], R, L
        )
        current = np.append(current, current_est)

        # Estimate I_L
        I_L_est = Estimate_IL(
            current_IL[-1], current[index + 1], current[index], V1[index + 1], V1[index], 
            R, L, L_1, time_1[index + 1], time_1[index]
        )
        current_IL = np.append(current_IL, I_L_est)

    return current

def sinusoid(t, A, theta):
    """ Sinusoidal function for curve fitting """
    return A * np.cos(2 * np.pi * 60 * t + theta)


# Knwong R_1 and L_1
def Current_Estimation_2(zero_crossings, V1, i, R, L, R_1, L_1, V1_positive_peaks, time_1):
    init_index = zero_crossings[0]
    start_index = zero_crossings[4]
    current = i[0:start_index + 1]
    current_IL = np.array([])
    freq = 60
    print (f'Start index:{start_index}')
    for index in range(start_index, len(V1) - 1):
        previous_zero_crossing = None
        previous_to_previous_zero_crossing = None

        for zc in zero_crossings:
            if zc < index:
                # Shift values before updating the latest
                
                previous_to_previous_zero_crossing = previous_zero_crossing
                previous_zero_crossing = zc

                x = previous_to_previous_zero_crossing 

                print 

                
            else:
                break
        
        # Find the frequency using current zero crossings around the current voltage zero crossing
        if index in zero_crossings:
            freq = calculate_frequency_from_voltage(V1, time_1, index)
            
            if index == start_index:
                
                # Estimate initial I_L
                I_L_mag, I_L_phase = Estimate_IL_init(
                    V1[zero_crossings[2]:index + 1], 
                    time_1[zero_crossings[2]:index + 1], 
                    R, L, R_1, L_1, freq
                )
                I_L_est = I_L_mag * (2 ** 0.5) * np.cos(
                    2 * np.pi * freq * (time_1[index] - time_1[V1_positive_peaks[0]]) + I_L_phase
                )
                current_IL = np.append(current_IL, I_L_est)
            
                

        # Estimate current for the next step
        current_est = Estimating_current_each_step(
            current[index], V1[index + 1], V1[index], time_1[index + 1], time_1[index], 
            R_1, L_1, current_IL[-1], R, L
        )
        current = np.append(current, current_est)

        # Estimate I_L
        I_L_est = Estimate_IL(
            current_IL[-1], current[index + 1], current[index], V1[index + 1], V1[index], 
            R, L, L_1, time_1[index + 1], time_1[index]
        )
        current_IL = np.append(current_IL, I_L_est)

    return current, current_IL

# Known R_1, L_1, IL[0]

def Current_Estimation_3(zero_crossings, V1, i, R, L, R_1, L_1, V1_positive_peaks, time_1, IL):
    init_index = zero_crossings[0]
    start_index = zero_crossings[4]
    current = i[0:start_index + 1]
    current_IL = IL[0:start_index + 1]
    freq = 60
    print (f'Start index:{start_index}')
    for index in range(start_index, len(V1) - 1):
        previous_zero_crossing = None
        previous_to_previous_zero_crossing = None

        for zc in zero_crossings:
            if zc < index:
                # Shift values before updating the latest
                
                previous_to_previous_zero_crossing = previous_zero_crossing
                previous_zero_crossing = zc

                x = previous_to_previous_zero_crossing 

                print 

                
            else:
                break
        
        # Find the frequency using current zero crossings around the current voltage zero crossing
        if index in zero_crossings:
            freq = calculate_frequency_from_voltage(V1, time_1, index)
            
            # if index == start_index:
                
            #     # Estimate initial I_L
                
            #     I_L_est = IL[index]
            #     current_IL = np.append(current_IL, I_L_est)
            
                

        # Estimate current for the next step
        current_est = Estimating_current_each_step(
            current[index], V1[index + 1], V1[index], time_1[index + 1], time_1[index], 
            R_1, L_1, current_IL[index], R, L
        )
        current = np.append(current, current_est)

        # Estimate I_L
        I_L_est = Estimate_IL(
            current_IL[-1], current[index + 1], current[index], V1[index + 1], V1[index], 
            R, L, L_1, time_1[index + 1], time_1[index]
        )
        current_IL = np.append(current_IL, I_L_est)

        

    return current, current_IL