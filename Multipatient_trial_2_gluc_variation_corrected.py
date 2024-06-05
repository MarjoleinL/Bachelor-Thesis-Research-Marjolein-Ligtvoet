#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:01:03 2024

@author: marjoleinligtvoet
"""


# import cyipopt 
import numpy as np
import os
import glob
# import numdifftools.nd_statsmodels as nd   
import matplotlib.pyplot as plt 
import scipy
from scipy.optimize import minimize
import pandas as pd
from scipy.interpolate import interp1d
import random
import statistics
import csv
# from csv import writer
from itertools import count
import copy

def sorbent_chamber(c_ddr, t):
    #have to convert to mg/ml from mmol/L as thomas eq uses that unit
    c_us = copy.deepcopy(c_ddr)
    for s, solute in enumerate(solutes):
        c_us[s] *= (molecular_weight[solute]/1000)
    # thomas model
    c_ds = c_us*(1/(1+np.exp(k_Th/f_avg*(q_e*x_AC-c_us*f_avg*t)))).ravel() 
    # change back to mmol/L
    for s, solute in enumerate(solutes):
        c_ds[s] /= (molecular_weight[solute]/1000)
    return np.array(c_ds)

molecular_weight = {'Sodium': 22.9898, 
                    'Glucose': 180.156,
                    'Phosphate': 94.9714,
                    'Potassium': 39.0983,
                    'Creatinine':113.12,
                    'Urea': 60.06,
                    }

def DR(c_ds, c_ddr,t):
    # print(data_ds[substance], data_w[substance], f_avg)
    c_ddr[t] = ((VDR * c_ddr[t-1] + f_avg * c_ds)/(VDR+f_avg)).ravel()
    return c_ddr

def mixing(c_DR, c_per):
    
    c_DR = np.add(((V_fill-f_avg)*c_per).to_numpy(), f_avg * c_DR)/V_fill
    return c_DR

def objective(mtac, predicted_cd, Cp, L, V, Vr, V_fill, mode, transport_status_key, PS_s, PS_l):
    '''The objective function needed to be minimised'''
    
    print(VDR)
    t = 480 #min
    
    predicted_cd, DR_conc = rk(t, mtac, predicted_cd, Cp, L, V, Vr, V_fill, mode, transport_status_key, PS_s, PS_l)
    
    return predicted_cd, DR_conc


def rk(t, mtac, predicted_cd, Cp, L, V, Vr, V_fill, mode, transport_status_key, PS_s, PS_l): 
    c_ddr = np.zeros((t,6)) 
    c_ddr[0] = cd.loc[0]
    DR_conc = np.zeros((t,6)) 
    for timestep in range(1,t): 
        
        Cd = np.array(predicted_cd.loc[timestep-1])
        Cp = np.array(cp.loc[0])
        
        
        PS_s = mtac * 0.998 * abya0_s #fraction small pore surface area - Rippe, A THREE-PORE MODEL OF PERITONEAL TRANSPORT table 1
        # print(PS_s)
        #The current albumin value is from Joost.
        #MTAC for large pores
        PS_l = mtac * 0.002 *abya0_l# Ref: two pore model-oberg,rippe, table 1, A0L/A0 value
        #Defining transport status
        if transport_status_key== 'slow':
            PS_s = mtac * 0.998 * abya0_s*(15000/25000) 
            PS_l = mtac * 0.002 *abya0_l*(15000/25000)
        elif transport_status_key == 'average':
            PS_s = mtac * 0.998 * abya0_s
            PS_l = mtac * 0.002 *abya0_l
        elif transport_status_key == 'fast':
            PS_s = mtac * 0.998 * abya0_s*(45000/25000)
            PS_l = mtac * 0.002 *abya0_l*(45000/25000)
        
        
        "Apply Runge Kutta Formulas to find next value of y"
        k1 = comdxdt(Cd, timestep, mtac,  Cp, L, V,  Vr, V_fill, PS_s, PS_l)
        # print(cd, k1)
        k2 = comdxdt(Cd + 0.5  *k1, timestep, mtac,  Cp, L, V,  Vr, V_fill, PS_s, PS_l)
        k3 = comdxdt(Cd + 0.5  *k2, timestep, mtac,   Cp, L, V,  Vr, V_fill, PS_s, PS_l)
        k4 = comdxdt(Cd + k3, timestep, mtac,  Cp, L, V,  Vr, V_fill, PS_s, PS_l)
        
        # Update next value of y
        Cd = Cd + (1 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
        # This is the conc post transfer with the plasma. We assume all steps take one after other
        predicted_cd.loc[timestep] = Cd
        
        if 'static' not in mode:
            
            
                
            if 'daytime' not in mode: 
                # then gets diluted through the 10 L dialysate reservoir
                DR_conc = DR(Cd, c_ddr, timestep)
                Cd = DR(Cd, c_ddr, timestep)[timestep]
                
            if 'sorbent' in mode:
                # the new conc goes through the sorbent chamber
                Cd = sorbent_chamber(Cd, timestep)
            # The fluid mixes in the peritoneal cavity before getting exchanged again
            Cd = mixing(Cd, predicted_cd.loc[timestep])
        
        # print(Cd)
        #print(UF)
        predicted_cd.loc[timestep] = Cd
        
        
    return predicted_cd, DR_conc


# the differential equations
def comdxdt(Cd, t, x,  Cp, L, V,  Vr, V_fill, PS_s, PS_l):
    '''
    

    Parameters
    ----------
    Cd : predicted dialysate concentration
    t : timepoint
        DESCRIPTION.
    x : intial matrix
        x[0:6] = MTAC
        x[6:12] = fct
        x[12:18] = SiCo
        x[18] = QL
    model : 1-6
        DESCRIPTION.

    Returns
    -------
    dxdt : conc gradient
        derivative

    '''
    
    solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
    
    af = 16.18 * (1 - np.exp (-0.00077*V[t]))/13.3187
    
    delP = delP0 - ((V[t] - (V_fill+Vr))/490)
    # print(delP, af)
    #peritoneal concentration gradient
   
    
    pr = [phi[i] *RT * (Cp[i]-Cd[i]) for i in range(len(solutes))]
    sigmas_pr = sum([sigma_s[i]*pr[i] for i in range(len(solutes))])
    sigmal_pr = sum([sigma_l[i]*pr[i] for i in range(len(solutes))])
    # print(pr, sigmas_pr)

    # #print("pr", pr, sum(sigma_s*pr.ravel()))
    # #volumetric flows across the pores
    J_vC = af*alpha[0]*LS*(delP - sum(pr))/1000 #l/min
    J_vS = af*alpha[1]*LS*(delP  - sigmas_pr)/1000 #l/min
    J_vL = af*alpha[2]*LS*(delP - sigmal_pr)/1000 #l/min
    # print(J_vC, J_vS, J_vL)

    # #Peclet numbers
    Pe_s = np.array([J_vS  * (1 - sigma_s[i])/(af*PS_s[i]) for i in range(len(solutes))])
    Pe_l = np.array([J_vL  * (1 - sigma_l[i])/(af*PS_l[i]) for i in range(len(solutes))])
    # print(Pe_s)
    
    # #solute flow rate
    J_sS = np.array([J_vS*(1-sigma_s[i])*(Cp[i]-Cd[i]*np.exp(-Pe_s[i]))/(1-np.exp(-Pe_s[i])) for i in range(len(solutes))])
    J_sL = np.array([J_vL*(1-sigma_l[i])*(Cp[i]-Cd[i]*np.exp(-Pe_l[i]))/(1-np.exp(-Pe_l[i])) for i in range(len(solutes))])
    # print(J_sS)

    # #print("Js", J_sS+J_sL)
    dxdt = np.ravel([(J_sS[i] + J_sL[i])/V[t]-Cd[i]*(J_vC + J_vS + J_vL-L)/V[t] for i in range(len(solutes))])
    # print(dxdt)
    return dxdt


#%%
'''

mode-flowrate(- daytime/nighttime)( - V-volume)

'''
#modes = ['static-100', 'sorbent-100-daytime', 'sorbent-100-V-5', 'sorbent-100-nighttime', 'CFPD-100-V-10']
#modes = ['static-100', 'sorbent-100-daytime', 'sorbent-100-nighttime', 'CFPD-100-V-10']
modes = ['sorbent-100-nighttime']
# modes = ['static-100', 'sorbent-100-V-1', 'sorbent-100-V-5', 'sorbent-100-V-10', 'sorbent-100-V-20']
# modes = ['static-100', 'sorbent-50', 'sorbent-100', 'sorbent-150', 'sorbent-200']
solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
      
t = 480 #min

#fig, ax = plt.subplots(1,1)
#markers = ['o', '^', '*', 's', '>', '<']
#total Urea removal 
UR = [] #For each of these four, the value is calculated for each mode
CR = []
PhR = []
PoR = []

gluc_values=([75.5, 126, 214])

num_patients = 10
results_per_patient = {} #Overall dictionary in which the data will be stored per patient, this big dictionary is replenished outside of the loop 

# Loop for creating multiple patients
for patient_index in range(num_patients):
    patient_data = {}  
        
    '''Plasma solute concentration''' # Randomly generate parameters for each patient (mmol/L)
    sod = np.random.randint(117, 141)  
    urea= np.random.randint(125, 432) / 10  
    crea = np.random.randint(154, 617) / 1000  
    pot = np.random.randint(410, 610) / 100  
    gluc = np.random.randint(350, 550) / 100 #10.1136/archdischild-2015-308336
    phos = np.random.randint(120, 297) / 100  

    # Plasma solute concentration (initial)
    cp = pd.DataFrame(columns=["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"])
    cp.loc[0] = [urea, crea, sod, phos, gluc, pot]
    
    
    patient_data['Plasma_Solute_Concentration'] = cp

    '''dialysate solute concentration at time 0'''
    sod = 132 #mmol/L
    urea = 0
    crea = 0
    pot = 0
    gluc = np.random.choice([75.5,126,214])
    phos = 0
    cd = pd.DataFrame(columns= ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"])
    cd.loc[0] = [urea, crea, sod, phos, gluc, pot]

    patient_data['Dialysate_Solute_Concentration'] = cd

    '''dialysate volume'''
    V = np.zeros(481)
    V_fill = np.random.randint(85,295) #ml
    Vr = np.random.randint(13.4,46.6) #ml
    while V[480] <= V_fill:
        V[480] = np.random.randint(99.2,399.8) #ml
    V[0] = V_fill + Vr 
    df_V = np.array([V[0], V[480]])
    f_V = interp1d([0,480], df_V)
    interpolated_V = f_V(range(0,t+1))
    V = np.transpose(interpolated_V)
    
    patient_data['Dialysate_Volume'] = V

    '''MTAC'''
    sod = 4.5 #ml/min  # https://doi.org/10.3747/pdi.2018.00225
    urea = np.random.randint(27.79,37.69)  
    crea = np.random.randint(33.51, 137.50)
    pot = np.random.randint(1024,1271)/100 # DOI: 10.1681/ASN.V7112385
    gluc = np.random.randint(1670,1814)/100 # DOI: 10.1681/ASN.V7112385
    phos = 10.2 # https://doi.org/10.3747/pdi.2018.00225
    
    mtac = np.array([urea, crea, sod, phos, gluc, pot])
    patient_data['MTAC'] = mtac

    for gluc in gluc_values:    
        for j, mode in enumerate(modes):
        
            k_Th = np.array([0, 0, 1.758, 1994, 0.160, 1295]) #from in vitro experiments in pigs
            q_e = np.array([0, 0, 0.053, 8.408e-18, 49.04, 0]) #from in vitro experiments in pigs, there is no data for urea and creatinine so their absorption was assumed zero
            f_avg = 100 #flowrateml/min
            x_AC = 300
            
        
            f_avg = int(mode.split('-')[1])
        
            VDR = 10000 #10 L
        
            if 'V' in mode.split('-'):
                VDR = int(mode.split('-')[-1])*1000
    
            mtac = np.array([urea, crea, sod, phos, gluc, pot])
            MTAC = mtac
    
            LS = 0.074 #ml/min/mmHg
        
            # fractional pore coefficients
            alpha = [0.020, 0.900, 0.080]
        
            delP0 = 22 #mmHg
        
            # constant
            RT = 19.3 #mmHg per mmol/l
          
            #small pore radius
            rs = 43 # Angstrom
            #large pore radius
            rl = 250
        
            solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
            #radius of molecules
            r = np.array([ 2.6, 3.0, 2.3, 2.77, 3.7, 2.8]) #the phosphate radius is approximated from its topological surface area
            #for radius, new paper by Oberg - https://journals.sagepub.com/doi/suppl/10.1177/08968608211069232
            #constants to calculate sigma
            gamma_s = r/rs
            gamma_l = r/rl
        
            L = 0.3 #l/min 
        
            abya0_s = 1+9/8*gamma_s*np.log(gamma_s)-1.56034*gamma_s+0.528155*gamma_s**2+\
                1.91521*gamma_s**3-2.81903*gamma_s**4+0.270788*gamma_s**5+1.10115*gamma_s**6+ 0.435933*gamma_s**7 #eq 21 two pore Ficoll
            abya0_l = 1+9/8*gamma_l*np.log(gamma_l)-1.56034*gamma_l+0.528155*gamma_l**2+\
                1.91521*gamma_l**3-2.81903*gamma_l**4+0.270788*gamma_l**5+1.10115*gamma_l**6+ 0.435933*gamma_l**7
        
            # #Osmotic reflection coefficients
            sigma_s = np.zeros(len(solutes))
            sigma_l = np.zeros(len(solutes))
            sigma = np.zeros(len(solutes))
    
            for i in range(len(solutes)):
                sigma_s[i] = 16/3 * (gamma_s[i])**2 - 20/3 * (gamma_s[i])**3 + 7/3 * (gamma_s[i])**4
                sigma_l[i] = 16/3 * (gamma_l[i])**2 - 20/3 * (gamma_l[i])**3 + 7/3 * (gamma_l[i])**4
                sigma[i] = alpha[0] + alpha[1] * sigma_s[i] + alpha[2] * sigma_l[i]
            
            phi = np.array([1, 1, 2*0.96, 1, 1, 1])
            
            for transport_status_key in ['slow', 'average', 'fast']: 
                PS_s = mtac * 0.998 * abya0_s #fraction small pore surface area - Rippe, A THREE-PORE MODEL OF PERITONEAL TRANSPORT table 1
                # print(PS_s)
                #The current albumin value is from Joost.
                #MTAC for large pores
                PS_l = mtac * 0.002 *abya0_l# Ref: two pore model-oberg,rippe, table 1, A0L/A0 value
                #Defining transport status
                if transport_status_key== 'slow':
                    PS_s = mtac * 0.998 * abya0_s*(15000/25000) 
                    PS_l = mtac * 0.002 *abya0_l*(15000/25000)
                elif transport_status_key == 'average':
                    PS_s = mtac * 0.998 * abya0_s
                    PS_l = mtac * 0.002 *abya0_l
                elif transport_status_key == 'fast':
                    PS_s = mtac * 0.998 * abya0_s*(45000/25000)
                    PS_l = mtac * 0.002 *abya0_l*(45000/25000)
            
                predicted_cd, DR_conc =  objective(mtac, cd, cp, L, V, Vr, V_fill, mode, transport_status_key, PS_s,PS_l)
                patient_data[transport_status_key]=copy.deepcopy(predicted_cd)
                    
                # total solute removal
            
                UR.append((V[t]*predicted_cd.loc[t-1, 'Urea'] + VDR * DR_conc[t-1,0])/1000)
                CR.append((V[t]*predicted_cd.loc[t-1, 'Creatinine'] + VDR * DR_conc[t-1,1])/1000)
                PhR.append((V[t]*predicted_cd.loc[t-1, 'Phosphate'] + VDR * DR_conc[t-1,3])/1000)
                PoR.append((V[t]*predicted_cd.loc[t-1, 'Potassium'] + VDR * DR_conc[t-1,5])/1000)
        
                patient_data['Urea_clearance'] = UR
                patient_data['Creatinine_clearance']=CR
                patient_data['Phosphate_clearance']=PhR
                patient_data['Potassium clearance']=PoR
        
            
            
            #urea
        
            #ax[0,0].plot(np.arange(t),predicted_cd['Urea']/cp.iloc[0,0], label = mode, marker = markers[j], markevery = 40)
            # # ax[0,0].text(0.6, 0.1, f'MTAC = {result["x"][0]:.2f} ml/min', transform=ax[0,0].transAxes)
            #ax[0,0].set_title("Urea")
        
            # #creatinine
    
            # ax[0,1].plot(np.arange(t),predicted_cd['Creatinine']/cp.iloc[0,1], marker = markers[j], markevery = 40)
            # # ax[0,1].text(0.6, 0.1, f'MTAC = {result["x"][1]:.2f} ml/min', transform=ax[0,1].transAxes)
            # ax[0,1].set_title("Creatinine")
    
            # #Sodium
    
            # ax[1,0].plot(np.arange(t),predicted_cd['Sodium']/cp.iloc[0,2], marker = markers[j], markevery = 40)
            # # ax[1,0].text(0.6, 0.5, f'MTAC = {result["x"][2]:.2f} ml/min', transform=ax[1,0].transAxes)
            # ax[1,0].set_title("Sodium")
    
            # #Phosphate
        
            # ax[1,1].plot(np.arange(t),predicted_cd['Phosphate']/cp.iloc[0,3], marker = markers[j] , markevery = 40)
            # # ax[1,1].text(0.6, 0.1, f'MTAC = {result["x"][3]:.2f} ml/min', transform=ax[1,1].transAxes)
            # ax[1,1].set_title("Phosphate")
        
            #Glucose
    
            # ax[2,0].plot(np.arange(t),predicted_cd['Glucose']/cp.iloc[0,4], marker = markers[j], markevery = 40)
            # # ax[2,0].text(0.6, 0.5, f'MTAC = {result["x"][4]:.4f} ml/min', transform=ax[2,0].transAxes)
            # ax[2,0].set_title("Glucose")
        
            #Potassium
        
            # ax[2,1].plot(np.arange(t),predicted_cd['Potassium']/cp.iloc[0,5], label = mode, marker = markers[j], markevery = 40)
            # # ax[2,1].text(0.6, 0.1, f'MTAC = {result["x"][5]:.2f} ml/min', transform=ax[2,1].transAxes)
            # ax[2,1].set_title("Potassium")
        
       
            #ax.plot(np.arange(t),predicted_cd['Glucose']/cp.iloc[0,4], marker = markers[j], markevery = 40, label = mode)
            #ax[2,0].text(0.6, 0.5, f'MTAC = {result["x"][4]:.4f} ml/min', transform=ax[2,0].transAxes)
            #ax.set_title("Glucose")
                
            #fig.supxlabel("time, min")
            #fig.supylabel("Dialysate/Plasma concentration")
            #plt.suptitle("Predictions of dialysate concentration")
            #plt.subplots_adjust(top=0.88,
             #                   bottom=0.11,
              #                  left=0.09,
               #                 right=0.9,
                #                hspace=0.295,
                 #               wspace=0.215)
            # print(var)
            # arr_reshaped = data.reshape(data.shape[0], -1)
            # np.savetxt("syn-data.csv", arr_reshaped)
            # np.savetxt('MTAC-syn-data.csv', MTAC)
     
        results_per_patient[patient_index] = patient_data #For each patient index, the information that should be stored in the results_per_patient library is patient_data
        
#plt.legend()
#plt.tight_layout()  


# Accessing patient data; serve the purpose of iterating over the results_per_patient dictionary and printing out each patient's index along with their corresponding data.
for patient_index, data in results_per_patient.items():
    print("Patient", patient_index)
    print(data)

clearances_array = np.array([UR, CR, PhR, PoR])
print(clearances_array)

# Calculate the mean clearance values for each mode per solute across all patients
# Reshape the array to have dimensions (solute_count, mode_count, patient_count)
reshaped_clearances_array = clearances_array.reshape(4, -1, 3)

UR_values = reshaped_clearances_array[0]
CR_values = reshaped_clearances_array[1]
PhR_values = reshaped_clearances_array[2]
PoR_values = reshaped_clearances_array[3]

# Calculate the mean for each solute and each protein intake level
mean_clearances = np.mean(reshaped_clearances_array, axis=1)

# Extract the mean values for low, mid, and high protein intake
mean_clearances_low = mean_clearances[0,:]
mean_clearances_mid = mean_clearances[1,:]
mean_clearances_high = mean_clearances[2,:]

mean_UR_clearances= np.mean(UR_values, axis=0)
mean_CR_clearances=np.mean(CR_values, axis=0)
mean_PhR_clearances=np.mean(PhR_values, axis=0)
mean_PoR_clearances=np.mean(PoR_values, axis=0)

#Standard deviation 
std_UR_clearances=np.std(UR_values, axis=0)
std_CR_clearances=np.std(CR_values, axis=0)
std_PhR_clearances=np.std(PhR_values, axis=0)
std_PoR_clearances=np.std(PoR_values, axis=0)

# Create a dictionary to store mean values for each mode (mmol)
mean_clearance_per_solute = {
    'UR': mean_UR_clearances,
    'CR': mean_CR_clearances,
    'PhR': mean_PhR_clearances,
    'PoR': mean_PoR_clearances
}

print("Mean values dictionary:")
print(mean_clearance_per_solute)


# Calculate the mean clearance values for each mode per solute across all patients
# Reshape the array to have dimensions (solute_count, mode_count, patient_count)
reshaped_clearances_array = clearances_array.reshape(4, -1, 3)

UR_array=np.array([UR])
CR_array=np.array([CR])
PhR_array=np.array([PhR])
PoR_array=np.array([PoR])

reshaped_UR_clearance= UR_array.reshape(10,3,3)
reshaped_CR_clearance= CR_array.reshape(10,3,3)
reshaped_PhR_clearance= PhR_array.reshape(10,3,3)
reshaped_PoR_clearance=PoR_array.reshape(10,3,3)

# Calculate the mean for each solute and each protein intake level --> this doesnt take into account glucose
mean_clearances = np.mean(reshaped_clearances_array, axis=1)

#gluc_values=([75.5, 126, 214])
#solute-DPI-glucose concentration
mean_values_TS_gluc={
    'urea-slow-low': np.mean(reshaped_UR_clearance[:,0,0]),
    'urea-slow-mid': np.mean(reshaped_UR_clearance[:,1,0]),
    'urea-slow-high': np.mean(reshaped_UR_clearance[:,2,0]),
    'urea-average-low': np.mean(reshaped_UR_clearance[:,0,1]),
    'urea-average-mid': np.mean(reshaped_UR_clearance[:,1,1]),
    'urea-average-high': np.mean(reshaped_UR_clearance[:,2,1]),
    'urea-fast-low': np.mean(reshaped_UR_clearance[:,0 ,2]),
    'urea-fast-mid': np.mean(reshaped_UR_clearance[:,1,2]),
    'urea-fast-high': np.mean(reshaped_UR_clearance[:,2,2]),
    'crea-slow-low': np.mean(reshaped_CR_clearance[:,0,0]),
    'crea-slow-mid': np.mean(reshaped_CR_clearance[:,1,0]),
    'crea-slow-high': np.mean(reshaped_CR_clearance[:,2,0]),
    'crea-average-low': np.mean(reshaped_CR_clearance[:,0,1]),
    'crea-average-mid': np.mean(reshaped_CR_clearance[:,1,1]),
    'crea-average-high': np.mean(reshaped_CR_clearance[:,2,1]),
    'crea-fast-low': np.mean(reshaped_CR_clearance[:,0 ,2]),
    'crea-fast-mid': np.mean(reshaped_CR_clearance[:,1,2]),
    'crea-fast-high': np.mean(reshaped_CR_clearance[:,2,2]),
    'phos-slow-low': np.mean(reshaped_PhR_clearance[:,0,0]),
    'phos-slow-mid': np.mean(reshaped_PhR_clearance[:,1,0]),
    'phos-slow-high': np.mean(reshaped_PhR_clearance[:,2,0]),
    'phos-average-low': np.mean(reshaped_PhR_clearance[:,0,1]),
    'phos-average-mid': np.mean(reshaped_PhR_clearance[:,1,1]),
    'phos-average-high': np.mean(reshaped_PhR_clearance[:,2,1]),
    'phos-fast-low': np.mean(reshaped_PhR_clearance[:,0 ,2]),
    'phos-fast-mid': np.mean(reshaped_PhR_clearance[:,1,2]),
    'phos-fast-high': np.mean(reshaped_PhR_clearance[:,2,2]),
    'pot-slow-low': np.mean(reshaped_PoR_clearance[:,0,0]),
    'pot-slow-mid': np.mean(reshaped_PoR_clearance[:,1,0]),
    'pot-slow-high': np.mean(reshaped_PoR_clearance[:,2,0]),
    'pot-average-low': np.mean(reshaped_PoR_clearance[:,0,1]),
    'pot-average-mid': np.mean(reshaped_PoR_clearance[:,1,1]),
    'pot-average-high': np.mean(reshaped_PoR_clearance[:,2,1]),
    'pot-igh-low': np.mean(reshaped_PoR_clearance[:,0 ,2]),
    'pot-fast-mid': np.mean(reshaped_PoR_clearance[:,1,2]),
    'pot-fast-high': np.mean(reshaped_PoR_clearance[:,2,2])
    }

mean_values_TS_gluc_array=np.array(list(mean_values_TS_gluc.values()))    
reshaped_mean_values_TS_gluc=mean_values_TS_gluc_array.reshape(4,3,3)
mean_UR_clearances=reshaped_mean_values_TS_gluc[0,:,:]
mean_CR_clearances=reshaped_mean_values_TS_gluc[1,:,:]
mean_PhR_clearances=reshaped_mean_values_TS_gluc[2,:,:]
mean_PoR_clearances=reshaped_mean_values_TS_gluc[3,:,:]

std_values_TS_gluc={
    'urea-slow-low': np.std(reshaped_UR_clearance[:,0,0]),
    'urea-slow-mid': np.std(reshaped_UR_clearance[:,1,0]),
    'urea-slow-high': np.std(reshaped_UR_clearance[:,2,0]),
    'urea-average-low': np.std(reshaped_UR_clearance[:,0,1]),
    'urea-average-mid': np.std(reshaped_UR_clearance[:,1,1]),
    'urea-average-high': np.std(reshaped_UR_clearance[:,2,1]),
    'urea-fast-low': np.std(reshaped_UR_clearance[:,0 ,2]),
    'urea-fast-mid': np.std(reshaped_UR_clearance[:,1,2]),
    'urea-fast-high': np.std(reshaped_UR_clearance[:,2,2]),
    'crea-slow-low': np.std(reshaped_CR_clearance[:,0,0]),
    'crea-slow-mid': np.std(reshaped_CR_clearance[:,1,0]),
    'crea-slow-high': np.std(reshaped_CR_clearance[:,2,0]),
    'crea-average-low': np.std(reshaped_CR_clearance[:,0,1]),
    'crea-average-mid': np.std(reshaped_CR_clearance[:,1,1]),
    'crea-average-high': np.std(reshaped_CR_clearance[:,2,1]),
    'crea-fast-low': np.std(reshaped_CR_clearance[:,0 ,2]),
    'crea-fast-mid': np.std(reshaped_CR_clearance[:,1,2]),
    'crea-fast-high': np.std(reshaped_CR_clearance[:,2,2]),
    'phos-slow-low': np.std(reshaped_PhR_clearance[:,0,0]),
    'phos-slow-mid': np.std(reshaped_PhR_clearance[:,1,0]),
    'phos-slow-high': np.std(reshaped_PhR_clearance[:,2,0]),
    'phos-average-low': np.std(reshaped_PhR_clearance[:,0,1]),
    'phos-average-mid': np.std(reshaped_PhR_clearance[:,1,1]),
    'phos-average-high': np.std(reshaped_PhR_clearance[:,2,1]),
    'phos-fast-low': np.std(reshaped_PhR_clearance[:,0 ,2]),
    'phos-fast-mid': np.std(reshaped_PhR_clearance[:,1,2]),
    'phos-fast-high': np.std(reshaped_PhR_clearance[:,2,2]),
    'pot-slow-low': np.std(reshaped_PoR_clearance[:,0,0]),
    'pot-slow-mid': np.std(reshaped_PoR_clearance[:,1,0]),
    'pot-slow-high': np.std(reshaped_PoR_clearance[:,2,0]),
    'pot-average-low': np.std(reshaped_PoR_clearance[:,0,1]),
    'pot-average-mid': np.std(reshaped_PoR_clearance[:,1,1]),
    'pot-average-high': np.std(reshaped_PoR_clearance[:,2,1]),
    'pot-fast-low': np.std(reshaped_PoR_clearance[:,0 ,2]),
    'pot-fast-mid': np.std(reshaped_PoR_clearance[:,1,2]),
    'pot-fast-high': np.std(reshaped_PoR_clearance[:,2,2])
    }

std_values_TS_gluc_array=np.array(list(std_values_TS_gluc.values()))    
reshaped_std_values_TS_gluc=std_values_TS_gluc_array.reshape(4,3,3)
std_UR_clearances=reshaped_std_values_TS_gluc[0,:,:]
std_CR_clearances=reshaped_std_values_TS_gluc[1,:,:]
std_PhR_clearances=reshaped_std_values_TS_gluc[2,:,:]
std_PoR_clearances=reshaped_std_values_TS_gluc[3,:,:]


all_dialysate_concentrations = []

# Iterate over the results for each patient
for patient_index, data in results_per_patient.items():
    # Extract dialysate concentrations for the current patient
    dialysate_concentrations = data['Dialysate_Solute_Concentration'].values
    # Append to the list
    all_dialysate_concentrations.extend(dialysate_concentrations)
    
# Convert the list to a NumPy array
all_dialysate_concentrations = np.array(all_dialysate_concentrations)

reshaped_cd_array= all_dialysate_concentrations.reshape(10, -1, 6)




#%%
#Plot for 4 solutes
transport_status=['slow','average', 'fast']
gluc_values = [75.5, 126, 214]
fig, ax = plt.subplots(4, 1, figsize=(10,20), sharex=True) 
solute_names = ['Urea', 'Creatinine', 'Phosphate', 'Potassium']
mean_clearances = [mean_UR_clearances, mean_CR_clearances, mean_PhR_clearances, mean_PoR_clearances]
std_clearances = [std_UR_clearances, std_CR_clearances, std_PhR_clearances, std_PoR_clearances]



for i, (solute, mean_values, std_values) in enumerate(zip(solute_names, mean_clearances, std_clearances)):
    for j, transport_status_key in enumerate(transport_status):
        for k, gluc_value in enumerate(gluc_values):
            # Compute the index for xtick
            xtick_index = j * len(gluc_values) + k
            ax[i].errorbar(xtick_index, mean_values[j, k], yerr=std_values[j, k], fmt='o', capsize=5)
    # Set solute name as title above each subplot
    ax[i].set_title(solute, fontsize=12)
      
xtick_labels = [f" TS= {transport_status_key}, CD_Glu={gluc}" for transport_status_key in transport_status for gluc in gluc_values]
for axis in ax:
    axis.set_xticks(range(len(transport_status) * len(gluc_values)))
    axis.set_xticklabels(xtick_labels, rotation=35, fontsize=8)
    
fig.supxlabel('Transport Status (TS), CD Glucose (mmol/L)')
fig.supylabel('Total solute removal (mmol)')
fig.suptitle('Influence of CD Glucose on Solute Clearance in relation to Transport Status', fontsize=14)

plt.tight_layout()
plt.show()

#%%

#Calculations to compare it to SAPD data
#to get the clearance from the model to ml/min
#→ elimination in mol to mg by multiplying by the molar mass for the solutes &1000
#→ elimination in mg dividing by the total time, 480 min
#→ calculate clearance in mg/min
#→ divide by the concentration
#→ calculate the other clearance to ml/min as well

#Clearance per mode, per solute in mg
mean_clearance_mg = {
    'UR': (mean_UR_clearances)*molecular_weight['Urea'],
    'CR': (mean_CR_clearances)*molecular_weight['Creatinine'],
    'PhR': (mean_PhR_clearances)*molecular_weight['Phosphate'],       
    'PoR': (mean_PoR_clearances)*molecular_weight['Potassium']
    }

#Elimination rate (mg/min)
Elimination_rate= {
    'UR':mean_clearance_mg['UR']/480,
    'CR':mean_clearance_mg['CR']/480,
    'PhR':mean_clearance_mg['PhR']/480,
    'PoR':mean_clearance_mg['PoR']/480
    }

#Mean plasma concentration (mmol/L)
plasma_concentration_data=[]
for patient_index, patient_data in results_per_patient.items():
    # Extract plasma concentration data for the current patient
    cp_data=patient_data['Plasma_Solute_Concentration']
    
    plasma_concentration_data.append(cp_data)

# Concatenate the list of DataFrames into a single DataFrame
plasma_concentration_df = pd.concat(plasma_concentration_data, ignore_index=True)

mean_cp_per_solute = np.mean(plasma_concentration_df, axis=0)

#Plasma concentration (mg/L)
plasma_concentration_mg= {
    'UR': mean_cp_per_solute['Urea']*molecular_weight['Urea'],
    'CR': mean_cp_per_solute['Creatinine']*molecular_weight['Creatinine'],
    'PhR': mean_cp_per_solute['Phosphate']*molecular_weight['Phosphate'],
    'PoR': mean_cp_per_solute['Potassium']*molecular_weight['Potassium']
    }

#Average solute clearance per mode (ml/min) (elimination rate/ concentration) 
Clearance_for_comparison={
    'UR': (Elimination_rate['UR']/plasma_concentration_mg['UR'])*1000,
    'CR': (Elimination_rate['CR']/plasma_concentration_mg['CR'])*1000,
    'PhR': (Elimination_rate['PhR']/plasma_concentration_mg['PhR'])*1000,
    'PoR': (Elimination_rate['PoR']/plasma_concentration_mg['PoR'])*1000
    }
#%%
import json

# Convert DataFrames to dictionaries
for patient_index, patient_data in results_per_patient.items():
    for key, value in patient_data.items():
        if isinstance(value, pd.DataFrame):
            patient_data[key] = value.to_dict()
        elif isinstance(value, np.ndarray):
            patient_data[key] = value.tolist()

# Specify the file path to save the JSON file
file_path = 'trail2_patients_gluc.json'

# Save the dictionary to a JSON file
with open(file_path, 'w') as f:
    json.dump(results_per_patient, f)

print(f"Dictionary saved to {file_path}")

# Read JSON file into a DataFrame
df_from_json = pd.read_json('trail2_patients_gluc.json')

# Save DataFrame to a CSV file
df_from_json.to_csv('trial2_patients_gluc.csv', index=False)


