# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:35:53 2023

@author: P70073624
"""

# import cyipopt 
import numpy as np
import os
import glob
# import numdifftools.nd_statsmodels as nd   
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import scipy
from scipy.optimize import minimize
import pandas as pd
from scipy.interpolate import interp1d
import random
# import statistics
import csv
# from csv import writer
from itertools import count
import copy


def sorbent_chamber(c_ddr, t):
    # Convert from mmol/L to mg/ml as thomas eq uses that unit
    c_us = copy.deepcopy(c_ddr)
    for s, solute in enumerate(solutes):
        c_us[s] *= (molecular_weight[solute]/1000)
    # Thomas model
    c_ds = c_us*(1/(1+np.exp(k_Th/f_avg*(q_e*x_AC-c_us*f_avg*t)))).ravel() 
    # Convert back to mmol/L
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

def objective(mtac, predicted_cd, Cp, L, V, Vr, V_fill, mode):
    '''The objective function needed to be minimised'''
    
    #print(VDR)
    t = 480 #min
    
    predicted_cd, DR_conc = rk(t, mtac, predicted_cd, Cp, L, V, Vr, V_fill, mode)

    
    return predicted_cd, DR_conc

def rk(t, mtac, predicted_cd, Cp, L, V, Vr, V_fill, mode):
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
modes = ['static-100', 'sorbent-100-daytime', 'sorbent-100-nighttime', 'CFPD-100-V-10']
# modes = ['static-100', 'sorbent-100-V-1', 'sorbent-100-V-5', 'sorbent-100-V-10', 'sorbent-100-V-20']
# modes = ['static-100', 'sorbent-50', 'sorbent-100', 'sorbent-150', 'sorbent-200']
solutes = ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"]
      
t = 480 #min

fig, ax = plt.subplots(3, 2, figsize=(12, 10))  #initiate plot for dialysate concentration
markers = ['o', '^', '*', 's', '>', '<']
#total Urea removal 
UR = []
CR = []
PhR = []
PoR = []

#Patient data from the study by Raaijmakers et al. (2010)
patients_data = [
    {
        'patient_id': 1,
        'Plasma solute concentration': {'Urea': 18, 'Creatinine': 0.365, 'Sodium': 137, 'Phosphate': 2.82, 'Glucose': 3.57, 'Potassium': 5.2}, #mmol/L
        'dialysate solute concentration at time 0': {'Urea': 0, 'Creatinine': 0, 'Sodium': 132, 'Phosphate': 0, 'Glucose': 75.5, 'Potassium': 0}, #mmol/L
        'MTAC': {'Urea': 27.79, 'Creatinine': 45.491,'Sodium': 4.5, 'Phosphate': 10.2, 'Glucose': 16.7, 'Potassium': 12.71},  # List of MTACs for each solute # ml/min
        'V_fill': 295,  # ml
        'Vr': 46.61,  # ml
        'V[480]': 399.8, #ml
        'f_avg': 34.1 #µl/min
    },
    {   'patient_id': 2,
        'Plasma solute concentration': {'Urea': 43.2, 'Creatinine': 0.365, 'Sodium': 117, 'Phosphate': 1.9, 'Glucose': np.random.randint(350, 550) / 100, 'Potassium': 4.2}, #mmol/L
        'dialysate solute concentration at time 0': {'Urea': 0, 'Creatinine': 0, 'Sodium': 132, 'Phosphate': 0, 'Glucose': 75.5, 'Potassium': 0}, #mmol/L
        'MTAC': {'Urea': 31.272, 'Creatinine': 33.514,'Sodium': 4.5, 'Phosphate': 10.2, 'Glucose': 16.7, 'Potassium': 12.71},  # List of MTACs for each solute # ml/min
        'V_fill': 185,  # ml
        'Vr': 29.23,  # ml
        'V[480]': 260.5,#ml
        'f_avg': 21.4 #ml/min
    },
    {   'patient_id': 3,
        'Plasma solute concentration': {'Urea': 21.4, 'Creatinine': 0.372, 'Sodium': 141, 'Phosphate': 2.26, 'Glucose': np.random.randint(350, 550) / 100, 'Potassium': 4.1}, #mmol/L
        'dialysate solute concentration at time 0': {'Urea': 0, 'Creatinine': 0, 'Sodium': 132, 'Phosphate': 0, 'Glucose': 75.5, 'Potassium': 0}, #mmol/L
        'MTAC': {'Urea': 28.498, 'Creatinine': 36.792,'Sodium': 4.5, 'Phosphate': 10.2, 'Glucose': 18.14, 'Potassium': 10.24},  # List of MTACs for each solute # ml/min
        'V_fill': 200,  # ml
        'Vr': 31.6,  # ml
        'V[480]': 314.9, #ml
        'f_avg': 23.1 #ml/min
    },
    {   'patient_id': 4, #Note in the study this is patient 5 since patient 4 was not included in this model due to varying glucose concentrations
        'Plasma solute concentration': {'Urea': 12.5, 'Creatinine': 0.154, 'Sodium': 141, 'Phosphate': 2.97, 'Glucose': np.random.randint(350, 550) / 100, 'Potassium':6.1}, #mmol/L
        'dialysate solute concentration at time 0': {'Urea': 0, 'Creatinine': 0, 'Sodium': 132, 'Phosphate': 0, 'Glucose': 126, 'Potassium': 0}, #mmol/L
        'MTAC': {'Urea': 31.464, 'Creatinine': 113.99,'Sodium': 4.5, 'Phosphate': 10.2, 'Glucose': 18.14, 'Potassium': 10.24},  # List of MTACs for each solute # ml/min
        'V_fill': 85, #ml
        'Vr': 13.43,#ml
        'V[480]': 99.15,#ml
        'f_avg': 9.8 #ml/min
     },
    {   'patient_id': 5, #Note in the study this is patient 6 since patient 4 was not included in this model due to varying glucose concentrations
        'Plasma solute concentration': {'Urea': 20.9, 'Creatinine': 0.168, 'Sodium': 139, 'Phosphate': 1.2, 'Glucose': np.random.randint(350, 550) / 100, 'Potassium':4.2}, #mmol/L
        'dialysate solute concentration at time 0': {'Urea': 0, 'Creatinine': 0, 'Sodium': 132, 'Phosphate': 0, 'Glucose': 214, 'Potassium': 0}, #mmol/L
        'MTAC': {'Urea':37.695, 'Creatinine':137.497,'Sodium': 4.5, 'Phosphate': 10.2, 'Glucose': 18.14, 'Potassium': 10.24},  # List of MTACs for each solute # ml/min
        'V_fill': 185, #ml
        'Vr': 29.23,  #ml
        'V[480]': 251.7,#ml
        'f_avg': 21.4 #ml/min
     }
]

#Integrate patient data as variables
for patient in patients_data:
    V = np.zeros(481)
    V_fill = patient['V_fill']
    Vr = patient['Vr']
    V[480]= patient['V[480]']
    V[0] = V_fill + Vr 
    df_V = np.array([V[0], V[480]])
    f_V = interp1d([0,480], df_V)
    interpolated_V = f_V(range(0,t+1))
    V = np.transpose(interpolated_V)
    patient['V']= V


for patient in patients_data:
    # Extract patient-specific data
    patient_id = patient['patient_id']
    cp = pd.DataFrame(patient['Plasma solute concentration'], index=[0])
    cd = pd.DataFrame(patient['dialysate solute concentration at time 0'], index=[0])
    
'''MTAC'''
for patient in patients_data:
    sod= patient['MTAC']['Sodium']
    urea = patient['MTAC']['Urea']
    crea = patient['MTAC']['Creatinine']
    pot = patient['MTAC']['Potassium']
    gluc = patient['MTAC']['Glucose']
    phos = patient['MTAC']['Phosphate']
    
    mtac= np.array([urea, crea, sod, phos, gluc, pot])
    
    for j, mode in enumerate(modes):
        
        k_Th = np.array([0, 0, 1.758, 1994, 0.160, 1295]) #from in vitro experiments in pigs
        q_e = np.array([0, 0, 0.053, 8.408e-18, 49.04, 0]) #from in vitro experiments in pigs, there is no data for urea and creatinine so their absorption was assumed zero
        x_AC = 300
        
        f_avg=patient['f_avg'] #flowrate ml/min 
    
        VDR = 10000 #10 L
        
        if 'V' in mode.split('-'):
            VDR = int(mode.split('-')[-1])*1000
        
        mtac= np.array([urea, crea, sod, phos, gluc, pot])
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
        
        predicted_cd, DR_conc =  objective(mtac, cd, cp, L, V, Vr, V_fill, mode)
        
        # total solute removal
        
        UR.append((V[t]*predicted_cd.loc[t-1, 'Urea'] + VDR * DR_conc[t-1,0])/1000)
        CR.append((V[t]*predicted_cd.loc[t-1, 'Creatinine'] + VDR * DR_conc[t-1,1])/1000)
        PhR.append((V[t]*predicted_cd.loc[t-1, 'Phosphate'] + VDR * DR_conc[t-1,3])/1000)
        PoR.append((V[t]*predicted_cd.loc[t-1, 'Potassium'] + VDR * DR_conc[t-1,5])/1000)
        
        
        #Plot dialysate concentrations over time
        #urea
    
        ax[0,0].plot(np.arange(t),predicted_cd['Urea']/cp.iloc[0,0], marker = markers[j], markevery = 40)
        # # ax[0,0].text(0.6, 0.1, f'MTAC = {result["x"][0]:.2f} ml/min', transform=ax[0,0].transAxes)
        ax[0,0].set_title("Urea")
        
        # #creatinine
        
        ax[0,1].plot(np.arange(t),predicted_cd['Creatinine']/cp.iloc[0,1],label=mode, marker = markers[j], markevery = 40)
        # # ax[0,1].text(0.6, 0.1, f'MTAC = {result["x"][1]:.2f} ml/min', transform=ax[0,1].transAxes)
        ax[0,1].set_title("Creatinine")
        
        # #Sodium
        
        ax[1,0].plot(np.arange(t),predicted_cd['Sodium']/cp.iloc[0,2], marker = markers[j], markevery = 40)
        # # ax[1,0].text(0.6, 0.5, f'MTAC = {result["x"][2]:.2f} ml/min', transform=ax[1,0].transAxes)
        ax[1,0].set_title("Sodium")
        
        # #Phosphate
        
        ax[1,1].plot(np.arange(t),predicted_cd['Phosphate']/cp.iloc[0,3], marker = markers[j] , markevery = 40)
        # # ax[1,1].text(0.6, 0.1, f'MTAC = {result["x"][3]:.2f} ml/min', transform=ax[1,1].transAxes)
        ax[1,1].set_title("Phosphate")
        
        #Glucose
        
        ax[2,0].plot(np.arange(t),predicted_cd['Glucose']/cp.iloc[0,4], marker = markers[j], markevery = 40)
        # # ax[2,0].text(0.6, 0.5, f'MTAC = {result["x"][4]:.4f} ml/min', transform=ax[2,0].transAxes)
        ax[2,0].set_title("Glucose")
        
        #Potassium
        
        ax[2,1].plot(np.arange(t),predicted_cd['Potassium']/cp.iloc[0,5], marker = markers[j], markevery = 40)
        # # ax[2,1].text(0.6, 0.1, f'MTAC = {result["x"][5]:.2f} ml/min', transform=ax[2,1].transAxes)
        ax[2,1].set_title("Potassium")
        
        #ax.plot(np.arange(t),predicted_cd['Glucose']/cp.iloc[0,4], marker = markers[j], markevery = 40, label = mode)
        # ax[2,0].text(0.6, 0.5, f'MTAC = {result["x"][4]:.4f} ml/min', transform=ax[2,0].transAxes)
        #ax.set_title("Glucose")
        
        
        fig.legend(loc='upper right', fontsize='x-small')
        fig.supxlabel("Time (min)")
        fig.supylabel("Dialysate/Plasma concentration")
        plt.suptitle("Predictions of Dialysate/Plasma Concentration")
        plt.subplots_adjust(top=0.9,
                            bottom=0.1,
                            left=0.09,
                            right=0.86,
                            hspace=0.35,
                            wspace=0.18)
        
        # print(var)
        # arr_reshaped = data.reshape(data.shape[0], -1)
        # np.savetxt("syn-data.csv", arr_reshaped)
        # np.savetxt('MTAC-syn-data.csv', MTAC)
        
clearances_array = np.array([UR, CR, PhR, PoR])
print(clearances_array)
reshaped_clearances_array = clearances_array.reshape(4, -1, 4) #mode, patients, solutes

#%%
#convert clearance data to mg/min so it could be compared to experimental data
#Calculations to compare it to SAPD data
#to get the clearance from the model to ml/min
#→ elimination in mol to mg by multiplying by the molar mass for the solutes &1000
#→ elimination in mg dividing by the total time, 480 min
#→ calculate clearance in mg/min
#→ divide by the concentration
#→ calculate the other clearance to ml/min as well

UR_array=np.array([UR])
CR_array=np.array([CR])
PhR_array=np.array([PhR])
PoR_array=np.array([PoR])

#Clearance per mode, per solute in mg
clearance_mg = {
    'UR':UR_array*molecular_weight['Urea'],
    'CR':CR_array*molecular_weight['Creatinine'],
    'PhR':PhR_array*molecular_weight['Phosphate'],
    'PoR':PoR_array*molecular_weight['Potassium']
    }

#Elimination rate (mg/min)
Elimination_rate_sp= {
    'UR':clearance_mg['UR']/480,
    'CR':clearance_mg['CR']/480,
    'PhR':clearance_mg['PhR']/48,
    'PoR':clearance_mg['PoR']/480
    }

mean_cp_per_solute_sp = { #mmol
    'UR':np.mean(cp['Urea']),
    'CR':np.mean(cp['Creatinine']),
    'PhR':np.mean(cp['Phosphate']),
    'PoR':np.mean(cp['Potassium'])
    }

#Plasma concentration (mg/L)
plasma_concentration_mg_sp= {
    'UR': mean_cp_per_solute_sp['UR']*molecular_weight['Urea'],
    'CR': mean_cp_per_solute_sp['CR']*molecular_weight['Creatinine'],
    'PhR': mean_cp_per_solute_sp['PhR']*molecular_weight['Phosphate'],
    'PoR': mean_cp_per_solute_sp['PoR']*molecular_weight['Potassium']
    }

#Average solute clearance per mode (ml/min) (elimination rate/ concentration)
Clearance_for_comparison_sp={
    'UR': (Elimination_rate_sp['UR']/plasma_concentration_mg_sp['UR'])*1000,
    'CR': (Elimination_rate_sp['CR']/plasma_concentration_mg_sp['CR'])*1000,
    'PhR': (Elimination_rate_sp['PhR']/plasma_concentration_mg_sp['PhR'])*1000,
    'PoR': (Elimination_rate_sp['PoR']/plasma_concentration_mg_sp['PoR'])*1000
    }

#%%
# Experimental data
# Converting from ml/min to mmol

UR_array_e=np.array([4.194797688, 3.079768786, 3.028901734, 1.513294798, 3.614450867])
CR_array_e=np.array([6.138728324,	3.143930636, 3.630057803, 3.852023121, 8.961271676])

#Plasma concentration (mg/L)
plasma_concentration_mg_sp_e= {
    'UR': mean_cp_per_solute_sp['UR']*molecular_weight['Urea'],
    'CR': mean_cp_per_solute_sp['CR']*molecular_weight['Creatinine'],
    }

#Eliminatioin rate (mg/min)= clearance (ml/min)*plasma concentration (mg/L) *1000
Elimination_rate_sp_e ={
    'UR': (UR_array_e*plasma_concentration_mg_sp_e['UR'])/1000,
    'CR': (CR_array_e*plasma_concentration_mg_sp_e['CR'])/1000
    }

#Clearance per solute in mg
clearance_mg_e = {
    'UR':Elimination_rate_sp_e['UR']*480,
    'CR':Elimination_rate_sp_e['CR']*480
    }

#Solute clearance (mmol) = clearance (mg) / molecular weight (g/mol)
Clearance_for_comparison_sp_e={
    'UR': clearance_mg_e['UR']/molecular_weight['Urea'],
    'CR': clearance_mg_e['CR']/molecular_weight['Creatinine']
    }

UR_values_e = Clearance_for_comparison_sp_e['UR']
CR_values_e = Clearance_for_comparison_sp_e['CR']

# Calculating the mean for every column
mean_UR_e = np.mean(UR_values_e, axis=0)
mean_CR_e = np.mean(CR_values_e, axis=0)


#Standard deviation
std_UR_e= np.std(UR_values_e, axis=0)
std_CR_e = np.std(CR_values_e, axis=0)

# Create a dictionary to store mean values for each mode (mmol)
mean_clearance_per_mode_per_solute_e = {
    'UR': mean_UR_e,
    'CR': mean_CR_e
}

print("Mean values dictionary:")
print(mean_clearance_per_mode_per_solute_e)


UR_values = reshaped_clearances_array[0,:,3]
CR_values = reshaped_clearances_array[1,:,3]

# Calculating the mean for every column
mean_UR_modes = np.mean(UR_values, axis=0)
mean_CR_modes = np.mean(CR_values, axis=0)


#Standard deviation
std_UR_modes= np.std(UR_values, axis=0)
std_CR_modes = np.std(CR_values, axis=0)

# Create a dictionary to store mean values for each mode (mmol)
mean_clearance_per_mode_per_solute = {
    'UR': mean_UR_modes,
    'CR': mean_CR_modes
}

print("Mean values dictionary:")
print(mean_clearance_per_mode_per_solute)

solutes_e=['Urea', 'Creatinine']
data_type=['Experimental', 'Predicted']
mean_UR=[mean_UR_e,mean_UR_modes]
mean_CR=[mean_CR_e,mean_CR_modes]
std_UR=[std_UR_e,std_UR_modes]
std_CR=[std_CR_e, std_CR_modes]

#%%
#Plot experimental and predicted solute clearance
# Plot mean values with error bars
fig, ax = plt.subplots(1,2, figsize=(8,6))  

for i, (mean_values, std_values, solute_title) in enumerate(zip([mean_UR, mean_CR],[std_UR, std_CR], ['Urea','Creatinine'])):
                                                  
    ax[i].errorbar(0.35, mean_values[0], yerr=std_values[0], fmt='o', capsize=5, color='tab:orange')
    ax[i].errorbar(0.65, mean_values[1], yerr=std_values[1], fmt='o', capsize=5, color='tab:blue')
    ax[i].set_xticks([0.35, 0.65])
    # Set x-tick labels at specific locations
    ax[i].set_xticklabels(data_type, rotation=45)
    ax[i].set_xlim(0.2, 0.8)
    fig.supylabel('Total solute removal (mmol)')
    ax[0].set_title('Urea')
    ax[1].set_title('Creatinine')
    
plt.suptitle('Experimental VS Predicted Mean Total Solute Removal')
plt.tight_layout()
plt.show()
