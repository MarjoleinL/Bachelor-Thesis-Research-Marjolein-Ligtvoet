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

def objective(mtac, predicted_cd, Cp, L, V, Vr, V_fill, mode):
    '''The objective function needed to be minimised'''
    
    print(VDR)
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

#fig, ax = plt.subplots(1,1)
#markers = ['o', '^', '*', 's', '>', '<']
#total Urea removal 
UR = []
CR = []
PhR = []
PoR = []

patient_data = {}  # Dictionary (smaller, patient specific) to store data for each patient
     
'''Plasma solute concentration''' # Randomly generate parameters for each patient (mmol/L)
sod = 129 
urea = 35.3  
crea = 0.406 
pot = 5.16 
gluc = 4.23 #value generated between ranges based on 10.1136/archdischild-2015-308336
phos = 1.99

# Plasma solute concentration (initial)
cp = pd.DataFrame(columns=["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"])
cp.loc[0] = [urea, crea, sod, phos, gluc, pot]
 
patient_data['Plasma_Solute_Concentration'] = cp

'''dialysate solute concentration at time 0'''
sod = 132 #mmol/L
urea = 0
crea = 0
pot = 0
gluc = 126
phos = 0
cd = pd.DataFrame(columns= ["Urea", "Creatinine", "Sodium", "Phosphate", "Glucose", "Potassium"])
cd.loc[0] = [urea, crea, sod, phos, gluc, pot]

patient_data['Dialysate_Solute_Concentration'] = cd

'''dialysate volume'''
V = np.zeros(481)
V[480] = 149 #ml
V_fill_values = np.array([116, 233, 350]) #ml vlaues for-50%, baseline and +50%
Vr = 40 #ml
for V_fill in V_fill_values:
    V[0] = V_fill + Vr   
df_V = np.array([V[0], V[480]])
f_V = interp1d([0,480], df_V)
interpolated_V = f_V(range(0,t+1))
V = np.transpose(interpolated_V)
 
patient_data['Dialysate_Volume'] = V

'''MTAC'''
sod = 4.5 #ml/min  # https://doi.org/10.3747/pdi.2018.00225
urea = 35.1 
crea = 57.3
pot = 11.5 # DOI: 10.1681/ASN.V7112385
gluc = 18.08 # DOI: 10.1681/ASN.V7112385
phos = 10.2 # https://doi.org/10.3747/pdi.2018.00225 
 
mtac = np.array([urea, crea, sod, phos, gluc, pot])
patient_data['MTAC'] = mtac


for V_fill in V_fill_values:
    for j, mode in enumerate(modes):
    
        k_Th = np.array([0, 0, 1.758, 1994, 0.160, 1295]) #from in vitro experiments in pigs
        q_e = np.array([0, 0, 0.053, 8.408e-18, 49.04, 0]) #from in vitro experiments in pigs, there is no data for urea and creatinine so their absorption was assumed zero
        f_avg = 100  #flowrate ml/min
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
    
        #Osmotic reflection coefficients
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
    
    
        #urea
        
        # ax[0,0].plot(np.arange(t),predicted_cd['Urea']/cp.iloc[0,0], label = mode, marker = markers[j], markevery = 40)
        # # ax[0,0].text(0.6, 0.1, f'MTAC = {result["x"][0]:.2f} ml/min', transform=ax[0,0].transAxes)
        # ax[0,0].set_title("Urea")
        
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
        # ax[2,0].text(0.6, 0.5, f'MTAC = {result["x"][4]:.4f} ml/min', transform=ax[2,0].transAxes)
        #ax.set_title("Glucose")
        
        #fig.supxlabel("time, min")
        #fig.supylabel("Dialysate/Plasma concentration")
        #plt.suptitle("Predictions of dialysate concentration")
        #plt.subplots_adjust(top=0.88,
                            #bottom=0.11,
                            #left=0.09,
                            #right=0.9,
                            #hspace=0.295,
                           # wspace=0.215)
    
        # print(var)
        # arr_reshaped = data.reshape(data.shape[0], -1)
        # np.savetxt("syn-data.csv", arr_reshaped)
        # np.savetxt('MTAC-syn-data.csv', MTAC)

#plt.legend()
#plt.tight_layout()

#%%
sensitivity_array= np.array([UR, CR, PhR, PoR])
# Reshape the array to have dimensions (solute_count, mode_count, patient_count)
sensitivity_results= sensitivity_array.reshape(4, -1, 4) #4,4,-1

# Calculate percentage change compared to V-fill 233
sensitivity_change_minus50 = ((sensitivity_results[:, 0, :] - sensitivity_results[:, 1, :]) / sensitivity_results[:, 1, :]) * 100
sensitivity_change_plus50 = ((sensitivity_results[:, 2, :] - sensitivity_results[:, 1, :]) / sensitivity_results[:, 1, :]) * 100

# Get the solute labels
solutes_ssa=['Urea', 'Creatinine', 'Phosphate', 'Potassium']
solute_count = len([UR, CR, PoR, PhR])
mode_count=len(modes)

# Set the position for bars
bar_width = 0.35

solute_colors = ['DarkGreen','Navy','DarkOrange','DarkViolet']  # Define colors for each solute
lighter_colors = ['SpringGreen','LightSkyBlue','Orange','Violet' ] # Define a lighter color for v_fill 116

fig, axes = plt.subplots(4,1, figsize=(10, 18), sharex=True, sharey=True)
# Adjust spacing between subplots and the top title
plt.subplots_adjust(top=1, hspace=1)

for i in range(solute_count):  
    # Plot bars for  V_fill 350
    print(sensitivity_change_plus50[i])
    axes[i].barh(np.arange(4)-bar_width/2, sensitivity_change_plus50[i], height=bar_width, label=('V_fill 350'),
        color=solute_colors[i], alpha=0.7)
    # Plot bars for V_fill 116 with a lighter shade
    axes[i].barh(np.arange(4) + bar_width/2, sensitivity_change_minus50[i], height=bar_width, label=('V_fill 116'),
        color=lighter_colors[i], alpha=0.5)
    
    # Annotate percentage change next to each bar
    for j, val in enumerate(sensitivity_change_plus50[i]):
        axes[i].annotate(f'{val:.6f}%', xy=(val, j - bar_width/2), xytext=(5, 0), textcoords='offset points',
                         fontsize=8, color='black', ha='left', va='center')
        
    for j, val in enumerate(sensitivity_change_minus50[i]):
        axes[i].annotate(f'{val:.6f}%', xy=(val, j + bar_width/2), xytext=(5, 0), textcoords='offset points',
                         fontsize=8, color='black', ha='left', va='center')    
    
    # Add a vertical line at x=0
    axes[i].axvline(0, color='k', linestyle='--')
    
    # Set title and labels for each subplot
    axes[i].set_title(f'Sensitivity Change for {solutes_ssa[i]}')
    axes[i].legend(fontsize='x-small', loc='lower left') 
    axes[i].set_yticks(np.arange(4)) 
    axes[i].set_yticklabels(modes)

plt.suptitle('Change in Solute Clearance Due to 50% Fill Volume Variation', fontsize=16)
fig.supxlabel('Change in Clearance (%)')  # Common x-label for all subplots
fig.supylabel('Modes')
plt.tight_layout()
plt.show()
#%%
