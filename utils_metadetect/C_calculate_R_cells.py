# -*- coding: utf-8 -*-
# @Author: sam
# @Date:   
# @Last Modified by:
# @Last Modified time: 

### Calculate the mean response and residual biases for the whole sample 

import os
import re
import sys
import glob

import numpy as np 
import pandas as pd
from scipy.optimize import curve_fit

## ++++++++++++++ I/O and general setups

# Gather runtag arg
if len(sys.argv) < 2:
    print("Error: No runTag provided!")
    print("Usage: python C_calculate_R_cells.py <runtag>")
    sys.exit(1)

RUNTAG = sys.argv[1] # corresponds to folder name

## path to test directory
test_dir = '/scratch/users/samjc/output'
out_file = 'R_per_cell.feather'
output_file = os.path.join(test_dir, RUNTAG, out_file)

## Shear types used in metadetect
shear_types = ['noshear', '1p', '1m', '2p', '2m']

## What is the fitting model used in metadetect
fit_model = 'wmom'

## Which shear weight to use
#### None = No weighting
# which_weight = None
#### weight_sigma_e = weight based on sigma_e
# which_weight = 'weight_sigma_e'
#### weight_e = weight based on e
# which_weight = 'weight_e'
#### shear_weight = weight_sigma_e * weight_e
which_weight = None # stars need no confidence weighting

## ++++++++++++++ Workhorse

## Get catalog (one)
cata = []

inpath_list = glob.glob(os.path.join(test_dir, RUNTAG,
                                'catalogues/shapes', 
                                '*.feather'))
print(f">>> Number of catalogues found in {test_dir}/{RUNTAG}: {len(inpath_list)} (should be one)")
if len(inpath_list) < 1:
    print('Error: no catalogs found. Aborting...')
    quit()


max_id = 0

for i in range(len(inpath_list)):
    inpath = inpath_list[i]


    cata_tmp = pd.read_feather(inpath)
    ## Drop zero-weight objects and useless columns for memory
    if which_weight is not None:
        cata_tmp = cata_tmp.loc[cata_tmp[which_weight]>0,
                                ['shear_type', 
                                 f'{fit_model}_s2n', 
                                 f'{fit_model}_g_1',
                                 f'{fit_model}_g_2', 
                                 f'{fit_model}_T', 
                                 f'{fit_model}_T_ratio',
                                 f'{fit_model}_band_flux',
                                 'cell_id',
                                 which_weight]]
        ## Renaming for easy use
        cata_tmp = cata_tmp.rename(columns={which_weight: 
                                            'weight'})
    else:
        cata_tmp = cata_tmp[ 
            ['shear_type', 
             f'{fit_model}_s2n', 
             f'{fit_model}_g_1',
             f'{fit_model}_g_2', 
             f'{fit_model}_T', 
             f'{fit_model}_T_ratio',
             f'{fit_model}_band_flux',
             'cell_id']]
        ## No weighting
        cata_tmp['weight'] = 1
        
    
    cata_tmp['cell_id'] += max_id #ensure unique cell id with multiple tiles

    max_id = cata_tmp['cell_id'].max() + 1
        
    cata.append(cata_tmp)
    del cata_tmp

cata = pd.concat(cata, ignore_index=True)


#cata = cata[cata[f'{fit_model}_T_ratio'] < 1.1]

#cata['weight'] = 1  # Override weights - use uniform weighting for stars

## Calculat shear response and residual shear bias for the whole sample
#g1_input_all = []
#g2_input_all = []
#g1_measured_all = []
#g2_measured_all = []

cells = cata['cell_id'].max() + 1
R = []
invalid_cells = 0

for idx in range(cells):
    cell = cata[cata['cell_id'] == idx]
    
    # Skip if no data in this cell
    if len(cell) == 0:
        invalid_cells += 1
        continue
        
    # Calculate approximate number of objects (total entries / 5 shear types)
    n_objects = len(cell) / 5

    try:
        # Calculate shear response for each cell
        g1_1p = np.average(cell.loc[cell['shear_type']=='1p', f'{fit_model}_g_1'],
                           weights=cell.loc[cell['shear_type']=='1p', 'weight'])
        g1_1m = np.average(cell.loc[cell['shear_type']=='1m', f'{fit_model}_g_1'],
                           weights=cell.loc[cell['shear_type']=='1m', 'weight'])
        R11 = (g1_1p - g1_1m) / 0.02
        
        g2_2p = np.average(cell.loc[cell['shear_type']=='2p', f'{fit_model}_g_2'],
                           weights=cell.loc[cell['shear_type']=='2p', 'weight'])
        g2_2m = np.average(cell.loc[cell['shear_type']=='2m', f'{fit_model}_g_2'],
                           weights=cell.loc[cell['shear_type']=='2m', 'weight'])
        R22 = (g2_2p - g2_2m) / 0.02
        
        resp = {
            'cell_id': idx,
            'n_objects': n_objects,
            'R11': R11,
            'R22': R22,
            'R': (R11 + R22) / 2
        }
        R.append(resp)
        
    except (ZeroDivisionError, ValueError):
        # Skip cells with zero weights or empty shear types
        invalid_cells += 1
        continue

    
# Convert responses to DataFrame
R_df = pd.DataFrame(R)

# Save to feather file
R_df.to_feather(output_file)
print(f"Saved {len(R_df)} cell responses to {output_file}")
print(f"\nResponse statistics:")
print(f"  Mean R: {R_df['R'].mean():.6f}")
print(f"  Std R:  {R_df['R'].std():.6f}")
print(f"  Min R:  {R_df['R'].min():.6f}")
print(f"  Max R:  {R_df['R'].max():.6f}")
print(f"Cells ommitted due to insufficient data: {invalid_cells}")
print(f"\nObject count statistics:")
print(f"  Mean objects per cell: {R_df['n_objects'].mean():.1f}")
print(f"  Median objects per cell: {R_df['n_objects'].median():.1f}")
print(f"  Min objects per cell: {R_df['n_objects'].min():.0f}")
print(f"  Max objects per cell: {R_df['n_objects'].max():.0f}")
print(f"  Total objects across all cells: {R_df['n_objects'].sum():.0f}")
