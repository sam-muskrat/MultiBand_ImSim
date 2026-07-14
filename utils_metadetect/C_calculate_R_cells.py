# -*- coding: utf-8 -*-
# @Author: sam
# @Date:
# @Last Modified by:
# @Last Modified time:

### Calculate the mean response and residual biases for the whole sample

import os
import glob

import numpy as np
import pandas as pd

## ++++++++++++++ I/O and general setups

## path to test directory
test_dir = '/Users/samcritchley/Coding/LSST/output/samtest35'
out_file = 'R_per_cell.feather'
output_file = os.path.join(test_dir, out_file)

## What is the fitting model used in metadetect
fit_model = 'wmom'
flux_col = f'{fit_model}_band_flux'

## Zero point used only to report an approximate magnitude per cell (flux -> mag)
## Matches the dmag=30 flux convention used elsewhere in this pipeline.
ZERO_POINT = 30.0

## ++++++++++++++ Workhorse

## Get catalog (one)
inpath_list = glob.glob(os.path.join(test_dir,
                                'catalogues/shapes',
                                '*.feather'))
if len(inpath_list) != 1:
    print('Error: too many catalogs found. Aborting...')
    quit()

cata = pd.read_feather(inpath_list[0])[
    ['shear_type',
     f'{fit_model}_s2n',
     f'{fit_model}_g_1',
     f'{fit_model}_g_2',
     f'{fit_model}_T',
     f'{fit_model}_T_ratio',
     flux_col,
     'cell_id']]

cells = cata['cell_id'].max() + 1
R = []
invalid_cells = 0
multi_object_cells = 0

for idx in range(cells):
    cell = cata[cata['cell_id'] == idx]

    # Skip if no data in this cell
    if len(cell) == 0:
        continue

    noshear = cell[cell['shear_type']=='noshear']
    n_objects = len(noshear)

    if n_objects > 1:
        multi_object_cells += 1
        print(f">>> Warning: cell {idx} has {n_objects} 'noshear' rows "
              f"(expected exactly 1 star per cell)")

    g1_1p_s = cell.loc[cell['shear_type']=='1p', f'{fit_model}_g_1']
    g1_1m_s = cell.loc[cell['shear_type']=='1m', f'{fit_model}_g_1']
    g2_2p_s = cell.loc[cell['shear_type']=='2p', f'{fit_model}_g_2']
    g2_2m_s = cell.loc[cell['shear_type']=='2m', f'{fit_model}_g_2']

    # Skip cells missing any of the four shear types (or noshear, needed for
    # mag_avg) - an empty-slice .mean() returns NaN silently (no exception),
    # which would otherwise poison R/mag_avg.
    if min(len(g1_1p_s), len(g1_1m_s), len(g2_2p_s), len(g2_2m_s), n_objects) == 0:
        invalid_cells += 1
        continue

    try:
        # Calculate shear response for each cell
        R11 = (g1_1p_s.mean() - g1_1m_s.mean()) / 0.02
        R22 = (g2_2p_s.mean() - g2_2m_s.mean()) / 0.02

        mag_avg = (ZERO_POINT - 2.5 * np.log10(noshear[flux_col])).mean()

        resp = {
            'cell_id': idx,
            'n_objects': n_objects,
            'mag_avg': mag_avg,
            'R11': R11,
            'R22': R22,
            'R': (R11 + R22) / 2
        }
        R.append(resp)

    except (ZeroDivisionError, ValueError):
        # Skip cells with empty shear types
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
print(f"Cells with more than 1 'noshear' row: {multi_object_cells}")
print(f"\nObject count statistics:")
print(f"  Mean objects per cell: {R_df['n_objects'].mean():.1f}")
print(f"  Median objects per cell: {R_df['n_objects'].median():.1f}")
print(f"  Min objects per cell: {R_df['n_objects'].min():.0f}")
print(f"  Max objects per cell: {R_df['n_objects'].max():.0f}")
print(f"  Total objects across all cells: {R_df['n_objects'].sum():.0f}")
