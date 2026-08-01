
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data

input_file = '../../output/samtest33/R_per_cell.feather'
R_df = pd.read_feather(input_file)

print(f"Loaded {len(R_df)} cell responses")
print(f"\nResponse statistics:")
print(f"  Mean R: {R_df['R'].mean():.6f}")
print(f"  Std R:  {R_df['R'].std():.6f}")
print(f"  Min R:  {R_df['R'].min():.6f}")
print(f"  Max R:  {R_df['R'].max():.6f}")

# Create histogram
plt.figure(figsize=(10, 6))

# Plot histogram
n, bins, patches = plt.hist(R_df['R'], bins=50, edgecolor='black', alpha=0.7)

# Add vertical line at mean
mean_R = R_df['R'].mean()
plt.axvline(mean_R, color='red', linestyle='--', linewidth=2, 
            label=f'Mean = {mean_R:.6f}')

# Add vertical line at zero
plt.axvline(0, color='green', linestyle='--', linewidth=1, 
            label='R = 0 (expected for stars)')

# Labels and title
plt.xlabel('Shear Response (R)', fontsize=12)
plt.ylabel('Number of Cells', fontsize=12)
plt.title('Distribution of Shear Response per Cell', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Save figure
output_plot = 'shear_response_histogram.png'
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\nSaved histogram to {output_plot}")

# Also show it
plt.show()

# Optional: Print outliers
threshold = 3 * R_df['R'].std()
outliers = R_df[np.abs(R_df['R'] - mean_R) > threshold]
if len(outliers) > 0:
    print(f"\nFound {len(outliers)} outlier cells (>3σ from mean):")
    print(outliers)
