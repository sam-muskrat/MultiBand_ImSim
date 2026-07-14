import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, laplace

# Load the data
input_file = '../../output/samtest35/R_per_cell.feather'
R_df = pd.read_feather(input_file)

print(f"Loaded {len(R_df)} cell responses")
print(f"\nResponse statistics:")
print(f"  Mean R: {R_df['R'].mean():.6f}")
print(f"  Std R:  {R_df['R'].std():.6f}")
print(f"  Min R:  {R_df['R'].min():.6f}")
print(f"  Max R:  {R_df['R'].max():.6f}")

# Create histogram
plt.figure(figsize=(10, 6))

# Plot histogram with density=True for proper scaling with Gaussian
n, bins, patches = plt.hist(R_df['R'], bins=75, edgecolor='black', alpha=0.7, 
                            weights=R_df['n_objects'], label='Observed')

# Calculate mean and std
mean_R = R_df['R'].mean()
std_R = R_df['R'].std()

# Generate Gaussian curve
#x = np.linspace(R_df['R'].min(), R_df['R'].max(), 100)
#gaussian = norm.pdf(x, mean_R, std_R) * sum(R_df['n_objects']) * (bins[1] - bins[0])
#plt.plot(x, gaussian, 'r-', linewidth=2, label=f'Gaussian fit\n(μ={mean_R:.6f}, σ={std_R:.6f})')

# Add Laplace distribution
#laplace_scale = std_R / np.sqrt(2)  # Convert std to Laplace scale parameter
#laplace_dist = laplace.pdf(x, loc=mean_R, scale=laplace_scale) * sum(R_df['n_objects']) * (bins[1] - bins[0])
#plt.plot(x, laplace_dist, 'b-', linewidth=2, label=f'Laplace fit\n(μ={mean_R:.6f}, b={laplace_scale:.6f})')


# Add vertical line at mean
plt.axvline(mean_R, color='red', linestyle='--', linewidth=2, alpha=0.5,
            label=f'Mean = {mean_R:.6f}')

# Add vertical line at zero
plt.axvline(0, color='green', linestyle='--', linewidth=1,
            label='R = 0 (expected for stars)')

# Labels and title
plt.xlabel('Shear Response (R)', fontsize=12)
plt.ylabel('Number of objects', fontsize=12)
plt.title('Stellar Shear Response using METADETECTION', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Save figure
output_plot = 'shear_response_histogram.png'
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\nSaved histogram to {output_plot}")

# Also show it
plt.show()

# Optional: Print outliers
#threshold = 3 * std_R
#outliers = R_df[np.abs(R_df['R'] - mean_R) > threshold]
#if len(outliers) > 0:
#    print(f"\nFound {len(outliers)} outlier cells (>3σ from mean):")
#    print(outliers)


    # Calculate kurtosis to quantify "peakedness"
#from scipy.stats import kurtosis
#kurt = kurtosis(R_df['R'])
#print(f"\nKurtosis: {kurt:.3f}")
#print("  > 0: Leptokurtic (more peaked than Gaussian)")
#print("  = 0: Mesokurtic (same as Gaussian)")
#print("  < 0: Platykurtic (flatter than Gaussian)")
