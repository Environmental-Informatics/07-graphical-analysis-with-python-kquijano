# -*- coding: utf-8 -*-
"""
Karoll Quijano - kquijano

ABE 651: Environmental Informatics

Assignment 07
Graphing Data with Python

"""


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import probplot


### 1. Open and read the contents of the file into Python

df = pd.read_csv("all_month.csv", sep=',', index_col=0, parse_dates=True)
# genfromtxt() will not work with this data file becasue it has less inferring skills than Pandas 
#  FutureWarning: read_table is deprecated, use read_csv instead, passing sep='\t'.

df.dtypes

# Removing two rows with NaN 
df_2 = df[df['mag'].notna()]



### 2. Using matplotlib conduct graphical analysis of the data

# Generate a histogram of earthquake magnitude, using a bin width of 1 and a range of 0 to 10. 
plt.hist(df_2['mag'], bins=(10), width= (1), range= (0,10), density=True)
plt.title("Earthquake Magnitude Histogram (bins= 10)")
plt.xlabel("Magnitude")
plt.ylabel("Density")
plt.savefig("01_histogram_magnitude_bins_10.png")
plt.show()
plt.close()


# Generate a KDE plot. Selections for the kernel type and kernel width
density = gaussian_kde(df_2['mag'])
xs = np.linspace(0,10,200)
density.covariance_factor = lambda : 0.25
density._compute_covariance()
plt.plot(xs,density(xs))
plt.title('Kernel Density Estimation Plot of Earthquake Magnitude')
plt.ylabel('Density')
plt.xlabel('Magnitude')
plt.savefig('02_kde_magnitude.png')
plt.show()
plt.close()

  
# Plot latitude versus longitude for all earthquakes
plt.scatter(df_2['longitude'], df_2['latitude'], s=1)
plt.title('Lat-Long Earthquake Distribution')
plt.ylabel('Latitude (degree)')
plt.xlabel('Longitude (degree)')
plt.savefig('03_earthquakes_distribution.png')
plt.show()
plt.close()


# Generate a normalized cumulative distribution plot of earthquake depths. 
plt.plot(np.sort(df_2['depth']),np.linspace(0,1,len(df_2['depth'])))
plt.title('Normalized Cumulative Distribution Plot for Earthquake Depths')
plt.ylabel('Cumulative density')
plt.xlabel('Earthquake depth (km)')
plt.savefig('04_cumulative_density_earthquakes_depth.png')
plt.show()
plt.close()


# Generate a scatter plot of earthquake magnitude (x-axis) with depth (y-axis). 
plt.scatter(df_2['mag'], df_2['depth'], s=1)
plt.title('Earthquake Magnitude Vs. Depth')
plt.ylabel('Earthquake depth (km)')
plt.xlabel('Earthquake magnitude')
plt.savefig('05_distribution_magnitude_depth.png')
plt.show()
plt.close()


# Generate a quantile or Q-Q plot of the earthquake magnitudes. 
fig = plt.figure()
ax = fig.add_subplot(111)
probplot(df_2['mag'], dist="norm", plot = plt)
ax.get_lines()[0].set_markersize(3.0)
plt.title('Q-Q Plot for Earthquake Magnitude')
plt.ylabel('Earthquake Magnitude')
plt.xlabel('Theoretical quantiles')
plt.savefig('06_qq_plot.png')
plt.show()
plt.close()