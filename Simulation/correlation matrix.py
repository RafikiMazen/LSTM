from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt

sns.set(style="white")

# Generate a large random dataset
rs = np.random.RandomState(33)
d = pd.read_csv("D:\Bachelor\Python works\Fed-Poll\clustered_sensors/1/0.csv")
# d = pd.read_csv("D:\Bachelor\Datasets/Sensor_readings__with_temperature__light__humidity_every_5_minutes_at_8_locations__trial__2014_to_2015_.csv")
# Compute the correlation matrix
d['timestamp'] = pd.to_datetime(d['timestamp'])
d['timestamp']=pd.to_numeric(d.timestamp)/10**11
# corr = d[['timestamp','ozone','sulfure_dioxide','nitrogen_dioxide','particullate_matter','carbon_monoxide']].corr()
corr = d[['timestamp','temp_max','temp_min','temp_avg','light_max','light_min','light_avg','humidity_min','humidity_max','humidity_avg']].corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))
#
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})