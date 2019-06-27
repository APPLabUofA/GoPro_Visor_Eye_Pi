import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)


# Initialize variables and plot
par_num = 5
# Load in participant dataframes {001,003,004,005}
for par in range(1,par_num+2):
    if par != 2:
        if par > 2:
            globals()["df_" + str(par-1)] = pd.read_csv((r'M:\Data\GoPro_Visor\Experiment_1\Video_Times\Dataframe_df3_final_00' + str(par)) + '.csv', sep=',') # pilot

        else:
            globals()["df_" + str(par)] = pd.read_csv((r'M:\Data\GoPro_Visor\Experiment_1\Video_Times\Dataframe_df3_final_00' + str(par)) + '.csv', sep=',') # pilot

#par = 4
#        # Add each dataframe to plot
#for ax, par, name in zip(axes.flatten(), globals()["df_" + str(par)], par):
#    if par != 2:
##        ax = fig.add_subplot(2, 3, par)
#        sns.distplot((globals()["df_" + str(par)][11,:],globals()["df_" + str(par)]['index']), hist=False) #label='EEG - Pi Event'
#        ax.set(title = 1, xlabel='ms')
#plt.show()    


# %% Raw Difference -  trial count
fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=2)

left   =  0.125  # the left side of the subplots of the figure
right  =  0.9    # the right side of the subplots of the figure
bottom =  0.1    # the bottom of the subplots of the figure
top    =  0.85    # the top of the subplots of the figure
wspace =  .5     # the amount of width reserved for blank space between subplots
hspace = 0.7    # the amount of height reserved for white space between subplots

# This function actually adjusts the sub plots using the above paramters
plt.subplots_adjust(
    left    =  left, 
    bottom  =  bottom, 
    right   =  right, 
    top     =  top, 
    wspace  =  wspace, 
    hspace  =  hspace
)
# The amount of space above titles
y_title_margin = 1 
fig.suptitle("Raw Difference - Trial Count vs Difference", fontsize=20)

# In statistics, kernel density estimation (KDE) is a non-parametric way to estimate the probability density function (PDF) of a random variable.  # norm_hist=False, kde=False
#In the following plots density refers to kernel density estimation (KDE) a non-parametric estimation of the probability density function
### 001,003
ax[0][0].set_title("001",                    y = y_title_margin)
sns.distplot(df_1['Difference']*1000, bins = 12, kde=False, rug = True, rug_kws={'color': 'black'}, ax=ax[0][0])
ax[0][1].set_title("003",         y = y_title_margin)
sns.distplot(df_2['Difference']*1000, bins = 12, kde=False, rug = True, rug_kws={'color': 'black'}, ax=ax[0][1])

[ax[0][i].set_xlabel("Time (ms)") for i in range(0, 2)]
[ax[0][i].set_ylabel("Trial Count") for i in range(0, 2)]


# 004,005
ax[1][0].set_title("004",                  y = y_title_margin)
sns.distplot(df_3['Difference']*1000, bins = 12, kde=False, rug = True, rug_kws={'color': 'black'}, ax=ax[1][0])
ax[1][1].set_title("005",       y = y_title_margin)
sns.distplot(df_4['Difference']*1000, bins = 12, kde=False, rug = True, rug_kws={'color': 'black'}, ax=ax[1][1])
#ax[1][1].set_xlim([-60,5])

[ax[1][i].set_xlabel("Time (ms)") for i in range(0, 2)]
[ax[1][i].set_ylabel("Trial Count") for i in range(0, 2)]

# %% Raw Difference -  trial density
fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=2)

left   =  0.125  # the left side of the subplots of the figure
right  =  0.9    # the right side of the subplots of the figure
bottom =  0.1    # the bottom of the subplots of the figure
top    =  0.85    # the top of the subplots of the figure
wspace =  .5     # the amount of width reserved for blank space between subplots
hspace =  0.7    # the amount of height reserved for white space between subplots

# This function actually adjusts the sub plots using the above paramters
plt.subplots_adjust(
    left    =  left, 
    bottom  =  bottom, 
    right   =  right, 
    top     =  top, 
    wspace  =  wspace, 
    hspace  =  hspace
)
# The amount of space above titles
y_title_margin = 1 
fig.suptitle("Raw Difference - Trial Density vs Difference", fontsize=20)

# In statistics, kernel density estimation (KDE) is a non-parametric way to estimate the probability density function (PDF) of a random variable.  # norm_hist=False, kde=False

### 001,003
ax[0][0].set_title("001",                    y = y_title_margin)
sns.distplot(df_1['Difference']*1000, bins = 12, rug = True, rug_kws={'color': 'black'}, ax=ax[0][0])
ax[0][1].set_title("003",         y = y_title_margin)
sns.distplot(df_2['Difference']*1000, bins = 12, rug = True, rug_kws={'color': 'black'}, ax=ax[0][1])

[ax[0][i].set_xlabel("Time (ms)") for i in range(0, 2)]
[ax[0][i].set_ylabel("Density") for i in range(0, 2)]


# 004,005
ax[1][0].set_title("004",                  y = y_title_margin)
sns.distplot(df_3['Difference']*1000, bins = 12, rug = True, rug_kws={'color': 'black'}, ax=ax[1][0])
ax[1][1].set_title("005",       y = y_title_margin)
sns.distplot(df_4['Difference']*1000, bins = 12, rug = True, rug_kws={'color': 'black'}, ax=ax[1][1])
#ax[1][1].set_xlim([-60,5])

[ax[1][i].set_xlabel("Time (ms)") for i in range(0, 2)]
[ax[1][i].set_ylabel("Density") for i in range(0, 2)]


# %% AP Transformed Difference - trial count

fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=2)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')


left   =  0.125  # the left side of the subplots of the figure
right  =  0.9    # the right side of the subplots of the figure
bottom =  0.1    # the bottom of the subplots of the figure
top    =  0.85    # the top of the subplots of the figure
wspace =  .5     # the amount of width reserved for blank space between subplots
hspace =  0.7    # the amount of height reserved for white space between subplots

# This function actually adjusts the sub plots using the above paramters
plt.subplots_adjust(
    left    =  left, 
    bottom  =  bottom, 
    right   =  right, 
    top     =  top, 
    wspace  =  wspace, 
    hspace  =  hspace
)

# The amount of space above titles
y_title_margin = 1
fig.suptitle("All Point Transformed Difference - Trial Count vs. Difference", fontsize=20)

### Bathrooms
ax[0][0].set_title("001",                    y = y_title_margin)
sns.distplot(df_1['AP_Transform']*1000,  bins = 12, kde=False, rug = True, rug_kws={'color': 'black'}, ax=ax[0][0])

ax[0][1].set_title("003",         y = y_title_margin)
sns.distplot(df_2['AP_Transform']*1000,  bins = 12, kde=False, rug = True, rug_kws={'color': 'black'}, ax=ax[0][1])

[ax[0][i].set_xlabel("Time (ms)") for i in range(0, 2)]
[ax[0][i].set_ylabel("Trial Count") for i in range(0, 2)]

### Square feet
ax[1][0].set_title("004",                  y = y_title_margin)
sns.distplot(df_3['AP_Transform']*1000, bins = 12, kde=False, rug = True, rug_kws={'color': 'black'}, ax=ax[1][0])

ax[1][1].set_title("005",       y = y_title_margin)
sns.distplot(df_4['AP_Transform']*1000, bins = 12, kde=False, rug = True, rug_kws={'color': 'black'}, ax=ax[1][1])
ax[1][1].set_xlim([-6,6])
#ax[1][1].text(0.5, 0.5, '\TeX\ hello world: $\int_0^\infty e^x dx$', size=20, ha='center', va='center')


[ax[1][i].set_xlabel("Time (ms)") for i in range(0, 2)]
[ax[1][i].set_ylabel("Trial Count") for i in range(0, 2)]


# %% AP Transformed Difference - trial density
fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=2)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')


left   =  0.125  # the left side of the subplots of the figure
right  =  0.9    # the right side of the subplots of the figure
bottom =  0.1    # the bottom of the subplots of the figure
top    =  0.85    # the top of the subplots of the figure
wspace =  .5     # the amount of width reserved for blank space between subplots
hspace =  0.7    # the amount of height reserved for white space between subplots

# This function actually adjusts the sub plots using the above paramters
plt.subplots_adjust(
    left    =  left, 
    bottom  =  bottom, 
    right   =  right, 
    top     =  top, 
    wspace  =  wspace, 
    hspace  =  hspace
)

# The amount of space above titles
y_title_margin = 1
fig.suptitle("All Point Transformed Difference - Trial Density vs Difference", fontsize=20)

### Bathrooms
ax[0][0].set_title("001",                    y = y_title_margin)
sns.distplot(df_1['AP_Transform']*1000,  bins = 12, rug = True, rug_kws={'color': 'black'}, ax=ax[0][0])

ax[0][1].set_title("003",         y = y_title_margin)
sns.distplot(df_2['AP_Transform']*1000,  bins = 12, rug = True, rug_kws={'color': 'black'}, ax=ax[0][1])

[ax[0][i].set_xlabel("Time (ms)") for i in range(0, 2)]
[ax[0][i].set_ylabel("Density") for i in range(0, 2)]

### Square feet
ax[1][0].set_title("004",                  y = y_title_margin)
sns.distplot(df_3['AP_Transform']*1000, bins = 12, rug = True, rug_kws={'color': 'black'}, ax=ax[1][0])

ax[1][1].set_title("005",       y = y_title_margin)
sns.distplot(df_4['AP_Transform']*1000, bins = 12, rug = True, rug_kws={'color': 'black'}, ax=ax[1][1])
ax[1][1].set_xlim([-6,6])
#ax[1][1].text(0.5, 0.5, '\TeX\ hello world: $\int_0^\infty e^x dx$', size=20, ha='center', va='center')


[ax[1][i].set_xlabel("Time (ms)") for i in range(0, 2)]
[ax[1][i].set_ylabel("Density") for i in range(0, 2)]



#%% Tail Transform

#fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=2)
##plt.rc('text', usetex=True)
##plt.rc('font', family='serif')
#
#
#left   =  0.125  # the left side of the subplots of the figure
#right  =  0.9    # the right side of the subplots of the figure
#bottom =  0.1    # the bottom of the subplots of the figure
#top    =  0.9    # the top of the subplots of the figure
#wspace =  .5     # the amount of width reserved for blank space between subplots
#hspace =  1.1    # the amount of height reserved for white space between subplots
#
## This function actually adjusts the sub plots using the above paramters
#plt.subplots_adjust(
#    left    =  left, 
#    bottom  =  bottom, 
#    right   =  right, 
#    top     =  top, 
#    wspace  =  wspace, 
#    hspace  =  hspace
#)
#
## The amount of space above titles
#y_title_margin = 1
#fig.suptitle("All Point Transformed Difference", fontsize=20)
#
#### Bathrooms
#ax[0][0].set_title("001",                    y = y_title_margin)
#sns.distplot(df_1['AP_Transform']*1000,       rug = True, rug_kws={'color': 'black'}, ax=ax[0][0])
#
#ax[0][1].set_title("003",         y = y_title_margin)
#sns.distplot(df_2['AP_Transform']*1000,   rug = True, rug_kws={'color': 'black'}, ax=ax[0][1])
#
#[ax[0][i].set_xlabel("ms") for i in range(0, 2)]
#[ax[0][i].set_ylabel("Trial Count") for i in range(0, 2)]
#
#### Square feet
#ax[1][0].set_title("004",                  y = y_title_margin)
#sns.distplot(df_3['AP_Transform']*1000, rug = True, rug_kws={'color': 'black'}, ax=ax[1][0])
#
#ax[1][1].set_title("005",       y = y_title_margin)
#sns.distplot(df_4['AP_Transform']*1000, bins=8, rug = True, rug_kws={'color': 'black'}, ax=ax[1][1])
#ax[1][1].set_xlim([-6,6])
##ax[1][1].text(0.5, 0.5, '\TeX\ hello world: $\int_0^\infty e^x dx$', size=20, ha='center', va='center')
#
#
#[ax[1][i].set_xlabel("ms") for i in range(0, 2)]
#[ax[1][i].set_ylabel("Trial Count") for i in range(0, 2)]