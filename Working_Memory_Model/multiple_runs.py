#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 23:36:01 2020

@author: vivekchari
"""
from model_Network import working_memory_model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

days = [0,20, 40, 60]
cumulative_primary_dataframe = pd.DataFrame()
cumulative_firing_dataframe = pd.DataFrame()
cumulative_accuracy_dataframe = pd.DataFrame()
for day in days:
    transmission = (1 - (.01 *day))
    n = working_memory_model(transmission = transmission, proportion = 1, day = day)
    primary_dataframe, firing_dataframe, df_correct = n.go()
    print 'Appending to dataframe...'
    cumulative_primary_dataframe = cumulative_primary_dataframe.append(primary_dataframe, ignore_index=True)
    
    cumulative_firing_dataframe = cumulative_firing_dataframe.append(firing_dataframe, ignore_index=True)
    
    cumulative_accuracy_dataframe = cumulative_accuracy_dataframe.append(df_correct, ignore_index=True)
    
    print cumulative_primary_dataframe.shape
    del n

def plot(cumulative_primary_dataframe, cumulative_firing_dataframe,cumulative_accuracy_dataframe):
    print 'Plotting Data...'
    sns.set(context = 'paper')
    a = sns.tsplot(time = 'time', value = 'wm', data = cumulative_primary_dataframe, unit = 'trial', ci = 95, condition = 'day')
    a.set(xlabel='time (s)',ylabel='Decoded $\hat{cue}$ value')
    a.set(ylim = (0,1))
    plt.show(a)
    print 'Plotting Data...'
    figure2, (ax3, ax4) = plt.subplots(1, 2)
    if len(cumulative_firing_dataframe.query("tuning=='strong'")) > 0:
        sns.tsplot(condition = 'day', time="time",value="firing_rate",unit="neuron-trial",ax=ax3,ci=95,data = cumulative_firing_dataframe.query("tuning=='strong'").reset_index())
    
    if len(cumulative_firing_dataframe.query("tuning=='nonpreferred'"))>0:
        sns.tsplot(time="time",value="firing_rate",unit="neuron-trial", ax=ax4,ci=95,data=cumulative_firing_dataframe.query("tuning=='nonpreferred'").reset_index(), condition = 'day')
    
    ax3.set(xlabel='time (s)',ylabel='Normalized Firing Rate',title='Preferred Direction', ylim = (0,350))
    ax4.set(xlabel='time (s)',ylim=(0,250),ylabel='',title='Nonpreferred Direction')
    plt.show(figure2)
    
    
    c = sns.tsplot(time='time', value='correct', unit='trial', condition='day',data = cumulative_accuracy_dataframe, ci=95)
    c.set(xlabel = 'time(s)', ylabel = 'DRT accuracy percentage')
    c.set(ylim = (.4,1.1))
    plt.show(c) 
 
plot(cumulative_primary_dataframe, cumulative_firing_dataframe,cumulative_accuracy_dataframe)
