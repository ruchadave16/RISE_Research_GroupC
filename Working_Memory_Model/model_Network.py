#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 21:54:34 2020

@author: vivekchari
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from helper import make_cues, make_addon, ch_dir, empirical_dataframe, MySolver, reset_gain_bias, primary_dataframe, firing_dataframe, get_correct
from pathos.multiprocessing import ProcessingPool as Pool
import nengo
from nengo.rc import rc
from nengo.dists import Choice, Exponential, Uniform
from pathos.helpers import freeze_support 
from nengo.utils.matplotlib import rasterplot
#from nengo_extras import CollapsingGexfConverter
from nengo_extras.plot_spikes import (cluster, merge, plot_spikes, preprocess_spikes, sample_by_variance)
from Ensemble_Collection import EnsembleCollection
nengo.rc.set('progress', 'progress_bar', 'nengo.utils.progress.TerminalProgressBar')
import scipy.stats as ss

class working_memory_model():
    '''
    Proportion: The percentage of neurons in the working memory network affected by the synaptic degradation(Default = 0)
    
    Degradation: The degradation that should be appled to the synaptic weights (Default = 0)
    '''
    def __init__(self, proportion = 0, degradation = 0):
        
        print 'Importing model parameters from parameters.txt'
        self.P = eval(open('parameters.txt').read())
        self.seed = self.P['seed']
        self.rng = np.random.RandomState(seed = self.seed)
        self.net_seed =  self.rng.randint(100)
        self.ens_seed =  self.rng.randint(100)
        self.con_seed =  self.rng.randint(100)
        self.sim_seed =  self.rng.randint(100)
        self.n_trials = self.P['n_trials']
        self.n_processes = self.P['n_processes']
        self.drug_type = str(self.P['drug_type'])
        self.decision_type=str(self.P['decision_type'])
        self.drugs=self.P['drugs']
        self.trials, self.perceived, self.cues = make_cues(self.P)
        self.P['timesteps'] = np.arange(0,int((self.P['t_cue']+ self.P['t_delay'])/self.P['dt_sample']))
        self.P['cues'] = self.cues
        self.P['perceived'] = self.perceived
        self.degradation = degradation
        self.proportion = proportion     
        self.spiking_array = []
        self.accuracy_data = []
        self.neuron_type = nengo.LIF()
        self.memory_threshhold = 0
        self.ground_zero = 500
        np.random.seed(self.ens_seed)


    def noise_bias_function(self,t):
		if self.drug_type=='neural':
			return np.random.normal(self.drug_effect_neural[self.drug],self.noise_wm)
		else:
			return np.random.normal(0.0,self.noise_wm)
        
    def noise_decision_function(self, t):
        if self.decision_type == 'default':
            return np.random.normal(0.0,self.noise_decision)
        elif self.decision_type == 'basal_ganglia':
			return np.random.normal(0.0, self.noise_decision,size = 2)
        
    def inputs_function(self,x):
        return x * self.tau_wm
    
    def cue_function(self, t):
        if t < self.t_cue and self.perceived[self.trial] != 0:
            self.cue = self.cue_scale * self.cues[self.trial]
            return self.cue_scale * self.cues[self.trial]
        else:
            self.cue = 0
            return 0
        
    def time_function(self,t):
        if t > self.t_cue:
            return self.time_scale
        else:
            return 0
        

    def wm_recurrent_function(self,x):
		if self.drug_type == 'functional':
			return x * self.drug_effect_functional[self.drug]
		else:
			return x

    def decision_function(self,x): 
        output=0.0
        if self.decision_type=='default':
            value = x[0] + x[1]
            if value > 0.0:
                output = 1.0
            elif value < 0.0:
                output = -1.0
        elif self.decision_type == 'basal_ganglia':
            if x[0] > x[1]:
                output = 1.0
            elif x[0] < x[1]:
                output = -1.0
        return output 

    def BG_rescale(self,x): #rescales -1 to 1 into 0.3 to 1, makes 2-dimensional
		pos_x = 0.5 * (x + 1)
		rescaled = 0.4 + 0.6 * pos_x, 0.4 + 0.6 * (1 - pos_x)
		return rescaled
    
    def f_dec(self,x):
        value = x[0]
        if value > self.memory_threshhold and self.cue > 0:
            return 1
        elif value < (-1 * self.memory_threshhold) and self.cue < 0:
           return 1
        return 0
    '''
    def stim_ensemble(self,ens, sim, bias = False):
        n_neurons = min(int(ens.n_neurons * self.stim_proportion), ens.n_neurons)
        idx = np.random.choice(np.arange(ens.n_neurons), replace=False, size=n_neurons)    
        if bias:
            bias_sig = sim.signals[sim.model.sig[ens.neurons]['bias']]
            bias_sig.setflags(write=True)
            bias_sig[idx] = bias_sig[idx] * self.stim_level'''

       
    def degrade_synaptic_connection(self, sim, ens):
        weights = sim.data[self.wm_recurrent].weights / sim.data[ens].gain[:, None]
        n_neurons = min(int(ens.n_neurons * self.proportion), ens.n_neurons)

        if self.proportion > .7:
            idx = np.random.choice(np.arange(ens.n_neurons), replace=False, size = n_neurons)
        else:
            x = np.arange(ens.n_neurons)
            xU, xL = x + (.75*n_neurons), x - (.75*n_neurons)
            #prob = ss.norm.cdf(xU, loc = self.ground_zero, scale = 3) - ss.norm.cdf(xL,loc = self.ground_zero, scale = 3)
            prob = ss.poisson.cdf(k = xU, mu = self.ground_zero) - ss.poisson.cdf(k = xL,mu = self.ground_zero)
            for i in prob:
                if round(i,3) == 0:
                    i = .001   
            prob = prob / prob.sum() #normalize the probabilities so their sum is 1
            idx = np.random.choice(x, size = n_neurons, p = prob, replace = False)
            if self.trial == 0:
                plt.hist(idx, bins = 1000)
                plt.xlim((0,1000))
                plt.xlabel('Neuron #')
                plt.ylabel('Frequency')
                plt.show()
        if self.trial == 0: print len(idx)    
        weights[idx] = weights[idx] * (1-self.degradation)
        return weights
    
    def run(self, params):
        self.decision_type = params[0]
        self.drug_type = params[1]
        self.drug = params[2]
        self.trial = params[3]
        self.seed = params[4]
        self.P = params[5]
        self.dt = self.P['dt']
        self.dt_sample = self.P['dt_sample']
        self.t_cue = self.P['t_cue']
        self.t_delay = self.P['t_delay']
        self.drug_effect_neural = self.P['drug_effect_neural']
        self.drug_effect_functional = self.P['drug_effect_functional']
        self.drug_effect_biophysical = self.P['drug_effect_biophysical']
        self.enc_min_cutoff = self.P['enc_min_cutoff']
        self.enc_max_cutoff = self.P['enc_max_cutoff']
        self.sigma_smoothing = self.P['sigma_smoothing']
        self.frac = self.P['frac']
        self.neurons_inputs = self.P['neurons_inputs']
        self.neurons_wm = self.P['neurons_wm']
        self.neurons_decide = self.P['neurons_decide']
        self.time_scale = self.P['time_scale']
        self.cue_scale = self.P['cue_scale']
        self.tau = self.P['tau']
        self.tau_wm = self.P['tau_wm']
        self.noise_wm = self.P['noise_wm']
        self.noise_decision = self.P['noise_decision']
        self.perceived = self.P['perceived']
        self.cues = self.P['cues']
        if self.drug_type == 'biophysical': 
            rc.set("decoder_cache", "enabled", "False") #don't try to remember old decoders
        else:
            rc.set("decoder_cache", "enabled", "True")
        
        if self.trial == 0:
            print '\nBuilding model...'
        with nengo.Network(seed = self.net_seed) as model:
            wm = nengo.Ensemble(self.neurons_wm, 2, neuron_type = self.neuron_type,seed = self.ens_seed, label = 'Working Memory')
            self.wm_recurrent = nengo.Connection(wm, wm, synapse = self.tau_wm, function = self.wm_recurrent_function, seed = self.con_seed, 
                                                 solver = nengo.solvers.LstsqL2(weights=True))
     
        with nengo.Simulator(model,dt = self.dt, seed = self.sim_seed) as sim:
            pass
        weights = self.degrade_synaptic_connection(sim, wm)
        
        with model:            
            if self.trial == 0: print '\nCreating Ensembles...'
            
            cue = nengo.Node(output = self.cue_function, label = 'Cue')
            time = nengo.Node(output = self.time_function, label = 'Time')
            inputs = nengo.Ensemble(self.neurons_inputs, 2, seed = self.ens_seed, label = 'Input Neurons')
            noise_wm_node = nengo.Node(output = self.noise_bias_function, label = 'Noise injection (WM node)')
            noise_decision_node = nengo.Node(output = self.noise_decision_function, label = 'Noise injection (decision node)')
            wm = nengo.Ensemble(self.neurons_wm, 2, neuron_type = self.neuron_type,seed = self.ens_seed, label = 'Working Memory')
            cor = nengo.Ensemble(1, 1, neuron_type = nengo.Direct(),seed = self.ens_seed, label = 'Accuracy sensor')
            
            if self.decision_type == 'default':
                decision = nengo.Ensemble(self.neurons_decide, 2, seed = self.ens_seed, label = 'Decision Maker')
                
            elif self.decision_type == 'basal_ganglia':
                utilities = nengo.networks.EnsembleArray(self.neurons_inputs, n_ensembles = 2, seed = self.ens_seed, label = 'Utility network')
                BasalGanglia = nengo.networks.BasalGanglia(2, self.neurons_decide)
                decision = nengo.networks.EnsembleArray(self.neurons_decide, n_ensembles = 2, intercepts = Uniform(0.2, 1), encoders = Uniform(1,1), seed = self.ens_seed, label = 'Decision ensemble (Basal Ganglia')
                temp = nengo.Ensemble(self.neurons_decide, 2,neuron_type = self.neuron_type, seed = self.ens_seed)
                bias = nengo.Node([1] * 2, label = 'bias node')
            output = nengo.Ensemble(self.neurons_decide, 1, neuron_type = self.neuron_type, seed = self.ens_seed, label = 'Output')
            
            if self.trial == 0: print '\nBuildiing Connections...'
            
            nengo.Connection(cue, inputs[0], synapse = None, seed = self.con_seed)
            nengo.Connection(time, inputs[1], synapse = None, seed = self.con_seed)
            nengo.Connection(inputs, wm, synapse = self.tau_wm, function=self.inputs_function, seed = self.con_seed)
            self.wm_recurrent = nengo.Connection(wm.neurons, wm.neurons, synapse = self.tau_wm, seed = self.con_seed, transform = weights)
            nengo.Connection(noise_wm_node, wm.neurons, synapse = self.tau_wm, transform = np.ones((self.neurons_wm,1))*self.tau_wm, seed = self.con_seed)
            if self.decision_type == 'default':
                wm_to_decision = nengo.Connection(wm[0], decision[0], synapse = self.tau, seed = self.con_seed)
                nengo.Connection(noise_decision_node, decision[1], synapse = None, seed = self.con_seed)
                nengo.Connection(decision, output,function = self.decision_function, seed = self.con_seed)
                nengo.Connection(decision, cor, synapse = self.tau,function = self.f_dec, seed = self.con_seed)

                
            elif self.decision_type == 'basal_ganglia':
                wm_to_decision = nengo.Connection(wm[0], utilities.input, synapse = self.tau, function = self.BG_rescale, seed = self.con_seed)
                nengo.Connection(BasalGanglia.output, decision.input, synapse = self.tau, seed = self.con_seed)
                nengo.Connection(noise_decision_node, BasalGanglia.input,synapse = None, seed = self.con_seed) #added external noise?
                nengo.Connection(bias, decision.input, synapse = self.tau, seed = self.con_seed)
                nengo.Connection(decision.input, decision.output, transform=(np.eye(2)-1), synapse = self.tau/2.0, seed = self.con_seed)
                nengo.Connection(decision.output,temp, seed = self.con_seed)
                nengo.Connection(temp,output,function = self.decision_function, seed = self.con_seed)
                nengo.Connection(temp, cor, synapse = self.tau,function = self.f_dec, seed = self.con_seed)
            
            if self.trial == 0: print 'Building Probes...'
            probe_wm = nengo.Probe(wm[0],synapse = 0.01, sample_every = self.dt_sample)
            probe_spikes = nengo.Probe(wm.neurons, 'spikes', sample_every = self.dt_sample)
            probe_output = nengo.Probe(output,synapse=None, sample_every = self.dt_sample)
            p_cor = nengo.Probe(cor, synapse=None, sample_every = self.dt_sample)
            #data_dir = ch_dir()
            #CollapsingGexfConverter().convert(model).write('model.gexf')                     


        print 'Running trial %s...' %(self.trial+1)
        with nengo.Simulator(model,dt = self.dt, seed = self.sim_seed) as sim:
            if self.drug_type == 'biophysical': 
                sim = reset_gain_bias(self.P, model, sim, wm, self.wm_recurrent, wm_to_decision, self.drug)
            sim.run(self.t_cue + self.t_delay)
            xyz = sim.data[probe_spikes]
            abc = np.abs(sim.data[p_cor])
            df_primary = primary_dataframe(self.P, sim, self.drug,self.trial, probe_wm, probe_output)
            df_firing = firing_dataframe(self.P,sim,self.drug,self.trial, sim.data[wm], probe_spikes)
            
        return [df_primary, df_firing, abc, xyz]
                
    def multiprocessing(self):
        
        print "decision_type=%s, trials=%s..." %(self.decision_type,self.n_trials)
        pool = Pool(nodes=self.n_processes)
        freeze_support()
        exp_params = []
        for drug in self.drugs:
            for trial in self.trials:
                exp_params.append([self.decision_type, self.drug_type, drug, trial, self.seed, self.P])
        #self.df_list = []
        self.df_list = pool.map(self.run, exp_params)
        #for i in range(self.n_trials):self.df_list.append(self.run(exp_params[i]))
        
        for i in range(len(self.df_list)):
            self.accuracy_data.append(self.df_list[i][2])
            
        self.accuracy_data = np.array(self.accuracy_data)
        
        for i in range(len(self.df_list)):
            self.spiking_array.append((self.df_list[i][3]))
        self.spiking_array = np.array(self.spiking_array)
        
        print 'Constructing Dataframes...'
        primary_dataframe = pd.concat([self.df_list[i][0] for i in range(len(self.df_list))], ignore_index=True)
        firing_dataframe = pd.concat([self.df_list[i][1] for i in range(len(self.df_list))], ignore_index=True)

        return primary_dataframe, firing_dataframe
    def export(self):
        print 'Exporting Data...'
        datadir = ch_dir()
        primary_dataframe.to_pickle('primary_data.pkl')
        firing_dataframe.to_pickle('firing_data.pkl')
        param_df = pd.DataFrame([self.P])
        param_df.reset_index().to_json('params.json',orient='records')

    def array_to_pandas(self,data, datatype, drug):
        n_trials = data.shape[0]
        n_timesteps = data.shape[1]
        columns = ('trial', 'drug', 'time', datatype)
        df = pd.DataFrame(columns=columns)
        for trial in range(n_trials):
            print 'Adding trial %s, to %s...' %(trial + 1, datatype)
            df_time = []
            for t in range(n_timesteps):
                df_temp = pd.DataFrame(
                    [[trial, drug + ' (model)', t*0.01, data[trial][t][0]]], columns=columns)
                df_time.append(df_temp)
                del df_temp
            df_trial = pd.concat(df_time, ignore_index=True)
            df = pd.concat([df, df_trial], ignore_index=True)
            del df_time
        return df

        
    def plot_data(self, primary_dataframe, firing_dataframe, ):
        print 'Plotting Data...'
        sns.set(context = 'paper')
        a = sns.tsplot(time = 'time', value = 'wm', data = primary_dataframe, unit = 'trial', ci = 95)
        a.set(xlabel='time (s)',ylabel='Decoded $\hat{cue}$ value',
              title= "Number of trials = %s Prop degraded = %s Degradation = %s" %(self.n_trials, self.proportion,self.degradation))
        a.set(ylim = (0,1))
        plt.show(a)

        '''
        print 'Plotting Data...'
        spikes = self.spiking_array[0]
        spike_sum = np.sum(spikes,axis=0)
        indices = np.argsort(spike_sum)[::-1]
        top_spikes=spikes[:,indices[0:50]]
        
        b = rasterplot(self.spiking_array[1], top_spikes, use_eventplot = True)
        b = plt.gca()
        b.invert_yaxis()
        b.set(xlabel = 'time (s)', ylabel = 'neuron \nactivity $a_i(t)$')
        plt.show(b)
        b = plot_spikes(preprocess_spikes(firing_dataframe['time'],top_spikes))
        b.xlabel("Time [s]")
        b.ylabel("Neuron number")
        plt.show(b)'''


        print 'Plotting Data...'
        
    	figure2, (ax3, ax4) = plt.subplots(1, 2)
        if len(firing_dataframe.query("tuning=='strong'")) > 0:
            sns.tsplot(time="time",value="firing_rate",unit="neuron-trial",ax=ax3,ci=95,data=firing_dataframe.query("tuning=='strong'").reset_index(), legend = False)
        
        if len(firing_dataframe.query("tuning=='nonpreferred'"))>0:
            sns.tsplot(time="time",value="firing_rate",unit="neuron-trial", ax=ax4,ci=95,data=firing_dataframe.query("tuning=='nonpreferred'").reset_index(), legend = False)
        
        ax3.set(xlabel='time (s)',ylabel='Normalized Firing Rate',title='Preferred Direction', ylim = (0,350))
        ax4.set(xlabel='time (s)',ylim=(0,250),ylabel='',title='Nonpreferred Direction')
        plt.show(figure2)
        
        
        df_correct = self.array_to_pandas(self.accuracy_data, 'correct', 'control')
        self.df_correct = pd.concat([df_correct], ignore_index=True)
        c = sns.tsplot(time='time', value='correct', unit='trial', condition='drug',data = self.df_correct, ci=95, legend = False)
        c.set(xlabel = 'time(s)', ylabel = 'DRT accuracy percentage',title= "Number of trials = %s Prop degraded = %s Degradation = %s" %(self.n_trials, self.proportion,self.degradation))
        c.set(ylim = (.4,1.1))
        plt.show(c)

      
    def go(self):
        primary_dataframe, firing_dataframe = self.multiprocessing()
        self.plot_data(primary_dataframe, firing_dataframe)
        self.primary_dataframe = primary_dataframe
        self.firing_dataframe = firing_dataframe

        

if __name__ == '__main__':
    #n = working_memory_model(proportion = .1, degradation = 1)
    #n.go()
    pass









