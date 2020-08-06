#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 21:47:31 2020

@author: vivekchari
"""
import warnings

import numpy as np

import nengo
from nengo.exceptions import ValidationError
from nengo.utils.compat import is_iterable, range
from nengo.utils.network import with_self
from nengo.dists import Choice, Exponential, Uniform

class EnsembleCollection(nengo.Network):
    def __init__(self):
        super(EnsembleCollection, self).__init__()
        self.n_ensembles = 2
        self.dimensions_per_ensemble = 2
        self.ensembles = [('bin1',500), ('bin2',500)]
        self.ea_ensembles = []
        self.tau = .01
        self.neuron_type = nengo.LIF()
        self.total_neurons = 1000
        with self:
            self.input = nengo.Node(size_in = self.dimensions_per_ensemble, 
                                    label = 'input')
            self.output = nengo.Ensemble(1, 2)

            for i  in range(self.n_ensembles):
                e = nengo.Ensemble(self.ensembles[i][1], self.dimensions_per_ensemble, neuron_type = self.neuron_type)
                
                nengo.Connection(self.input,e, synapse=None)
                self.ea_ensembles.append(e)
                
#            nengo.Connection(self.input,self.ea_ensembles[0], synapse=None)
            
            for i in range(self.n_ensembles):
                for j in range(self.n_ensembles):
                    if i == j:
                        pass
                    elif i != j:
                        nengo.Connection(self.ea_ensembles[i],self.ea_ensembles[j], 
                                     synapse = self.tau)
                
    
            self.out = self.add_output('out', function = self.reduction_function, synapse = None)
            nengo.Connection(self.out, self.output, synapse = None)
    @with_self
    def add_output(self, name, function, synapse = None, **conn_kwargs):
        dims_per_ens = self.dimensions_per_ensemble
        
        sizes = np.zeros(self.n_ensembles, dtype=int)
        
        if is_iterable(function) and all(callable(f) for f in function):
            if len(list(function)) != self.n_ensembles:
                raise ValidationError(
                    "Must have one function per ensemble", attr='function')

            for i, func in enumerate(function):
                sizes[i] = np.asarray(func(np.zeros(dims_per_ens))).size
                
        elif callable(function):
            sizes[:] = np.asarray(function(np.zeros(dims_per_ens))).size
            function = [function] * self.n_ensembles
            
        elif function is None:
            
            sizes[:] = dims_per_ens
            
            function = [None] * self.n_ensembles
            
        else:
            
            raise ValidationError("'function' must be a callable, list of "
                                  "callables, or None", attr='function')

        output = nengo.Node(output=None, size_in = sizes.sum(), label=name)
        
        setattr(self, name, output)
        indices = np.zeros(len(sizes) + 1, dtype=int)
        indices[1:] = np.cumsum(sizes)
        for i, e in enumerate(self.ea_ensembles):
            nengo.Connection(
                e, output[indices[i]:indices[i+1]], function = function[i],
                synapse=synapse, **conn_kwargs)

        return output
    
    @with_self
    def add_neuron_input(self):
        if isinstance(self.ea_ensembles[0].neuron_type, nengo.Direct):
            raise ValidationError(
                "Ensembles use Direct neuron type. "
                "Cannot give neuron input to Direct neurons.",
                attr='ea_ensembles[0].neuron_type', obj=self)

        self.neuron_input = nengo.Node(
            size_in = self.total_neurons,
            label="neuron_input")

        for i, ens in enumerate(self.ea_ensembles):
            nengo.Connection(
                self.neuron_input[i * 500:
                                  (i + 1) * 500],
                ens.neurons, synapse=None)
        return self.neuron_input    
    def reduction_function(self, x):
        return x[0]
    
    @with_self
    def add_neuron_output(self):
        """Adds a node that collects the neural output of all ensembles.

        Direct neuron output is useful for plotting the spike raster of
        all neurons in the ensemble array.

        This node is accessible through the 'neuron_output' attribute
        of this ensemble array.
        """
        if isinstance(self.ea_ensembles[0].neuron_type, nengo.Direct):
            raise ValidationError(
                "Ensembles use Direct neuron type. "
                "Cannot get neuron output from Direct neurons.",
                attr='ea_ensembles[0].neuron_type', obj=self)

        self.neuron_output = nengo.Node(
            size_in=500 * self.n_ensembles,
            label="neuron_output")

        for i, ens in enumerate(self.ea_ensembles):
            nengo.Connection(
                ens.neurons,
                self.neuron_output[i * 500:
                                   (i + 1) * 500],
                synapse=None)
        return self.neuron_output

            

if __name__ == '__main__':
    n = EnsembleCollection()
    print n.ea_ensembles[1]