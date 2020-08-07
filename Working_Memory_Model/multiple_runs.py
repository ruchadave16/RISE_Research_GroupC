#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 23:36:01 2020

@author: vivekchari
"""
from model_Network import working_memory_model

n = working_memory_model(proportion = .3, degradation = .0)
n.go()
