import math
import pandas as pd
import numpy as np

#This is a synapse loss function DESCRIBE MORE

neurons = 1000 #Total number of neurons in the system
day_step_size = 20 #How often do you want to simulate the model in days
tot_days = 1000 #Total number of days you want to run for - here 0 - 1000 inclusive
total_steps = tot_days / day_step_size

#Function to calculate the synapse transmission as time continues on 
def synapse_trans(tau, t):
	trans = math.exp((math.log(0.9)*t)/tau)
	if trans <= (1/math.e):
		trans = 1.0
	return trans

#This is the initial probability function that DOES NOT account for the previous number affected
def prob(t):
	probability_aff = (0.001 + (0.00001*t))
	return probability_aff

prev_num_affected = 0
day_and_num_affected = [] #List of lists that have [day number, number affected by amyloid or dead] 
#This loop goes through each time in the total days and finds the number of cells affected by amyloid plaques and appends the day and number to the end of the list above
for t in range(0, tot_days + 1): #Goes from 0 to 1000
	if t == 0:
		num_affected = prob(t)*neurons
	else:
		num_affected = prev_num_affected + (prev_num_affected * prob(t))
	prev_num_affected = num_affected
	if t % day_step_size == 0 or t == 0:
		day_and_num_affected.append([t, num_affected])

#This calculates the new number of cells that will be affected after the current day step
list_new_affected = [] #List of total cells that will be affected after this day - index is the day step (Day 0 is 0, Day 20 is 1, etc)
prev_days_total = 0
for x in day_and_num_affected:
	new_affected = x[-1] - prev_days_total
	prev_days_total = x[-1]
	list_new_affected.append(new_affected)

#This creates a dataframe that holds all the data needed for the simulation
init_day = day_and_num_affected[0]
L0 = neurons - init_day[1]
print(L0)
for step in range(0, total_steps + 1):
	number_transfers = list_new_affected[step]

#Create dataframe that holds synapse transmission from day 0 to 200 - After day 185, cell loses connections
#This is for tau = 20 
df = []
for t in range(0,201,5):
	df.append([t, synapse_trans(20, t)])

data = pd.DataFrame(df)
data.columns = ["Day Number", "Amount of Synapse Transmission"]
