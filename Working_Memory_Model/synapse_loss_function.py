import math
#This is a synapse loss function DESCRIBE MORE

def synapse_loss(tau, t):
	loss = math.exp((math.log(0.9)*t)/tau)
	if loss <= (1/math.e):
		loss = 1.0
	return loss

print synapse_loss(30, 284)