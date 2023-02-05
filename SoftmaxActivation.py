from numpy import exp
import numpy as np
import matplotlib.pyplot as plt
# calculate the softmax of a vector
def softmax(vector):
	e = exp(vector)
	return e / e.sum()
def sigmoid(x):
    return 1/(1 + np.exp(-x))
# define data
data = [-1.5, 2.2, -0.8, 3.6]
# convert list of numbers to a list of probabilities
print(f"Input vector:{data}")
result_softmax = softmax(data)
# report the probabilities
print(f"softmax result:{result_softmax}")
sum_softmax=0.0
for i in range(0, len(result_softmax)):    
   sum_softmax = sum_softmax + result_softmax[i];    
     
print(f"Sum of all the elements of softmax results: {sum_softmax}");    
print("")
sig_result=[0] *len(data)
sum_sigmoid=0
for i in range(0, len(data)):    
   sig_result[i] = sigmoid(data[i]);
   print(f"Sigmoid result {i}: {sig_result[i]}")
   sum_sigmoid = sum_sigmoid + sig_result[i];    
     
print(f"Sum of all the elements of Sigmoid results: {sum_sigmoid}");    

x = np.linspace(-10, 10, 100)
y = softmax(x)
plt.scatter(x, y) 
plt.title('Softmax Function') 
plt.show()

