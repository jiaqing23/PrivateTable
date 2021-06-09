import numdifftools as nd 
import numpy as np
import random
from datetime import datetime
from private_mechanisms import gaussian_mechanism
from privacy_budget import PrivacyBudget

def predict(theta, x):
    return np.dot(theta, x);

def loss_function(theta, x, y):
    return (predict(theta,x) - y)**2

def gradient(theta, x, y):
    return 2*(predict(theta,x) - y)*x

def private_SGD(gradient_function, learning_rate, group_size, gradient_norm_bound, number_of_steps, 
                    x, y, theta, epsilon, delta):

    random.seed(datetime.now())
    idx = list(range(len(x)))
    random.shuffle(idx)
    x, y = x[idx], y[idx]
    number_of_group = len(x)//group_size
    
    loss = []

    for step in range(number_of_steps):
        group_id = int(random.random()*number_of_group)
        x_group, y_group = x[group_size*group_id: group_size*(group_id+1)], y[group_size*group_id: group_size*(group_id+1)]
        total_grad = 0
        total_loss = 0
        
        for i in range(len(x_group)):
            grad = gradient_function(theta, x_group[i], y_group[i])
            grad /= max(1, np.linalg.norm(grad)/gradient_norm_bound)
            total_grad += grad
            total_loss += loss_function(theta, x_group[i], y_group[i])
            
        total_grad = gaussian_mechanism(total_grad, gradient_norm_bound, PrivacyBudget(epsilon, delta))
        total_grad /= len(x_group)
        total_loss /= len(x_group)

        theta -= learning_rate * total_grad
        print(total_loss)
    
    return theta, PrivacyBudget(group_size/len(x)*epsilon*np.sqrt(number_of_steps), delta) #need to be changed


#self created dataset, y = 2x_1 + x_2
random.seed(0)
x = [[random.random(), random.random()] for i in range(2000)]
y = [(i[0]*3 + i[1]) for i in x]
theta = [random.random(), random.random()]
x, y, theta = np.array(x), np.array(y), np.array(theta)

theta, privact_budget = private_SGD(gradient_function = gradient,
                                    learning_rate = 0.05, 
                                    group_size = 400, 
                                    gradient_norm_bound = 10, 
                                    number_of_steps = 100, 
                                    x = x, 
                                    y = y, 
                                    theta = theta, 
                                    epsilon = 0.9, 
                                    delta = 0.5)

print(theta, privact_budget)