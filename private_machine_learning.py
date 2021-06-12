import numpy as np
from numpy.random import normal
import random
from datetime import datetime
from private_mechanisms import gaussian_mechanism
from privacy_budget import PrivacyBudget
from privacy_budget_tracker import *

def private_SGD(gradient_function, get_weights_function, update_weights_function, learning_rate_function, 
                train_data, group_size, gradient_norm_bound, number_of_steps, sigma, moment_privacy_budget_tracker,
                test_interval=None, test_function=None):

    def gaussian_noise(x, standard_deviation):
        shape = (1, ) if isinstance(x, (int, float)) else x.shape
        noise = normal(loc=0.,
                    scale=standard_deviation,
                    size=shape)
        return x + noise
    
    random.seed(datetime.now())
    idx = list(range(len(train_data)))
    random.shuffle(idx)
    train_data = train_data[idx]
    number_of_group = len(train_data)//group_size
    
    for step in range(number_of_steps):
        group_id = int(random.random()*number_of_group)
        train_data_group = train_data[group_size*group_id: group_size*(group_id+1)]
        total_grad = None
        total_loss = 0
        
        for i in range(len(train_data_group)):
            grad = np.array(gradient_function(train_data_group[i]), dtype=object)
            grad /= max(1, np.linalg.norm(np.hstack([np.array(i).flatten() for i in grad]))/gradient_norm_bound)
            total_grad = (total_grad + grad) if i > 0 else grad
        
        total_grad = np.array([gaussian_noise(i, sigma) for i in total_grad], dtype=object)
        total_grad /= len(train_data_group)
        
        weights = get_weights_function()
        for i in range(len(weights)):
            weights[i] -= learning_rate_function(step+1) * total_grad[i]
        update_weights_function(weights)
        

        if test_function and step%test_interval==0:
            test_function()

    moment_privacy_budget_tracker.update_privacy_loss(group_size/len(train_data), 
                                                        sigma, number_of_steps, moment_order = 32 , target_delta = 0.5/len(train_data))
