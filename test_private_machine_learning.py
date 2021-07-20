import random

import numpy as np
import pandas as pd
import pytest

from privacy_budget import PrivacyBudget
from privacy_budget_tracker import MomentPrivacyBudgetTracker
from private_machine_learning import private_SGD
from utils import check_absolute_error


@pytest.fixture
def data():
    np.random.seed(1)
    x = np.random.rand(1000)*100
    data = [(i, 5*i+8) for i in x]
    return data


def test_private_SGD(data):

    train_data, test_data = data[:800], data[800:]
    param = np.random.rand(2)  # y = param[0]*x+param[1]

    def gradient_function(batch_data):
        x, y = batch_data
        y_pred = param[0]*x + param[1]

        d0 = -2.0 * x * (y-y_pred)
        d1 = -2.0 * (y-y_pred)

        return [d0, d1]

    def get_weights_function():
        return np.copy(param)

    def learning_rate_function(step):
        if step < 10:
            return 0.1
        elif step < 50:
            return 0.01
        else:
            return 0.005

    def update_weights_function(new_weight):
        param[:] = new_weight

    def test_function():
        n = len(test_data)
        x = np.array([i[0] for i in test_data])
        y = np.array([i[1] for i in test_data])
        y_pred = param[0]*x + param[1]

        loss = 1.0/n*np.sum((y_pred-y)**2)

        check_absolute_error(loss, 0., 20.)

    moment_accountant = MomentPrivacyBudgetTracker(PrivacyBudget(10, 0.001))

    private_SGD(gradient_function=gradient_function,
                get_weights_function=get_weights_function,
                update_weights_function=update_weights_function,
                learning_rate_function=learning_rate_function,
                train_data=train_data,
                group_size=100,
                gradient_norm_bound=10,
                number_of_steps=100,
                sigma=1,
                moment_privacy_budget_tracker=moment_accountant,
                test_interval=100,
                test_function=test_function
                )

    check_absolute_error(moment_accountant.consumed_privacy_budget.epsilon, 8.805554, 1e-6)
    check_absolute_error(moment_accountant.consumed_privacy_budget.delta, 0.000625, 1e-6)
