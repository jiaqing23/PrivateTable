import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import privacy_budget
import private_mechanisms
import private_table
from tests import adult_test
from tests import test_private_table

if __name__ == '__main__':
    # # formating source code
    # #os.system('isort *.py **/*.py')
    # #os.system('autopep8 --global-config setup.cfg -i *.py **/*.py')
    # # For Windows use below
    # os.system('isort .')
    # os.system('autopep8 --global-config setup.cfg -i -r .')

    # # type checking
    # #os.system('mypy *.py tests/*.py')
    # # For Windows use below
    # os.system('mypy .')
    # os.system('mypy tests')

    # # run unit tests using pytest
    # os.system('pytest -s .')

    # # clean up
    # #os.system('py3clean .')
    # os.system('rm -r .pytest_cache')
    # # For Windows use below
    # os.system('pyclean .')

    privacybudget = privacy_budget.PrivacyBudget(1,0.1)
    adultTest = adult_test.AdultTest()
    educationtable = adultTest.education_table()
    agetable = adultTest.age_table()

    test_private_table.test_private_laplace_mean(agetable, 'Age', privacybudget)
    test_private_table.test_private_gaussian_mean(agetable, 'Age', privacybudget)
