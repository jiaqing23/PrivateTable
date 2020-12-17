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

    adultTest = adult_test.AdultTest()
    educationTable = adultTest.education_table()
    eduYearsTable = adultTest.edu_years_table()
    ageTable = adultTest.age_table()

    '''
    AgeTable
    '''
    print("======= Testing for Laplacian mean on AgeTable =======")
    test_private_table.test_private_laplace_mean(example_private_table=ageTable,
                                                 column='Age',
                                                 privacy_budget=privacy_budget.PrivacyBudget(1, 0))
    print("======================================================")

    print("======= Testing for Gaussian mean on AgeTable ========")
    test_private_table.test_private_gaussian_mean(example_private_table=ageTable,
                                                  column='Age',
                                                  privacy_budget=privacy_budget.PrivacyBudget(1, 0.1))
    print("======================================================")



    '''
    EducationTable
    '''
    print("========= Testing for Mode on EducationTable =========")
    test_private_table.test_private_mode(example_private_table=educationTable,
                                         column='Education',
                                         privacy_budget=privacy_budget.PrivacyBudget(1, 0.1))
    print("======================================================")

    '''
    EduYearsTable
    '''
    print("==== Testing for Laplacian mean on EduYearsTable =====")
    test_private_table.test_private_laplace_mean(example_private_table=eduYearsTable,
                                                 column='Education-Num',
                                                 privacy_budget=privacy_budget.PrivacyBudget(1, 0))
    print("======================================================")

    print("===== Testing for Gaussian mean on EduYearsTable =====")
    test_private_table.test_private_gaussian_mean(example_private_table=eduYearsTable,
                                                  column='Education-Num',
                                                  privacy_budget=privacy_budget.PrivacyBudget(1, 0.1))
    print("======================================================")

    print("== Testing for Numerical Histogram on EduYearsTable ==")
    test_private_table.test_private_numerical_hist(example_private_table=eduYearsTable,
                                                   column='Education-Num',
                                                   bins=[5, 10, 15, 20],
                                                   privacy_budget=privacy_budget.PrivacyBudget(1, 0))
    print("======================================================")

    print("===== Testing for Laplacian std on EduYearsTable =====")
    test_private_table.test_private_std(example_private_table=eduYearsTable,
                                        column='Education-Num',
                                        privacy_budget=privacy_budget.PrivacyBudget(1, 0))
    print("======================================================")

    print("===== Testing for Laplacian var on EduYearsTable =====")
    test_private_table.test_private_var(example_private_table=eduYearsTable,
                                        column='Education-Num',
                                        privacy_budget=privacy_budget.PrivacyBudget(1, 0))
    print("======================================================")

    print("===== Testing for Laplacian max on EduYearsTable =====")
    test_private_table.test_private_max(example_private_table=eduYearsTable,
                                        column='Education-Num',
                                        privacy_budget=privacy_budget.PrivacyBudget(1, 0))
    print("======================================================")

    print("===== Testing for Laplacian min on EduYearsTable =====")
    test_private_table.test_private_min(example_private_table=eduYearsTable,
                                        column='Education-Num',
                                        privacy_budget=privacy_budget.PrivacyBudget(1, 0))
    print("======================================================")

    print("===== Testing for Laplacian median on EduYearsTable =====")
    test_private_table.test_private_median(example_private_table=eduYearsTable,
                                        column='Education-Num',
                                        privacy_budget=privacy_budget.PrivacyBudget(1, 0))
    print("======================================================")