from typing import List

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from data_domain import CategoricalDataDomain, RealDataDomain
from privacy_budget import PrivacyBudget
from private_table import PrivateTable
from utils import check_absolute_error


@pytest.fixture
def example_table():
    """creating a table from the Age and Education columns of the adult dataset"""
    adult_data = pd.read_csv("adult.data.txt",
                             names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
                                    "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                                    "Hours per week", "Country", "Target"])
    data = {'Age': adult_data["Age"].tolist(), 'Education': adult_data["Education"].tolist()}
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def example_private_table():
    adult_data = pd.read_csv("adult.data.txt",
                             names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
                                    "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                                    "Hours per week", "Country", "Target"])
    data = {'Age': adult_data["Age"].tolist(), 'Education': adult_data["Education"].tolist()}
    df = pd.DataFrame(data)
    domains = {'Age': RealDataDomain(17., 90.),
               'Education': CategoricalDataDomain([' Bachelors', ' HS-grad', ' 11th', ' Masters',
                                                   ' 9th', ' Some-college', ' Assoc-acdm', ' Assoc-voc', ' 7th-8th', ' Doctorate',
                                                   ' Prof-school', ' 5th-6th', ' 10th', ' 1st-4th', ' Preschool', ' 12th'])}
    return PrivateTable(df, domains, PrivacyBudget(100000.0, 1.), option="Advance")


def test_column_names(example_table: DataFrame):
    """check to ensure column names correspond to the domains"""
    domains = {'Age': RealDataDomain(0., 130.),
               'Education': CategoricalDataDomain([' Bachelors', ' HS-grad', ' 11th', ' Masters',
                                                   ' 9th', ' Some-college', ' Assoc-acdm', ' Assoc-voc', ' 7th-8th', ' Doctorate',
                                                   ' Prof-school', ' 5th-6th', ' 10th', ' 1st-4th', ' Preschool', ' 12th'])}
    t = PrivateTable(example_table, domains, PrivacyBudget(1.0, 0.))
    assert 'Age' in t._columns
    assert 'Education' in t._columns


def test_private_mean(example_private_table: PrivateTable):
    """check private mean implementation using Age in adult dataset."""
    noisy_mean = example_private_table.mean('Age', PrivacyBudget(10000.))
    check_absolute_error(noisy_mean, 38.58164675532078, 1.)


def test_private_gaussian_mean(example_private_table: PrivateTable):
    """check private guassian mean implementation using Age in adult dataset."""
    noisy_mean = example_private_table.gaussian_mean('Age', PrivacyBudget(10000., 0.1))
    check_absolute_error(noisy_mean, 38.58164675532078, 1.)


def test_private_categorical_hist(example_private_table: PrivateTable):
    """check private hist implementation for categorical column of Education in adult dataset.
    bins:       HS-grad, Bachelors etc
    
    """
    noisy_hist = example_private_table.cat_hist('Education', PrivacyBudget(10000.))

    err = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    noisy_hist.sort()
    assert all(np.abs(noisy_hist-[51, 168, 333, 413, 433, 514, 576, 646,
                                  933, 1067, 1175, 1382, 1723, 5355, 7291, 10501]) < err)
    del noisy_hist


def test_private_numerical_hist(example_private_table: PrivateTable):
    """check private hist implementation for numerical column of Age in adult dataset.
    bins:         17, 18, 19 ... 90

    """
    bins: List[float] = [int(i) for i in range(17, 91)] 
    noisy_hist = example_private_table.num_hist('Age', bins, PrivacyBudget(10000.))
    err = [int(1) for i in range(1, 74)]
    noisy_hist.sort()
    assert all(np.abs(noisy_hist-[1, 1, 3, 3, 6, 10, 12, 20, 22, 22, 23, 29,
                                  43, 45, 46, 51, 64, 67, 72, 89, 108, 120, 150, 151, 178, 208, 230,
                                  258, 300, 312, 355, 358, 366, 366, 395, 415, 419, 464, 478, 543, 550,
                                  577, 595, 602, 708, 712, 720, 724, 734, 737, 753, 765, 770, 780, 785,
                                  794, 798, 808, 813, 816, 827, 828, 835, 841, 858, 861, 867, 875, 876,
                                  877, 886, 888, 898]) < err)
    del noisy_hist, bins


def test_private_std(example_private_table: PrivateTable):
    """check private std implementation using Age in adult dataset."""
    noisy_std = example_private_table.std('Age', PrivacyBudget(10000.))
    check_absolute_error(noisy_std, 13.640432553581146, 1.)
    del noisy_std


def test_private_var(example_private_table: PrivateTable):
    """check private var implementation using Age in adult dataset."""
    noisy_var = example_private_table.var('Age', PrivacyBudget(10000.))
    check_absolute_error(noisy_var, 186.06140024879625, 2.)
    del noisy_var


def test_private_max(example_private_table: PrivateTable):
    """check private max implementation using Age in adult dataset."""
    noisy_max = example_private_table.max('Age', PrivacyBudget(10000.))
    check_absolute_error(noisy_max, 90., 1.)
    del noisy_max


def test_private_min(example_private_table: PrivateTable):
    """check private min implementation using Age in adult dataset."""
    noisy_min = example_private_table.min('Age', PrivacyBudget(10000.))
    check_absolute_error(noisy_min, 17., 1.)
    del noisy_min


def test_private_median(example_private_table: PrivateTable):
    """check private median implementation using Age in adult dataset."""
    noisy_median = example_private_table.median('Age', PrivacyBudget(10000.))
    check_absolute_error(noisy_median, 37., 1.)
    del noisy_median


def test_private_mode(example_private_table: PrivateTable):
    """check private mode implementation using Education in adult dataset."""
    noisy_mode = example_private_table.mode('Education', PrivacyBudget(10000.))
    assert noisy_mode == " HS-grad"
    del noisy_mode
