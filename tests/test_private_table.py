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
    """an example dataframe for reusing by unit tests."""
    data = {'Name': ['Tom', 'Jack', 'Steve', 'Jack'], 'Age': [28, 34, 29, 42]}
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def example_private_table():
    data = {'Name': ['Tom', 'Jack', 'Steve', 'Jack'], 'Age': [28, 34, 29, 42]}
    df = pd.DataFrame(data)
    domains = {'Name': CategoricalDataDomain(['Tom', 'Jack', 'Steve', 'Eve', 'Adam', 'Lucifer']),
               'Age': RealDataDomain(0., 130.)}
    return PrivateTable(df, domains, PrivacyBudget(100000.0, 1000.))


def test_column_names(example_table: DataFrame):
    domains = {'Name': CategoricalDataDomain(['Tom', 'Jack', 'Steve', 'Eve', 'Adam', 'Lucifer']),
               'Age': RealDataDomain(0., 130.)}
    t = PrivateTable(example_table, domains, PrivacyBudget(1.0, 0.))
    assert 'Age' in t._columns
    assert 'Name' in t._columns


def test_private_mean(example_private_table: PrivateTable):
    """check private mean implementation."""
    noisy_mean = example_private_table.mean('Age', PrivacyBudget(10000.))
    check_absolute_error(noisy_mean, 33.2, 1.)


def test_private_gaussian_mean(example_private_table: PrivateTable):
    """check private guassian mean implementation."""
    noisy_mean = np.mean([example_private_table.gaussian_mean('Age', PrivacyBudget(0.99, 0.5)) for i in range(100)])
    check_absolute_error(noisy_mean, 33.2, 10.)


def test_private_categorical_hist(example_private_table: PrivateTable):
    """check private hist implementation for categorical column."""
    noisy_hist = example_private_table.cat_hist('Name', PrivacyBudget(10000.))

    err = [1, 1, 1]
    noisy_hist.sort()
    assert all(np.abs(noisy_hist-[1, 1, 2]) < err)
    del noisy_hist


def test_private_numerical_hist(example_private_table: PrivateTable):
    """check private hist implementation for numerical column.
    bins:         |.......|.......|.......|
    boundaries:   a0      a1      a2      a3

    """
    bins: List[float] = [20, 30, 40, 50]  # [a0, a1, a2, a3]
    noisy_hist = example_private_table.num_hist('Age', bins, PrivacyBudget(10000.))
    err = [1, 1, 1]
    noisy_hist.sort()
    assert all(np.abs(noisy_hist-[1, 1, 2]) < err)
    del noisy_hist, bins


def test_private_std(example_private_table: PrivateTable):
    """check private std implementation."""
    noisy_std = example_private_table.std('Age', PrivacyBudget(10000.))
    check_absolute_error(noisy_std, 5.54, 1.)
    del noisy_std


def test_private_var(example_private_table: PrivateTable):
    """check private var implementation."""
    noisy_var = example_private_table.var('Age', PrivacyBudget(10000.))
    check_absolute_error(noisy_var, 30.69, 2.)
    del noisy_var


def test_private_max(example_private_table: PrivateTable):
    """check private max implementation."""
    noisy_max = example_private_table.max('Age', PrivacyBudget(10000.))
    check_absolute_error(noisy_max, 42., 1.)
    del noisy_max


def test_private_min(example_private_table: PrivateTable):
    """check private min implementation."""
    noisy_min = example_private_table.min('Age', PrivacyBudget(10000.))
    check_absolute_error(noisy_min, 28., 1.)
    del noisy_min


def test_private_median(example_private_table: PrivateTable):
    """check private median implementation."""
    noisy_median = example_private_table.median('Age', PrivacyBudget(10000.))
    check_absolute_error(noisy_median, 31.5, 1.)
    del noisy_median


def test_private_mode(example_private_table: PrivateTable):
    """check private mode implementation."""
    noisy_mode = example_private_table.mode('Name', PrivacyBudget(10000.))
    assert noisy_mode == "Jack"
    del noisy_mode
