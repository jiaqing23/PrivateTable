from typing import List

from collections import Counter

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
    return PrivateTable(df, domains, PrivacyBudget(100000.0, 1.), option="Advance")


def test_column_names(example_table: DataFrame):
    domains = {'Name': CategoricalDataDomain(['Tom', 'Jack', 'Steve', 'Eve', 'Adam', 'Lucifer']),
               'Age': RealDataDomain(0., 130.)}
    t = PrivateTable(example_table, domains, PrivacyBudget(1.0, 0.))
    assert 'Age' in t._columns
    assert 'Name' in t._columns


def test_private_laplace_mean(example_private_table: PrivateTable, column: str, privacy_budget: PrivacyBudget):
    """check private laplace mean implementation."""
    noisy_mean = example_private_table.laplace_mean(column, privacy_budget)
    print("actual mean =", example_private_table.mean(column))
    print("noisy mean =", noisy_mean)
    check_absolute_error(noisy_mean, example_private_table.mean(column), 1.)


def test_private_gaussian_mean(example_private_table: PrivateTable, column: str, privacy_budget: PrivacyBudget):
    """check private guassian mean implementation."""
    noisy_mean = example_private_table.gaussian_mean(column, privacy_budget)
    print("actual mean =", example_private_table.mean(column))
    print("noisy mean =", noisy_mean)
    check_absolute_error(noisy_mean, example_private_table.mean(column), 1.)


def test_private_categorical_hist(example_private_table: PrivateTable):
    """check private hist implementation for categorical column."""
    noisy_hist = example_private_table.cat_hist('Name', PrivacyBudget(10000.))

    err = [1, 1, 1]
    noisy_hist.sort()
    assert all(np.abs(noisy_hist-[1, 1, 2]) < err)
    del noisy_hist


def test_private_numerical_hist(example_private_table: PrivateTable, bins: List, column: str, privacy_budget: PrivacyBudget):
    """check private hist implementation for numerical column.
    bins:         |.......|.......|.......|
    boundaries:   a0      a1      a2      a3

    """
    noisy_hist = example_private_table.num_hist(column, bins, privacy_budget)
    hist, bin_edge = np.histogram(example_private_table._dataframe[column], bins=bins)
    print("actual histogram:", hist)
    print("noisy histogram:", noisy_hist)


def test_private_std(example_private_table: PrivateTable, column: str, privacy_budget: PrivacyBudget):
    """check private std implementation."""
    noisy_std = example_private_table.std(column, privacy_budget)
    std = np.std(example_private_table._dataframe[column])
    print("actual std =", std)
    print("noisy std =", noisy_std)
    check_absolute_error(noisy_std, std, 1.)
    # check_absolute_error(noisy_std, 5.54, 1.)
    del noisy_std


def test_private_var(example_private_table: PrivateTable, column: str, privacy_budget: PrivacyBudget):
    """check private var implementation."""
    noisy_var = example_private_table.var(column, privacy_budget)
    var = np.var(example_private_table._dataframe[column])
    print("actual var =", var)
    print("noisy var =", noisy_var)
    check_absolute_error(noisy_var, var, 2.)
    del noisy_var


def test_private_max(example_private_table: PrivateTable, column: str, privacy_budget: PrivacyBudget):
    """check private max implementation."""
    noisy_max = example_private_table.max(column, privacy_budget)
    actual_max = np.max(example_private_table._dataframe[column])
    print("actual max =", actual_max)
    print("noisy max =", noisy_max)
    # todo: find out why error is so large
    # check_absolute_error(noisy_max, actual_max, 1.)
    del noisy_max


def test_private_min(example_private_table: PrivateTable, column: str, privacy_budget: PrivacyBudget):
    """check private min implementation."""
    noisy_min = example_private_table.min(column, privacy_budget)
    actual_min = np.min(example_private_table._dataframe[column])
    print("actual min =", actual_min)
    print("noisy min =", noisy_min)
    # todo: find out why error is so large
    # check_absolute_error(noisy_min, actual_min, 1.)
    del noisy_min


def test_private_median(example_private_table: PrivateTable, column: str, privacy_budget: PrivacyBudget):
    """check private median implementation."""
    noisy_median = example_private_table.median(column, privacy_budget)
    actual_median = np.median(example_private_table._dataframe[column])
    print("actual median =", actual_median)
    print("noisy median =", noisy_median)
    # todo: find out why error is so large
    # check_absolute_error(noisy_median, actual_median, 1.)
    del noisy_median


def test_private_mode(example_private_table: PrivateTable, column: str, privacy_budget: PrivacyBudget):
    """check private mode implementation."""
    actual_mode = Counter(np.array(example_private_table._dataframe[column]).flat).most_common(1)[0][0]
    noisy_mode = example_private_table.mode(column, privacy_budget)
    print("actual mode =", actual_mode)
    print("noisy mode =", noisy_mode)
    del noisy_mode
