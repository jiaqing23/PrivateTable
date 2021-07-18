import os
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
    """creating a table from the iris dataset"""
    iris_data = pd.read_csv(os.path.join("dataset", "iris_data.txt"),
                            names=["Sepal Length", "Sepal Width", "Petal Length",
                                   "Petal Width", "Class"])
    data = {'Sepal Length': iris_data["Sepal Length"].tolist(),
            'Sepal Width': iris_data["Sepal Width"].tolist(),
            'Petal Length': iris_data["Petal Length"].tolist(),
            'Petal Width': iris_data["Petal Width"].tolist(),
            'Class': iris_data["Class"].tolist()}
    df = pd.DataFrame(data)
    return df


@ pytest.fixture
def example_private_table():
    iris_data = pd.read_csv(os.path.join("dataset", "iris_data.txt"),
                            names=["Sepal Length", "Sepal Width", "Petal Length",
                                   "Petal Width", "Class"])
    data = {'Sepal Length': iris_data["Sepal Length"].tolist(),
            'Sepal Width': iris_data["Sepal Width"].tolist(),
            'Petal Length': iris_data["Petal Length"].tolist(),
            'Petal Width': iris_data["Petal Width"].tolist(),
            'Class': iris_data["Class"].tolist()}
    df = pd.DataFrame(data)
    domains = {'Sepal Length': RealDataDomain(0., 10.),
               'Sepal Width': RealDataDomain(0., 10.),
               'Petal Length': RealDataDomain(0., 10.),
               'Petal Width': RealDataDomain(0., 10.),
               'Class': CategoricalDataDomain(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])}
    return PrivateTable(df, domains, PrivacyBudget(100000.0, 1.))


def test_column_names(example_table: DataFrame):
    """check to ensure column names correspond to the domains"""
    domains = {'Sepal Length': RealDataDomain(0., 10.),
               'Sepal Width': RealDataDomain(0., 10.),
               'Petal Length': RealDataDomain(0., 10.),
               'Petal Width': RealDataDomain(0., 10.),
               'Class': CategoricalDataDomain(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])}
    t = PrivateTable(example_table, domains, PrivacyBudget(1.0, 0.))
    assert 'Sepal Length' in t._columns
    assert 'Sepal Width' in t._columns
    assert 'Petal Length' in t._columns
    assert 'Petal Width' in t._columns
    assert 'Class' in t._columns


def test_private_mean_sepal_length(example_private_table: PrivateTable):
    """check private mean implementation using Sepal Length in iris dataset."""
    noisy_mean = example_private_table.mean('Sepal Length', PrivacyBudget(10000.))
    check_absolute_error(noisy_mean, 5.843333333333335, 1.)


def test_private_mean_sepal_width(example_private_table: PrivateTable):
    """check private mean implementation using Sepal Width in iris dataset."""
    noisy_mean = example_private_table.mean('Sepal Width', PrivacyBudget(10000.))
    check_absolute_error(noisy_mean, 3.0540000000000007, 1.)


def test_private_mean_petal_length(example_private_table: PrivateTable):
    """check private mean implementation using Petal Length in iris dataset."""
    noisy_mean = example_private_table.mean('Petal Length', PrivacyBudget(10000.))
    check_absolute_error(noisy_mean, 3.7586666666666693, 1.)


def test_private_mean_petal_width(example_private_table: PrivateTable):
    """check private mean implementation using Petal Width in iris dataset."""
    noisy_mean = example_private_table.mean('Petal Width', PrivacyBudget(10000.))
    check_absolute_error(noisy_mean, 1.1986666666666672, 1.)


def test_private_gaussian_mean_sepal_length(example_private_table: PrivateTable):
    """check private gaussian mean implementation using Sepal Length in iris dataset."""
    noisy_mean = example_private_table.gaussian_mean('Sepal Length', PrivacyBudget(0.99, 0.5))
    check_absolute_error(noisy_mean, 5.843333333333335, 1.)


def test_private_gaussian_mean_sepal_width(example_private_table: PrivateTable):
    """check private gaussian mean implementation using Sepal Width in iris dataset."""
    noisy_mean = example_private_table.gaussian_mean('Sepal Width', PrivacyBudget(0.99, 0.5))
    check_absolute_error(noisy_mean, 3.0540000000000007, 1.)


def test_private_gaussian_mean_petal_length(example_private_table: PrivateTable):
    """check private gaussian mean implementation using Petal Length in iris dataset."""
    noisy_mean = example_private_table.gaussian_mean('Petal Length', PrivacyBudget(0.99, 0.5))
    check_absolute_error(noisy_mean, 3.7586666666666693, 1.)


def test_private_gaussian_mean_petal_width(example_private_table: PrivateTable):
    """check private gaussian mean implementation using Petal Width in iris dataset."""
    noisy_mean = example_private_table.gaussian_mean('Petal Width', PrivacyBudget(0.99, 0.5))
    check_absolute_error(noisy_mean, 1.1986666666666672, 1.)


def test_private_std_sepal_length(example_private_table: PrivateTable):
    """check private std implementation using Sepal Length in iris dataset."""
    noisy_std = example_private_table.std('Sepal Length', PrivacyBudget(10000.))
    check_absolute_error(noisy_std, 0.8280661279778629, 1.)
    del noisy_std


def test_private_std_sepal_width(example_private_table: PrivateTable):
    """check private std implementation using Sepal Width in iris dataset."""
    noisy_std = example_private_table.std('Sepal Width', PrivacyBudget(10000.))
    check_absolute_error(noisy_std, 0.4335943113621737, 1.)
    del noisy_std


def test_private_std_petal_length(example_private_table: PrivateTable):
    """check private std implementation using Petal Length in iris dataset."""
    noisy_std = example_private_table.std('Petal Length', PrivacyBudget(10000.))
    check_absolute_error(noisy_std, 1.7644204199522617, 1.)
    del noisy_std


def test_private_std_petal_width(example_private_table: PrivateTable):
    """check private std implementation using Petal Width in iris dataset."""
    noisy_std = example_private_table.std('Petal Width', PrivacyBudget(10000.))
    check_absolute_error(noisy_std, 0.7631607417008414, 1.)
    del noisy_std


def test_private_var_sepal_length(example_private_table: PrivateTable):
    """check private var implementation using Sepal Length in iris dataset."""
    noisy_var = example_private_table.var('Sepal Length', PrivacyBudget(10000.))
    check_absolute_error(noisy_var, 0.6856935123042505, 1.)
    del noisy_var


def test_private_var_sepal_width(example_private_table: PrivateTable):
    """check private var implementation using Sepal Width in iris dataset."""
    noisy_var = example_private_table.var('Sepal Width', PrivacyBudget(10000.))
    check_absolute_error(noisy_var, 0.18800402684563763, 1.)
    del noisy_var


def test_private_var_petal_length(example_private_table: PrivateTable):
    """check private var implementation using Petal Length in iris dataset."""
    noisy_var = example_private_table.var('Petal Length', PrivacyBudget(10000.))
    check_absolute_error(noisy_var, 3.1131794183445156, 1.)
    del noisy_var


def test_private_var_petal_width(example_private_table: PrivateTable):
    """check private var implementation using Petal Width in iris dataset."""
    noisy_var = example_private_table.var('Petal Width', PrivacyBudget(10000.))
    check_absolute_error(noisy_var, 0.5824143176733784, 1.)
    del noisy_var


def test_private_max_sepal_length(example_private_table: PrivateTable):
    """check private max implementation using Sepal Length in iris dataset."""
    noisy_max = example_private_table.max('Sepal Length', PrivacyBudget(10000.))
    check_absolute_error(noisy_max, 7.9, 1.)
    del noisy_max


def test_private_max_sepal_width(example_private_table: PrivateTable):
    """check private max implementation using Sepal Width in iris dataset."""
    noisy_max = example_private_table.max('Sepal Width', PrivacyBudget(10000.))
    check_absolute_error(noisy_max, 4.4, 1.)
    del noisy_max


def test_private_max_petal_length(example_private_table: PrivateTable):
    """check private max implementation using Petal Length in iris dataset."""
    noisy_max = example_private_table.max('Petal Length', PrivacyBudget(10000.))
    check_absolute_error(noisy_max, 6.9, 1.)
    del noisy_max


def test_private_max_petal_width(example_private_table: PrivateTable):
    """check private max implementation using Petal Width in iris dataset."""
    noisy_max = example_private_table.max('Petal Width', PrivacyBudget(10000.))
    check_absolute_error(noisy_max, 2.5, 1.)
    del noisy_max


def test_private_min_sepal_length(example_private_table: PrivateTable):
    """check private min implementation using Sepal Length in iris dataset."""
    noisy_min = example_private_table.min('Sepal Length', PrivacyBudget(10000.))
    check_absolute_error(noisy_min, 4.3, 1.)
    del noisy_min


def test_private_min_sepal_width(example_private_table: PrivateTable):
    """check private min implementation using Sepal Width in iris dataset."""
    noisy_min = example_private_table.min('Sepal Width', PrivacyBudget(10000.))
    check_absolute_error(noisy_min, 2.0, 1.)
    del noisy_min


def test_private_min_petal_length(example_private_table: PrivateTable):
    """check private min implementation using Petal Length in iris dataset."""
    noisy_min = example_private_table.min('Petal Length', PrivacyBudget(10000.))
    check_absolute_error(noisy_min, 1.0, 1.)
    del noisy_min


def test_private_min_petal_width(example_private_table: PrivateTable):
    """check private min implementation using Petal Width in iris dataset."""
    noisy_min = example_private_table.min('Petal Width', PrivacyBudget(10000.))
    check_absolute_error(noisy_min, 0.1, 1.)
    del noisy_min


def test_private_median_sepal_length(example_private_table: PrivateTable):
    """check private median implementation using Sepal Length in iris dataset."""
    noisy_median = example_private_table.median('Sepal Length', PrivacyBudget(10000.))
    check_absolute_error(noisy_median, 5.8, 1.)
    del noisy_median


def test_private_median_sepal_width(example_private_table: PrivateTable):
    """check private median implementation using Sepal Width in iris dataset."""
    noisy_median = example_private_table.median('Sepal Width', PrivacyBudget(10000.))
    check_absolute_error(noisy_median, 3.0, 1.)
    del noisy_median


def test_private_median_petal_length(example_private_table: PrivateTable):
    """check private median implementation using Petal Length in iris dataset."""
    noisy_median = example_private_table.median('Petal Length', PrivacyBudget(10000.))
    check_absolute_error(noisy_median, 4.35, 1.)
    del noisy_median


def test_private_median_petal_width(example_private_table: PrivateTable):
    """check private median implementation using Petal Width in iris dataset."""
    noisy_median = example_private_table.median('Petal Width', PrivacyBudget(10000.))
    check_absolute_error(noisy_median, 1.3, 1.)
    del noisy_median


def test_private_categorical_hist(example_private_table: PrivateTable):
    """check private hist implementation for categorical column of Classes in iris dataset.
    bins:       Iris-setosa, Iris-versicolor, Iris-virginica

    """
    noisy_hist = example_private_table.cat_hist('Class', PrivacyBudget(10000.))

    err = [1, 1, 1]
    noisy_hist.sort()
    assert all(np.abs(noisy_hist-[50, 50, 50]) < err)
    del noisy_hist
