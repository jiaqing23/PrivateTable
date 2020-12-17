from typing import List

import pandas as pd
import pytest
from pandas import DataFrame

from data_domain import CategoricalDataDomain, RealDataDomain
from privacy_budget import PrivacyBudget
from private_table import PrivateTable
from utils import check_absolute_error


class AdultTest: 
    def __init__ (self): 
        self.actual_data = pd.read_csv("adult.data.txt",
            names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
            "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
            "Hours per week", "Country", "Target"])

    def age_table(self): 
        age_data = self.actual_data["Age"]
        df = pd.DataFrame(age_data)
        domains = {"Age": RealDataDomain(0., 130.)}
        return PrivateTable(df, domains, PrivacyBudget(100000.0, 0.))

    def edu_years_table(self):
        edu_years_data = self.actual_data["Education-Num"]
        df = pd.DataFrame(edu_years_data)
        domains = {"Education-Num": RealDataDomain(0., 30.)}
        return PrivateTable(df, domains, PrivacyBudget(100000.0, 0.))

    def education_table(self): 
        education_data = self.actual_data["Education"]
        df = pd.DataFrame(education_data)
        domains = {"Education": CategoricalDataDomain([' Bachelors', ' HS-grad', ' 11th', ' Masters', 
        ' 9th', ' Some-college', ' Assoc-acdm', ' Assoc-voc', ' 7th-8th', ' Doctorate', 
        ' Prof-school', ' 5th-6th', ' 10th', ' 1st-4th', ' Preschool', ' 12th'])}
        return PrivateTable(df, domains, PrivacyBudget(100000.0, 0.))




