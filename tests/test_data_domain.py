from data_domain import CategoricalDataDomain, RealDataDomain


def test_real_data_domain():
    dd = RealDataDomain(-1., 1.)
    assert dd.contains(0.) == True
    assert dd.contains(-2.) == False
    assert dd.contains(3.) == False


def test_cat_data_domain():
    dd = CategoricalDataDomain([1, 2, 3])
    assert dd.contains(2) == True
    assert dd.contains(None) == False
    assert dd.contains('2') == False
