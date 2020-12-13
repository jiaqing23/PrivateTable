from privacy_budget import PrivacyBudget, combine_privacy_losses


def test_combine_privacy_losses():
    e1 = PrivacyBudget(1., 0.01)
    e2 = PrivacyBudget(0.2, 0.004)
    e3 = combine_privacy_losses([e1, e2])
    expected_e3 = PrivacyBudget(1. + 0.2, 0.01 + 0.004)
    assert e3 == expected_e3


def test_privacy_budget_class():
    e1 = PrivacyBudget(1., 0.01)
    e2 = PrivacyBudget(0.2, 0.004)
    e3 = PrivacyBudget(1 + 0.2, 0.01 + 0.004)
    assert e3 == e1 + e2
