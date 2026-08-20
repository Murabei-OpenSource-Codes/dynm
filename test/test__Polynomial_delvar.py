"""Test dlm model parameters estimation."""
import numpy as np
import unittest
from dynm.dynamic_model import BayesianDynamicModel

# Simulating the data
nobs = 80
sd_y = 1e-08

y = np.zeros(nobs)
x = np.zeros([nobs, 2])

# Initial information
y[0] = 1e-05
mu = 1e-05

# First observation
np.random.seed(1111)
for t in range(1, nobs):
    # Random errors
    nu = np.random.normal(loc=0, scale=sd_y)

    # Observation
    y[t] = mu + nu


class TestPolynomialandRegression(unittest.TestCase):
    """Tests BayesianDynamicModel results for Dynamic Linear Model."""

    def test__observational_variance_discount_estimate(self):
        """Test models variance estimation with variance discount."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": .995,
            }
        }

        # Fit
        mod1 = BayesianDynamicModel(model_dict=model_dict).fit(y=y)
        mod2 = BayesianDynamicModel(model_dict=model_dict, delvar=.9).fit(y=y)

        estimate_error1 = np.abs(mod1.s - sd_y ** 2)
        estimate_error2 = np.abs(mod2.s - sd_y ** 2)

        self.assertTrue(estimate_error2 < estimate_error1)

    def test__observational_variance_discount_likelihood(self):
        """Test models likelihood with variance discount."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": .995,
            }
        }

        # Fit
        mod1 = BayesianDynamicModel(model_dict=model_dict).fit(y=y)
        mod2 = BayesianDynamicModel(model_dict=model_dict, delvar=.9).fit(y=y)

        llk1 = mod1.llk
        llk2 = mod2.llk

        self.assertTrue(llk1 < llk2)

    def test__observational_variance_credible_interval(self):
        """Test observational variance credible interval."""
        model_dict = {
            "polynomial": {
                "m0": np.array([10]),
                "C0": np.array([[9]]),
                "ntrend": 1,
                "discount": .995,
            }
        }

        # Fit
        mod = BayesianDynamicModel(model_dict=model_dict).fit(y=y)
        posterior = mod.dict_filter.get("posterior")
        var_df = posterior[
            posterior["parameter"] == "observational_variance"]

        lower = var_df["ci_lower"].to_numpy()
        upper = var_df["ci_upper"].to_numpy()
        mean = var_df["mean"].to_numpy()

        self.assertTrue(np.all(lower <= mean))
        self.assertTrue(np.all(mean <= upper))
        self.assertTrue(np.all(lower >= 0.0))
