"""Auxiliary functions for results formatting."""
import numpy as np
import pandas as pd
from scipy import stats
from typing import List


def tidy_parameters(dict_parameters: dict, entry_m: str, entry_v: str,
                    names_parameters: List,
                    index_seas_parameters: List = None,
                    F: np.ndarray = None):
    """Transform state-space moments from a dict to a long DataFrame.

    Args:
        dict_parameters (dict):
            Posterior (``m``, ``C``) and prior (``a``, ``R``) moments
            for the state-space parameters over time.
        entry_m (str):
            Key in ``dict_parameters`` for the mean vectors.
        entry_v (str):
            Key in ``dict_parameters`` for the covariance matrices.
        names_parameters (List):
            Names of each state-parameter component.
        index_seas_parameters (List):
            Optional indices of seasonal model components.
        F (np.ndarray):
            Optional regression vector used to sum seasonal harmonics.

    Returns:
        pd.DataFrame:
            Tidy table with ``parameter``, ``mean``, and ``variance``
            for each time.
    """

    def _get_mean(x: np.ndarray):
        """Extract the state mean vector as a DataFrame.

        Args:
            x (np.ndarray):
                Mean vector of state-space parameters (prior or
                posterior).

        Returns:
            pd.DataFrame:
                Mean values indexed by parameter name.
        """
        df_out = pd.DataFrame(
            data={"mean": x[:, 0]}, index=names_parameters)
        if index_seas_parameters:
            j = 1
            lt = []
            for iseas in index_seas_parameters:
                m_seas = x[iseas][:, 0]
                F_seas = F[iseas]
                sum_seas = F_seas.T @ m_seas
                lt.append(pd.DataFrame(
                    data={"mean": sum_seas}, index=["Sum Seas " + str(j)]))
                j = j + 1
            df_out = pd.concat([df_out, pd.concat(lt)])
        return df_out

    def _get_var(x: np.ndarray):
        """Extract the covariance diagonal as a DataFrame.

        Args:
            x (np.ndarray):
                Covariance matrix of state-space parameters (prior or
                posterior).

        Returns:
            pd.DataFrame:
                Variance values indexed by parameter name.
        """
        df_out = pd.DataFrame(
            data={"variance": np.diag(x)}, index=names_parameters)
        if index_seas_parameters:
            j = 1
            lt = []
            for iseas in index_seas_parameters:
                cov_seas = x[np.ix_(iseas, iseas)]
                F_seas = F[iseas]
                sum_seas = F_seas.T @ cov_seas @ F_seas
                lt.append(pd.DataFrame(
                    data={"variance": sum_seas[:, 0]},
                    index=["Sum Seas " + str(j)]))
                j = j + 1
            df_out = pd.concat([df_out, pd.concat(lt)])

        return df_out

    df_mean_parms = pd.concat(
        list(map(_get_mean, dict_parameters[entry_m])))
    df_var_parms = pd.concat(
        list(map(_get_var, dict_parameters[entry_v])))

    df_state_parameters = pd.concat(
        [df_mean_parms.reset_index(),
         df_var_parms.reset_index(drop=True)], axis=1)

    renamed_state_parameters_df = (
        df_state_parameters
        .rename(columns={"index": "parameter"})
        .copy()
    )

    return renamed_state_parameters_df[["parameter", "mean", "variance"]]


def _add_credible_interval_studentt(
        pd_df: pd.DataFrame,
        entry_m: str,
        entry_v: str,
        level=float):
    """Add Student-t credible intervals to a results DataFrame.

    Args:
        pd_df (pd.DataFrame):
            Table with a time column ``t`` and moment columns named by
            ``entry_m`` and ``entry_v``.
        entry_m (str):
            Column name of the location parameter.
        entry_v (str):
            Column name of the scale-squared parameter.
        level (float):
            Tail probability. ``0.05`` yields a 95% interval.

    Returns:
        pd.DataFrame:
            Input table with ``ci_lower`` and ``ci_upper`` columns.
    """
    df = pd_df["t"].to_numpy() + 1
    mu = pd_df[entry_m].to_numpy()
    sigma = np.sqrt(pd_df[entry_v].to_numpy() + 10e-300)

    # Calculate intervals
    pd_df["ci_lower"] = stats.t.ppf(
        q=level / 2,
        df=df,
        loc=mu,
        scale=sigma
    )

    pd_df["ci_upper"] = stats.t.ppf(
        q=1 - level / 2,
        df=df,
        loc=mu,
        scale=sigma
    )

    return pd_df


def _add_credible_interval_gamma(
        pd_df: pd.DataFrame,
        entry_a: str,
        entry_b: str,
        level=float):
    """Add gamma credible intervals to a results DataFrame.

    Args:
        pd_df (pd.DataFrame):
            Table with gamma shape and rate columns.
        entry_a (str):
            Column name of the gamma shape parameter.
        entry_b (str):
            Column name of the gamma rate parameter.
        level (float):
            Tail probability. ``0.05`` yields a 95% interval.

    Returns:
        pd.DataFrame:
            Input table with ``ci_lower`` and ``ci_upper`` columns.
    """
    a = pd_df[entry_a].to_numpy()
    b = 1 / (pd_df[entry_b].to_numpy() + 10e-300)

    # Calculate intervals
    pd_df["ci_lower"] = stats.gamma.ppf(q=level / 2, a=a, scale=b)
    pd_df["ci_upper"] = stats.gamma.ppf(q=1 - level / 2, a=a, scale=b)

    return pd_df


def _create_mod_label_column(mod, t: int):
    """Build model-block labels for each state component and time.

    Args:
        mod:
            Fitted model with DLM and DNM submodel attributes.
        t (int):
            Number of time points to repeat the label block.

    Returns:
        list:
            Structural-block labels aligned with stacked state rows.
    """
    poly_mod = mod.dlm.polynomial_model
    regr_mod = mod.dlm.regression_model
    seas_mod = mod.dlm.seasonal_model

    tf_mod = mod.dnm.transfer_function_model
    ar_mod = mod.dnm.autoregressive_model

    poly_lb = np.repeat("polynomial", len(poly_mod.m))
    regr_lb = np.repeat("regression", len(regr_mod.m))
    seas_lb = np.repeat("seasonal", len(seas_mod.m))

    ar_lb = np.repeat("autoregressive", len(ar_mod.m))
    tf_lb = list(np.repeat(
        ['transfer_function_' + str(i + 1) for i in range(tf_mod.ntfm)],
        2 * tf_mod.lambda_order + tf_mod.gamma_order))

    mod_lb = t * list(
        np.concatenate([poly_lb, regr_lb, seas_lb, ar_lb, tf_lb]))

    return mod_lb


def build_predictive_df(mod, dict_predict: dict, level: float):
    """Build a predictive DataFrame with Student-t intervals.

    Args:
        mod:
            Fitted model. Kept for a consistent builder signature.
        dict_predict (dict):
            Mapping with at least ``t``, ``f``, and ``q``.
        level (float):
            Tail probability. ``0.05`` yields a 95% interval.

    Returns:
        pd.DataFrame:
            Predictive moments with ``ci_lower`` and ``ci_upper``.
    """
    df_predictive = pd.DataFrame(dict_predict)

    # Compute credible intervals
    df_predictive = _add_credible_interval_studentt(
        pd_df=df_predictive, entry_m="f",
        entry_v="q", level=level)

    return df_predictive


def build_posterior_df(
        mod,
        dict_posterior: dict,
        entry_m: str,
        entry_v: str,
        t: int,
        level: float,
        smooth: bool = False):
    """Build a posterior state DataFrame with Student-t intervals.

    Args:
        mod:
            Fitted model used for parameter names and block labels.
        dict_posterior (dict):
            State moments keyed by ``entry_m`` and ``entry_v``.
        entry_m (str):
            Key for mean vectors (``m`` or ``a``).
        entry_v (str):
            Key for covariance matrices (``C`` or ``R``).
        t (int):
            Number of time points in the series.
        level (float):
            Tail probability. ``0.05`` yields a 95% interval.
        smooth (bool):
            If True, time is indexed backwards from ``mod.t``.
            Defaults to False.

    Returns:
        pd.DataFrame:
            Posterior moments with labels and credible intervals.
    """
    # Organize the posterior parameters
    df_posterior = tidy_parameters(
        dict_parameters=dict_posterior,
        entry_m=entry_m, entry_v=entry_v,
        names_parameters=mod.names_parameters)

    # Create model labels
    df_posterior["mod"] = _create_mod_label_column(mod=mod, t=t)

    # Add time column on posterior_df
    if smooth:
        t_index = mod.t - np.arange(0, mod.t)
    else:
        t_index = np.arange(1, t + 1)

    df_posterior["t"] = np.repeat(t_index, mod.m.shape[0])
    df_posterior["t"] = df_posterior["t"].astype(int)

    # Round variance
    df_posterior["variance"] = df_posterior["variance"].round(10)

    # Compute credible intervals
    df_posterior = _add_credible_interval_studentt(
        pd_df=df_posterior, entry_m="mean",
        entry_v="variance", level=level)

    return df_posterior


def build_variance_df(
        mod,
        dict_observation_var: dict,
        level: float):
    """Build an observational-variance DataFrame with gamma intervals.

    Args:
        mod:
            Fitted model. Kept for a consistent builder signature.
        dict_observation_var (dict):
            Mapping with ``t``, ``d``, ``n``, and ``mean``.
        level (float):
            Tail probability. ``0.05`` yields a 95% interval.

    Returns:
        pd.DataFrame:
            Observational variance with ``ci_lower`` and ``ci_upper``.
    """
    # Organize observational variance
    df_var = (
        pd.DataFrame(dict_observation_var)
        .assign(
            variance=lambda x: x.d / (x.n ** 2),
            parameter="observational_variance",
            mod="observational_variance"
        )
        .copy()
    )

    # Organize observational variance
    df_var = _add_credible_interval_gamma(
        pd_df=df_var,
        entry_a="d",
        entry_b="n",
        level=level
    )

    df_var = df_var.drop(['d', 'n'], axis=1).copy()

    return df_var
