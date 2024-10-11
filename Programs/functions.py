# Import libraries
from numpy import *
import pandas as pd
import itertools as it
import statsmodels.api as sm
from scipy import optimize

# Define a function to create a data frame of the right form
def expand(d):
    rows = it.product(*d.values())
    return pd.DataFrame.from_records(rows, columns=d.keys())

# Define the leisure utility function
def v_of_ell(x, epsilon=1.0, theta=8.795358530547285):
    return -(theta * epsilon / (1 + epsilon)) * (1 - x)**((1 + epsilon) / epsilon)

# Define the consumption and leisure interpolation/extrapolation function
def filter(x, penalty):
    # Linearly interpolate the missing values coming before the oldest available age
    x = x.interpolate(limit_direction='backward')

    # HP-filter the resulting series
    x[x.notna()] = sm.tsa.filters.hpfilter(x[x.notna()], penalty)[1]

    # Keep the resulting series constant for missing ages coming after the oldest available age
    return x.interpolate(method='ffill', limit_direction='forward')

# Define the level consumption-equivalent welfare calculation function
def cew_level(S_i=None, S_j=None, I_i=None, I_j=None, c_i_bar=None, c_j_bar=None, ell_i_bar=None, ell_j_bar=None, beta=1, g=0, age_min=0, age_max=100, frisch=1.0, # Standard parameters
              S_intercept=None, I_intercept=None, c_intercept=None, ell_intercept=None, vsl=7.4e6, c_nominal=None, age_min_intercept=40, age_max_intercept=100, # Intercept parameters
              inequality=False, c_i_bar_nd=None, c_j_bar_nd=None, Elog_of_c_i=None, Elog_of_c_j=None, Elog_of_c_i_nd=None, Elog_of_c_j_nd=None, Ev_of_ell_i=None, Ev_of_ell_j=None): # Inequality parameters
    
    # Define the leisure utility function
    def v_of_ell(x, epsilon=frisch):
        theta = 1.1443127394313553 * (1 - 0.353) / (1 - 0.7113406008484311)**(1 / epsilon + 1)
        return -(theta * epsilon / (1 + epsilon)) * (1 - x)**((1 + epsilon) / epsilon)
    
    # Restrict on selected ages
    S_i = S_i[age_min:age_max + 1] / S_i[age_min]
    S_j = S_j[age_min:age_max + 1] / S_j[age_min]
    I_i = I_i[age_min:age_max + 1]
    I_j = I_j[age_min:age_max + 1]
    c_i_bar = c_i_bar[age_min:age_max + 1]
    c_j_bar = c_j_bar[age_min:age_max + 1]
    ell_i_bar = ell_i_bar[age_min:age_max + 1]
    ell_j_bar = ell_j_bar[age_min:age_max + 1]
    S_intercept = S_intercept[age_min_intercept:age_max_intercept + 1] / S_intercept[age_min_intercept]
    I_intercept = I_intercept[age_min_intercept:age_max_intercept + 1]
    c_intercept = c_intercept[age_min_intercept:age_max_intercept + 1]
    ell_intercept = ell_intercept[age_min_intercept:age_max_intercept + 1]

    # Define the sequence of discount rates
    beta_age = beta**linspace(0, age_max - age_min, age_max - age_min + 1)
    beta_age_intercept = beta**linspace(0, age_max_intercept - age_min_intercept, age_max_intercept - age_min_intercept + 1)

    # Define the sequence of growth rates
    g_age = g * linspace(0, age_max - age_min, age_max - age_min + 1)
    g_age_intercept = g * linspace(0, age_max_intercept - age_min_intercept, age_max_intercept - age_min_intercept + 1)

    # Calculate the intercept
    u_bar = (vsl / c_nominal - dot(beta_age_intercept * S_intercept * (1 - I_intercept), log(c_intercept) + v_of_ell(ell_intercept) + g_age_intercept)) / sum(beta_age_intercept * S_intercept * (1 - I_intercept))

    # Define the lower case survival rates
    s_i = beta_age * S_i * (1 - I_i) / sum(beta_age * S_i * (1 - I_i))
    s_j = beta_age * S_j * (1 - I_j) / sum(beta_age * S_j * (1 - I_j))
    Delta_s_EV = beta_age * (S_j - S_i) * (1 - I_j) / sum(beta_age * S_i * (1 - I_i))
    Delta_s_CV = beta_age * (S_j - S_i) * (1 - I_i) / sum(beta_age * S_j * (1 - I_j))
    Delta_i_EV = beta_age * (I_i - I_j) * S_i / sum(beta_age * S_i * (1 - I_i))
    Delta_i_CV = beta_age * (I_i - I_j) * S_j / sum(beta_age * S_j * (1 - I_j))

    # Calculate consumption-equivalent welfare with the inequality terms
    if inequality:
        # Restrict on selected ages for the inequality terms
        Elog_of_c_i = Elog_of_c_i[age_min:age_max + 1]
        Elog_of_c_j = Elog_of_c_j[age_min:age_max + 1]
        Elog_of_c_i_nd = Elog_of_c_i_nd[age_min:age_max + 1]
        Elog_of_c_j_nd = Elog_of_c_j_nd[age_min:age_max + 1]
        c_i_bar_nd = c_i_bar_nd[age_min:age_max + 1]
        c_j_bar_nd = c_j_bar_nd[age_min:age_max + 1]
        Ev_of_ell_i = Ev_of_ell_i[age_min:age_max + 1]
        Ev_of_ell_j = Ev_of_ell_j[age_min:age_max + 1]

        # Calculate flow utility for each group
        flow_EV = u_bar + g_age + Elog_of_c_j + Ev_of_ell_j
        flow_CV = u_bar + g_age + Elog_of_c_i + Ev_of_ell_i

        # Calculate the EV and CV life expectancy terms, and average them
        LE_EV = sum(Delta_s_EV * flow_EV)
        LE_CV = sum(Delta_s_CV * flow_CV)
        LE = (LE_EV + LE_CV) / 2

        # Calculate the EV and CV incarceration terms, and average them
        I_EV = sum(Delta_i_EV * flow_EV)
        I_CV = sum(Delta_i_CV * flow_CV)
        I = (I_EV + I_CV) / 2

        # Calculate the EV and CV consumption terms, and average them
        C_EV = log(sum(s_i * c_j_bar * exp(g_age))) - log(sum(s_i * c_i_bar * exp(g_age)))
        C_CV = log(sum(s_j * c_j_bar * exp(g_age))) - log(sum(s_j * c_i_bar * exp(g_age)))
        C = (C_EV + C_CV) / 2

        # Calculate the EV and CV consumption inequality terms, and average them
        CI_EV = sum(s_i * (Elog_of_c_j_nd - Elog_of_c_i_nd)) - (log(sum(s_i * c_j_bar_nd * exp(g_age))) - log(sum(s_i * c_i_bar_nd * exp(g_age))))
        CI_CV = sum(s_j * (Elog_of_c_j_nd - Elog_of_c_i_nd)) - (log(sum(s_j * c_j_bar_nd * exp(g_age))) - log(sum(s_j * c_i_bar_nd * exp(g_age))))
        CI = (CI_EV + CI_CV) / 2

        # Calculate the EV and CV leisure terms, and average them
        L_EV = v_of_ell(sum(s_i * ell_j_bar)) - v_of_ell(sum(s_i * ell_i_bar))
        L_CV = v_of_ell(sum(s_j * ell_j_bar)) - v_of_ell(sum(s_j * ell_i_bar))
        L = (L_EV + L_CV) / 2

        # Calculate the EV and CV leisure inequality terms, and average them
        LI_EV = sum(s_i * (Ev_of_ell_j - Ev_of_ell_i)) - L_EV
        LI_CV = sum(s_j * (Ev_of_ell_j - Ev_of_ell_i)) - L_CV
        LI = (LI_EV + LI_CV) / 2

        # Calculate the EV and CV consumption-equivalent welfare, and average them
        log_lambda_EV = LE_EV + I_EV + C_EV + CI_EV + L_EV + LI_EV
        log_lambda_CV = LE_CV + I_CV + C_CV + CI_CV + L_CV + LI_CV
        log_lambda = (log_lambda_EV + log_lambda_CV) / 2

        # Store the results in a dictionary
        d = {'LE_CV':         LE_CV,
             'LE_EV':         LE_EV,
             'LE':            LE,
             'I_CV':          I_CV,
             'I_EV':          I_EV,
             'I':             I,
             'C_CV':          C_CV,
             'C_EV':          C_EV,
             'C':             C,
             'CI_CV':         CI_CV,
             'CI_EV':         CI_EV,
             'CI':            CI,
             'L_CV':          L_CV,
             'L_EV':          L_EV,
             'L':             L,
             'LI_CV':         LI_CV,
             'LI_EV':         LI_EV,
             'LI':            LI,
             'log_lambda_CV': log_lambda_CV,
             'log_lambda_EV': log_lambda_EV,
             'log_lambda':    log_lambda,
             'u_bar':         u_bar}
    else:
        # Compute flow utility for each group
        flow_EV = u_bar + log(sum(s_j * c_j_bar * exp(g_age))) + v_of_ell(sum(s_j * ell_j_bar))
        flow_CV = u_bar + log(sum(s_i * c_i_bar * exp(g_age))) + v_of_ell(sum(s_i * ell_i_bar))

        # Calculate the EV and CV life expectancy terms, and average them
        LE_EV = sum(Delta_s_EV) * flow_EV
        LE_CV = sum(Delta_s_CV) * flow_CV
        LE = (LE_EV + LE_CV) / 2

        # Calculate the EV and CV incarceration terms, and average them
        I_EV = sum(Delta_i_EV * flow_EV)
        I_CV = sum(Delta_i_CV * flow_CV)
        I = (I_EV + I_CV) / 2

        # Calculate the EV and CV consumption terms, and average them
        C_EV = log(sum(s_i * c_j_bar * exp(g_age))) - log(sum(s_i * c_i_bar * exp(g_age)))
        C_CV = log(sum(s_j * c_j_bar * exp(g_age))) - log(sum(s_j * c_i_bar * exp(g_age)))
        C = (C_EV + C_CV) / 2

        # Calculate the EV and CV leisure terms, and average them
        L_EV = v_of_ell(sum(s_i * ell_j_bar)) - v_of_ell(sum(s_i * ell_i_bar))
        L_CV = v_of_ell(sum(s_j * ell_j_bar)) - v_of_ell(sum(s_j * ell_i_bar))
        L = (L_EV + L_CV) / 2

        # Calculate the EV and CV consumption-equivalent welfare, and average them
        log_lambda_EV = LE_EV + C_EV + L_EV
        log_lambda_CV = LE_CV + C_CV + L_CV
        log_lambda = (log_lambda_EV + log_lambda_CV) / 2

        # Store the results in a dictionary
        d = {'LE_CV':         LE_CV,
             'LE_EV':         LE_EV,
             'LE':            LE,
             'I_CV':          I_CV,
             'I_EV':          I_EV,
             'I':             I,
             'C_CV':          C_CV,
             'C_EV':          C_EV,
             'C':             C,
             'L_CV':          L_CV,
             'L_EV':          L_EV,
             'L':             L,
             'log_lambda_CV': log_lambda_CV,
             'log_lambda_EV': log_lambda_EV,
             'log_lambda':    log_lambda,
             'u_bar':         u_bar}

    # Return the dictionary
    return d

# Define the growth consumption-equivalent welfare calculation function
def cew_growth(S_i=None, S_j=None, I_i=None, I_j=None, c_i_bar=None, c_j_bar=None, ell_i_bar=None, ell_j_bar=None, beta=1, g=0, age_min=0, age_max=100, T=None, # Standard parameters
               S_intercept=None, I_intercept=None, c_intercept=None, ell_intercept=None, vsl=7.4e6, c_nominal=None, age_min_intercept=40, age_max_intercept=100, # Intercept parameters
               inequality=False, c_i_bar_nd=None, c_j_bar_nd=None, Elog_of_c_i=None, Elog_of_c_j=None, Elog_of_c_i_nd=None, Elog_of_c_j_nd=None, Ev_of_ell_i=None, Ev_of_ell_j=None): # Inequality parameters
    # Restrict on selected ages
    S_i = S_i[age_min:age_max + 1] / S_i[age_min]
    S_j = S_j[age_min:age_max + 1] / S_j[age_min]
    I_i = I_i[age_min:age_max + 1]
    I_j = I_j[age_min:age_max + 1]
    c_i_bar = c_i_bar[age_min:age_max + 1]
    c_j_bar = c_j_bar[age_min:age_max + 1]
    ell_i_bar = ell_i_bar[age_min:age_max + 1]
    ell_j_bar = ell_j_bar[age_min:age_max + 1]
    S_intercept = S_intercept[age_min_intercept:age_max_intercept + 1] / S_intercept[age_min_intercept]
    I_intercept = I_intercept[age_min_intercept:age_max_intercept + 1]
    c_intercept = c_intercept[age_min_intercept:age_max_intercept + 1]
    ell_intercept = ell_intercept[age_min_intercept:age_max_intercept + 1]

    # Define the sequence of discount rates
    beta_age = beta**linspace(0, age_max - age_min, age_max - age_min + 1)
    beta_age_intercept = beta**linspace(0, age_max_intercept - age_min_intercept, age_max_intercept - age_min_intercept + 1)

    # Define the sequence of growth rates
    g_age = g * linspace(0, age_max - age_min, age_max - age_min + 1)
    g_age_intercept = g * linspace(0, age_max_intercept - age_min_intercept, age_max_intercept - age_min_intercept + 1)

    # Calculate the intercept
    u_bar = (vsl / c_nominal - dot(beta_age_intercept * S_intercept * (1 - I_intercept), log(c_intercept) + v_of_ell(ell_intercept) + g_age_intercept)) / sum(beta_age_intercept * S_intercept * (1 - I_intercept))

    # Define the lower case survival rates
    s_i = beta_age * S_i * (1 - I_i) / sum(beta_age * S_i * (1 - I_i))
    s_j = beta_age * S_j * (1 - I_j) / sum(beta_age * S_j * (1 - I_j))
    Delta_s_EV = beta_age * (S_j - S_i) * (1 - I_j) / sum(beta_age * S_i * (1 - I_i))
    Delta_s_CV = beta_age * (S_j - S_i) * (1 - I_i) / sum(beta_age * S_j * (1 - I_j))
    Delta_i_EV = beta_age * (I_i - I_j) * S_i / sum(beta_age * S_i * (1 - I_i))
    Delta_i_CV = beta_age * (I_i - I_j) * S_j / sum(beta_age * S_j * (1 - I_j))

    # Calculate consumption-equivalent welfare with the inequality terms
    if inequality:
        # Restrict on selected ages for the inequality terms
        Elog_of_c_i = Elog_of_c_i[age_min:age_max + 1]
        Elog_of_c_j = Elog_of_c_j[age_min:age_max + 1]
        Elog_of_c_i_nd = Elog_of_c_i_nd[age_min:age_max + 1]
        Elog_of_c_j_nd = Elog_of_c_j_nd[age_min:age_max + 1]
        c_i_bar_nd = c_i_bar_nd[age_min:age_max + 1]
        c_j_bar_nd = c_j_bar_nd[age_min:age_max + 1]
        Ev_of_ell_i = Ev_of_ell_i[age_min:age_max + 1]
        Ev_of_ell_j = Ev_of_ell_j[age_min:age_max + 1]

        # Calculate flow utility for each group
        flow_EV = u_bar + g_age + Elog_of_c_j + Ev_of_ell_j
        flow_CV = u_bar + g_age + Elog_of_c_i + Ev_of_ell_i

        # Calculate the EV and CV life expectancy terms, and average them
        LE_EV = sum(Delta_s_EV * flow_EV) / T
        LE_CV = sum(Delta_s_CV * flow_CV) / T
        LE = (LE_EV + LE_CV) / 2

        # Calculate the EV and CV incarceration terms, and average them
        I_EV = sum(Delta_i_EV * flow_EV) / T
        I_CV = sum(Delta_i_CV * flow_CV) / T
        I = (I_EV + I_CV) / 2

        # Calculate the EV and CV consumption terms, and average them
        C_EV = (log(sum(s_i * c_j_bar)) - log(sum(s_i * c_i_bar))) / T
        C_CV = (log(sum(s_j * c_j_bar)) - log(sum(s_j * c_i_bar))) / T
        C = (C_EV + C_CV) / 2

        # Calculate the EV and CV consumption inequality terms, and average them
        CI_EV = (sum(s_i * (Elog_of_c_j_nd - Elog_of_c_i_nd)) - (log(sum(s_i * c_j_bar_nd)) - log(sum(s_i * c_i_bar_nd)))) / T
        CI_CV = (sum(s_j * (Elog_of_c_j_nd - Elog_of_c_i_nd)) - (log(sum(s_j * c_j_bar_nd)) - log(sum(s_j * c_i_bar_nd)))) / T
        CI = (CI_EV + CI_CV) / 2

        # Calculate the EV and CV leisure terms, and average them
        L_EV = (v_of_ell(sum(s_i * ell_j_bar)) - v_of_ell(sum(s_i * ell_i_bar))) / T
        L_CV = (v_of_ell(sum(s_j * ell_j_bar)) - v_of_ell(sum(s_j * ell_i_bar))) / T
        L = (L_EV + L_CV) / 2

        # Calculate the EV and CV leisure inequality terms, and average them
        LI_EV = (sum(s_i * (Ev_of_ell_j - Ev_of_ell_i)) - (v_of_ell(sum(s_i * ell_j_bar)) - v_of_ell(sum(s_i * ell_i_bar)))) / T
        LI_CV = (sum(s_j * (Ev_of_ell_j - Ev_of_ell_i)) - (v_of_ell(sum(s_j * ell_j_bar)) - v_of_ell(sum(s_j * ell_i_bar)))) / T
        LI = (LI_EV + LI_CV) / 2

        # Calculate the EV and CV consumption-equivalent welfare, and average them
        log_lambda_EV = LE_EV + I_EV + C_EV + CI_EV + L_EV + LI_EV
        log_lambda_CV = LE_CV + I_CV + C_CV + CI_CV + L_CV + LI_CV
        log_lambda = (log_lambda_EV + log_lambda_CV) / 2

        # Store the results in a dictionary
        d = {'LE_CV':         LE_CV,
             'LE_EV':         LE_EV,
             'LE':            LE,
             'I_CV':          I_CV,
             'I_EV':          I_EV,
             'I':             I,
             'C_CV':          C_CV,
             'C_EV':          C_EV,
             'C':             C,
             'CI_CV':         CI_CV,
             'CI_EV':         CI_EV,
             'CI':            CI,
             'L_CV':          L_CV,
             'L_EV':          L_EV,
             'L':             L,
             'LI_CV':         LI_CV,
             'LI_EV':         LI_EV,
             'LI':            LI,
             'log_lambda_CV': log_lambda_CV,
             'log_lambda_EV': log_lambda_EV,
             'log_lambda':    log_lambda,
             'u_bar':         u_bar}
    else:
        # Compute flow utility for each group
        flow_EV = u_bar + log(sum(s_j * c_j_bar * exp(g_age))) + v_of_ell(sum(s_j * ell_j_bar))
        flow_CV = u_bar + log(sum(s_i * c_i_bar * exp(g_age))) + v_of_ell(sum(s_i * ell_i_bar))

        # Calculate the EV and CV life expectancy terms, and average them
        LE_EV = (sum(Delta_s_EV) * flow_EV) / T
        LE_CV = (sum(Delta_s_CV) * flow_CV) / T
        LE = (LE_EV + LE_CV) / 2

        # Calculate the EV and CV incarceration terms, and average them
        I_EV = sum(Delta_i_EV * flow_EV) / T
        I_CV = sum(Delta_i_CV * flow_CV) / T
        I = (I_EV + I_CV) / 2

        # Calculate the EV and CV consumption terms, and average them
        C_EV = (log(sum(s_i * c_j_bar)) - log(sum(s_i * c_i_bar))) / T
        C_CV = (log(sum(s_j * c_j_bar)) - log(sum(s_j * c_i_bar))) / T
        C = (C_EV + C_CV) / 2

        # Calculate the EV and CV leisure terms, and average them
        L_EV = (v_of_ell(sum(s_i * ell_j_bar)) - v_of_ell(sum(s_i * ell_i_bar))) / T
        L_CV = (v_of_ell(sum(s_j * ell_j_bar)) - v_of_ell(sum(s_j * ell_i_bar))) / T
        L = (L_EV + L_CV) / 2

        # Calculate the EV and CV consumption-equivalent welfare, and average them
        log_lambda_EV = LE_EV + I_EV + C_EV + L_EV
        log_lambda_CV = LE_CV + I_CV + C_CV + L_CV
        log_lambda = (log_lambda_EV + log_lambda_CV) / 2

        # Store the results in a dictionary
        d = {'LE_CV':         LE_CV,
             'LE_EV':         LE_EV,
             'LE':            LE,
             'I_CV':          I_CV,
             'I_EV':          I_EV,
             'I':             I,
             'C_CV':          C_CV,
             'C_EV':          C_EV,
             'C':             C,
             'L_CV':          L_CV,
             'L_EV':          L_EV,
             'L':             L,
             'log_lambda_CV': log_lambda_CV,
             'log_lambda_EV': log_lambda_EV,
             'log_lambda':    log_lambda,
             'u_bar':         u_bar}

    # Return the dictionary
    return d

# Define the level consumption-equivalent welfare calculation function
def cew_level_gamma(S_i=None, S_j=None, I_i=None, I_j=None, Eu_of_c_and_ell_i=None, Eu_of_c_and_ell_j=None, age_min=0, age_max=100, gamma=2, epsilon=1, theta=8.795358530547285, # Standard parameters
                    S_intercept=None, I_intercept=None, c_intercept=None, ell_intercept=None, vsl=7.4e6, c_nominal=None, ell_bar=None, age_min_intercept=40, age_max_intercept=100): # Intercept parameters
    # Restrict on selected ages
    S_i = S_i[age_min:age_max + 1] / S_i[age_min]
    S_j = S_j[age_min:age_max + 1] / S_j[age_min]
    I_i = I_i[age_min:age_max + 1]
    I_j = I_j[age_min:age_max + 1]
    S_intercept = S_intercept[age_min_intercept:age_max_intercept + 1] / S_intercept[age_min_intercept]
    I_intercept = I_intercept[age_min_intercept:age_max_intercept + 1]
    c_intercept = c_intercept[age_min_intercept:age_max_intercept + 1]
    ell_intercept = ell_intercept[age_min_intercept:age_max_intercept + 1]
    Eu_of_c_and_ell_i = Eu_of_c_and_ell_i[age_min:age_max + 1]
    Eu_of_c_and_ell_j = Eu_of_c_and_ell_j[age_min:age_max + 1]

    # Define the flow utility function from consumption and leisure
    def u(c, ell, gamma=gamma, epsilon=1, theta=8.795358530547285):
        return c**(1 - gamma) * (1 + (gamma - 1) * theta * epsilon * (1 - ell)**((1 + epsilon) / epsilon) / (1 + epsilon))**gamma / (1 - gamma)

    # Define the marginal utility function from consumption
    def du_dc(c, ell, gamma=gamma, epsilon=1, theta=8.795358530547285):
        return (1 + (gamma - 1) * theta * epsilon * (1 - ell)**((1 + epsilon) / epsilon) / (1 + epsilon))**gamma / c**gamma

    # Calculate the intercept
    u_bar = (vsl * du_dc(c_nominal, ell_bar) - dot(S_intercept * (1 - I_intercept), u(c_intercept, ell_intercept) - 1 / (1 - gamma))) / sum(S_intercept * (1 - I_intercept))

    # Define the EV and CV consumption-equivalent welfare functions
    def EV(x):
        return sum(S_i * (1 - I_i) * (u_bar + x**(1 - gamma) * Eu_of_c_and_ell_i - 1 / (1 - gamma))) - sum(S_j * (1 - I_j) * (u_bar + Eu_of_c_and_ell_j - 1 / (1 - gamma)))
    def CV(x):
        return sum(S_i * (1 - I_i) * (u_bar + Eu_of_c_and_ell_i - 1 / (1 - gamma))) - sum(S_j * (1 - I_j) * (u_bar + x**(gamma - 1) * Eu_of_c_and_ell_j - 1 / (1 - gamma)))

    # Calculate the EV and CV consumption-equivalent welfare, and average them
    solution_EV = optimize.root(EV, [1.0])
    solution_CV = optimize.root(CV, [1.0])

    # Store the results in a dictionary
    d = {'lambda_EV':      solution_EV.x,
         'lambda_CV':      solution_CV.x,
         'lambda_average': exp((log(solution_EV.x) + log(solution_CV.x)) / 2),
         'u_bar':          u_bar}
         
    # Return the dictionary
    return d

# Define the level consumption-equivalent welfare calculation function with the morbidity adjustment
def cew_level_incarceration(S_i=None, S_j=None, I_i=None, I_j=None, c_i_bar=None, c_j_bar=None, ell_i_bar=None, ell_j_bar=None, age_min=0, age_max=100, # Standard parameters
                            S_intercept=None, I_intercept=None, c_intercept=None, c_intercept_I=None, ell_intercept=None, ell_intercept_I=None, vsl=7.4e6, c_nominal=None, age_min_intercept=40, age_max_intercept=100, # Intercept parameters
                            c_i_bar_nd=None, c_j_bar_nd=None, Elog_of_c_i=None, Elog_of_c_j=None, Elog_of_c_i_nd=None, Elog_of_c_j_nd=None, Ev_of_ell_i=None, Ev_of_ell_j=None, # Inequality parameters
                            Elog_of_c_I=None, Ev_of_ell_I=None, incarceration_parameter=None): # Incarceration adjustment parameters
    
    # Restrict on selected ages
    S_i = S_i[age_min:age_max + 1] / S_i[age_min]
    S_j = S_j[age_min:age_max + 1] / S_j[age_min]
    I_i = I_i[age_min:age_max + 1]
    I_j = I_j[age_min:age_max + 1]
    c_i_bar = c_i_bar[age_min:age_max + 1]
    c_j_bar = c_j_bar[age_min:age_max + 1]
    ell_i_bar = ell_i_bar[age_min:age_max + 1]
    ell_j_bar = ell_j_bar[age_min:age_max + 1]
    Elog_of_c_i = Elog_of_c_i[age_min:age_max + 1]
    Elog_of_c_j = Elog_of_c_j[age_min:age_max + 1]
    Elog_of_c_i_nd = Elog_of_c_i_nd[age_min:age_max + 1]
    Elog_of_c_j_nd = Elog_of_c_j_nd[age_min:age_max + 1]
    c_i_bar_nd = c_i_bar_nd[age_min:age_max + 1]
    c_j_bar_nd = c_j_bar_nd[age_min:age_max + 1]
    Ev_of_ell_i = Ev_of_ell_i[age_min:age_max + 1]
    Ev_of_ell_j = Ev_of_ell_j[age_min:age_max + 1]
    Elog_of_c_I = Elog_of_c_I[age_min:age_max + 1]
    Ev_of_ell_I = Ev_of_ell_I[age_min:age_max + 1]
    S_intercept = S_intercept[age_min_intercept:age_max_intercept + 1] / S_intercept[age_min_intercept]
    I_intercept = I_intercept[age_min_intercept:age_max_intercept + 1]
    c_intercept = c_intercept[age_min_intercept:age_max_intercept + 1]
    c_intercept_I = c_intercept_I[age_min_intercept:age_max_intercept + 1]
    ell_intercept = ell_intercept[age_min_intercept:age_max_intercept + 1]
    ell_intercept_I = ell_intercept_I[age_min_intercept:age_max_intercept + 1]

    # Calculate the intercept
    u_bar = (vsl / c_nominal - dot(S_intercept * (1 - I_intercept), log(c_intercept) + v_of_ell(ell_intercept)) \
                             - incarceration_parameter * dot(S_intercept * I_intercept, log(c_intercept_I) + v_of_ell(ell_intercept_I))) \
                             / sum(S_intercept * (1 - I_intercept * (1 - incarceration_parameter)))

    # Calculate the EV and CV incarceration terms, and average them
    I_EV = incarceration_parameter * sum((u_bar + Elog_of_c_I + Ev_of_ell_I) * (S_j * I_j - S_i * I_i)) / sum(S_i * (1 - I_i))
    I_CV = incarceration_parameter * sum((u_bar + Elog_of_c_I + Ev_of_ell_I) * (S_j * I_j - S_i * I_i)) / sum(S_j * (1 - I_j))
    I = (I_EV + I_CV) / 2

    # Adjust the survival rates for incarceration
    S_i = S_i * (1 - I_i)
    S_j = S_j * (1 - I_j)

    # Define the lower case survival rates
    s_i = S_i / sum(S_i)
    s_j = S_j / sum(S_j)
    Delta_s_EV = (S_j - S_i) / sum(S_i)
    Delta_s_CV = (S_j - S_i) / sum(S_j)

    # Calculate flow utility for each group
    flow_EV = u_bar + Elog_of_c_j + Ev_of_ell_j
    flow_CV = u_bar + Elog_of_c_i + Ev_of_ell_i

    # Calculate the EV and CV life expectancy terms, and average them
    LE_EV = sum(Delta_s_EV * flow_EV)
    LE_CV = sum(Delta_s_CV * flow_CV)
    LE = (LE_EV + LE_CV) / 2

    # Calculate the EV and CV consumption terms, and average them
    C_EV = log(sum(s_i * c_j_bar)) - log(sum(s_i * c_i_bar))
    C_CV = log(sum(s_j * c_j_bar)) - log(sum(s_j * c_i_bar))
    C = (C_EV + C_CV) / 2

    # Calculate the EV and CV consumption inequality terms, and average them
    CI_EV = sum(s_i * (Elog_of_c_j_nd - Elog_of_c_i_nd)) - (log(sum(s_i * c_j_bar_nd)) - log(sum(s_i * c_i_bar_nd)))
    CI_CV = sum(s_j * (Elog_of_c_j_nd - Elog_of_c_i_nd)) - (log(sum(s_j * c_j_bar_nd)) - log(sum(s_j * c_i_bar_nd)))
    CI = (CI_EV + CI_CV) / 2

    # Calculate the EV and CV leisure terms, and average them
    L_EV = v_of_ell(sum(s_i * ell_j_bar)) - v_of_ell(sum(s_i * ell_i_bar))
    L_CV = v_of_ell(sum(s_j * ell_j_bar)) - v_of_ell(sum(s_j * ell_i_bar))
    L = (L_EV + L_CV) / 2

    # Calculate the EV and CV leisure inequality terms, and average them
    LI_EV = sum(s_i * (Ev_of_ell_j - Ev_of_ell_i)) - L_EV
    LI_CV = sum(s_j * (Ev_of_ell_j - Ev_of_ell_i)) - L_CV
    LI = (LI_EV + LI_CV) / 2

    # Calculate the EV and CV consumption-equivalent welfare, and average them
    log_lambda_EV = LE_EV + C_EV + CI_EV + L_EV + LI_EV + I_EV
    log_lambda_CV = LE_CV + C_CV + CI_CV + L_CV + LI_CV + I_CV
    log_lambda = (log_lambda_EV + log_lambda_CV) / 2

    # Store the results in a dictionary
    d = {'LE_CV':         LE_CV,
         'LE_EV':         LE_EV,
         'LE':            LE,
         'C_CV':          C_CV,
         'C_EV':          C_EV,
         'C':             C,
         'CI_CV':         CI_CV,
         'CI_EV':         CI_EV,
         'CI':            CI,
         'L_CV':          L_CV,
         'L_EV':          L_EV,
         'L':             L,
         'LI_CV':         LI_CV,
         'LI_EV':         LI_EV,
         'LI':            LI,
         'I_CV':          I_CV,
         'I_EV':          I_EV,
         'I':             I,
         'log_lambda_CV': log_lambda_CV,
         'log_lambda_EV': log_lambda_EV,
         'log_lambda':    log_lambda,
         'u_bar':         u_bar}

    # Return the dictionary
    return d