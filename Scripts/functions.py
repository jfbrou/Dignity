# Import libraries
from numpy import *
import pandas as pd
import itertools as it
import statsmodels.api as sm

# Define a function to create a data frame of the right form
def expand(d):
    rows = it.product(*d.values())
    return pd.DataFrame.from_records(rows, columns=d.keys())

# Define the leisure utility function
def v_of_ell(x, ϵ=1.0, θ=14.2):
    return -(θ * ϵ / (1 + ϵ)) * (1 - x)**((1 + ϵ) / ϵ)

# Define the Beer population interpolation function
def beer_population(x):
    # Convert the population series to a numpy array
    x = x.to_numpy()

    # Aggregate population in 5-year age groups
    P5 = zeros(21)
    for i in range(20):
        P5[i] = sum(x[5 * i:5 * i + 5])
    P5[20] = x[100]

    # Define the coefficients
    B = array([[0.3333, -0.1636, -0.0210, 0.0796, -0.0283],
               [0.2595, -0.0780, 0.0130, 0.0100, -0.0045],
               [0.1924, 0.0064, 0.0184, -0.0256, 0.0084],
               [0.1329, 0.0844, 0.0054, -0.0356, 0.0129],
               [0.0819, 0.1508, -0.0158, -0.0284, 0.0115],
               [0.0404, 0.2000, -0.0344, -0.0128, 0.0068],
               [0.0093, 0.2268, -0.0402, 0.0028, 0.0013],
               [-0.0108, 0.2272, -0.0248, 0.0112, -0.0028],
               [-0.0198, 0.1992, 0.0172, 0.0072, -0.0038],
               [-0.0191, 0.1468, 0.0822, -0.0084, -0.0015],
               [-0.0117, 0.0804, 0.1570, -0.0284, 0.0027],
               [-0.0020, 0.0160, 0.2200, -0.0400, 0.0060],
               [0.0050, -0.0280, 0.2460, -0.0280, 0.0050],
               [0.0060, -0.0400, 0.2200, 0.0160, -0.0020],
               [0.0027, -0.0284, 0.1570, 0.0804, -0.0117],
               [-0.0015, -0.0084, 0.0822, 0.1468, -0.0191],
               [-0.0038, 0.0072, 0.0172, 0.1992, -0.0198],
               [-0.0028, 0.0112, -0.0248, 0.2272, -0.0108],
               [0.0013, 0.0028, -0.0402, 0.2268, 0.0093],
               [0.0068, -0.0128, -0.0344, 0.2000, 0.0404]])

    # Interpolate population by single years of age
    P = zeros(100)
    P[:10] = dot(B[:10, :], P5[:5])
    for i in range(17):
        P[10 + 5 * i:15 + 5 * i] = dot(B[10:15, :], P5[i:i + 5])
    P[95:] = dot(B[15:, :], P5[16:])

    # Return the single years of age smoothed population estimates
    return append(P, nan)

# Define the Beer death interpolation function
def beer_death(x):
    # Convert the death series to a numpy array
    x = x.to_numpy()

    # Aggregate deaths in 5-year age groups
    D5 = zeros(21)
    D5[0] = 2.45580 * sum(x[1:3]) - 0.59332 * sum(x[5:10]) - 0.01965 * sum(x[10:15]) - 0.22004 * sum(x[15:20]) - 0.08055 * sum(x[20:25])
    for i in range(1, 20):
        D5[i] = sum(x[5 * i:5 * i + 5])
    D5[20] = x[100]

    # Define the coefficients
    B = array([[0.3333, -0.1636, -0.0210, 0.0796, -0.0283],
               [0.2595, -0.0780, 0.0130, 0.0100, -0.0045],
               [0.1924, 0.0064, 0.0184, -0.0256, 0.0084],
               [0.1329, 0.0844, 0.0054, -0.0356, 0.0129],
               [0.0819, 0.1508, -0.0158, -0.0284, 0.0115],
               [0.0404, 0.2000, -0.0344, -0.0128, 0.0068],
               [0.0093, 0.2268, -0.0402, 0.0028, 0.0013],
               [-0.0108, 0.2272, -0.0248, 0.0112, -0.0028],
               [-0.0198, 0.1992, 0.0172, 0.0072, -0.0038],
               [-0.0191, 0.1468, 0.0822, -0.0084, -0.0015],
               [-0.0117, 0.0804, 0.1570, -0.0284, 0.0027],
               [-0.0020, 0.0160, 0.2200, -0.0400, 0.0060],
               [0.0050, -0.0280, 0.2460, -0.0280, 0.0050],
               [0.0060, -0.0400, 0.2200, 0.0160, -0.0020],
               [0.0027, -0.0284, 0.1570, 0.0804, -0.0117],
               [-0.0015, -0.0084, 0.0822, 0.1468, -0.0191],
               [-0.0038, 0.0072, 0.0172, 0.1992, -0.0198],
               [-0.0028, 0.0112, -0.0248, 0.2272, -0.0108],
               [0.0013, 0.0028, -0.0402, 0.2268, 0.0093],
               [0.0068, -0.0128, -0.0344, 0.2000, 0.0404]])

    # Interpolate deaths by single years of age
    D = zeros(100)
    D[:5] = x[:5]
    D[5:10] = dot(B[5:10, :], D5[:5])
    for i in range(17):
        D[10 + 5 * i:15 + 5 * i] = dot(B[10:15, :], D5[i:i + 5])
    D[95:] = dot(B[15:, :], D5[16:])

    # Return the single years of age smoothed death estimates
    return append(D, nan)

# Define the consumption and leisure interpolation/extrapolation function
def filter(x, penalty):
    # Linearly interpolate the missing values coming before the oldest available age
    x = x.interpolate(limit_direction='backward')

    # HP-filter the resulting series
    x[x.notna()] = sm.tsa.filters.hpfilter(x[x.notna()], penalty)[1]

    # Keep the resulting series constant for missing ages coming after the oldest available age
    return x.interpolate(method='ffill', limit_direction='forward')

# Define the level consumption-equivalent welfare calculation function
def cew_level(S_i=None, S_j=None, c_i_bar=None, c_j_bar=None, ell_i_bar=None, ell_j_bar=None, beta=1, g=0, age_min=0, age_max=100, # Standard parameters
              S_intercept=None, c_intercept=None, ell_intercept=None, vsl=7.4e6, c_nominal=None, age_min_intercept=40, age_max_intercept=100, # Intercept parameters
              inequality=False, c_i_bar_nd=None, c_j_bar_nd=None, Elog_of_c_i=None, Elog_of_c_j=None, Elog_of_c_i_nd=None, Elog_of_c_j_nd=None, Ev_of_ell_i=None, Ev_of_ell_j=None): # Inequality parameters
    # Restrict on selected ages
    S_i = S_i[age_min:age_max + 1] / S_i[age_min]
    S_j = S_j[age_min:age_max + 1] / S_j[age_min]
    c_i_bar = c_i_bar[age_min:age_max + 1]
    c_j_bar = c_j_bar[age_min:age_max + 1]
    ell_i_bar = ell_i_bar[age_min:age_max + 1]
    ell_j_bar = ell_j_bar[age_min:age_max + 1]
    S_intercept = S_intercept[age_min_intercept:age_max_intercept + 1] / S_intercept[age_min_intercept]
    c_intercept = c_intercept[age_min_intercept:age_max_intercept + 1]
    ell_intercept = ell_intercept[age_min_intercept:age_max_intercept + 1]

    # Define the sequence of discount rates
    beta_age = beta**linspace(age_min, age_max, age_max - age_min + 1)
    beta_age_intercept = beta**linspace(age_min_intercept, age_max_intercept, age_max_intercept - age_min_intercept + 1)

    # Define the sequence of growth rates
    g_age = g * linspace(age_min, age_max, age_max - age_min + 1)
    g_age_intercept = g * linspace(age_min_intercept, age_max_intercept, age_max_intercept - age_min_intercept + 1)

    # Calculate the intercept
    u_bar = (vsl / c_nominal - dot(beta_age_intercept * S_intercept, log(c_intercept) + v_of_ell(ell_intercept) + g_age_intercept)) / sum(beta_age_intercept * S_intercept)

    # Define the lower case survival rates
    s_i = beta_age * S_i / sum(beta_age * S_i)
    s_j = beta_age * S_j / sum(beta_age * S_j)
    Delta_s_EV = beta_age * (S_j - S_i) / sum(beta_age * S_i)
    Delta_s_CV = beta_age * (S_j - S_i) / sum(beta_age * S_j)

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
        log_lambda_EV = LE_EV + C_EV + CI_EV + L_EV + LI_EV
        log_lambda_CV = LE_CV + C_CV + CI_CV + L_CV + LI_CV
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

# Define the level consumption-equivalent welfare calculation function with the morbidity adjustment
def cew_level_morbidity(S_i=None, S_j=None, c_i_bar=None, c_j_bar=None, ell_i_bar=None, ell_j_bar=None, age_min=0, age_max=100, # Standard parameters
                        S_intercept=None, halex_intercept=None, c_intercept=None, ell_intercept=None, vsl=7.4e6, c_nominal=None, age_min_intercept=40, age_max_intercept=100, # Intercept parameters
                        c_i_bar_nd=None, c_j_bar_nd=None, Elog_of_c_i=None, Elog_of_c_j=None, Elog_of_c_i_nd=None, Elog_of_c_j_nd=None, Ev_of_ell_i=None, Ev_of_ell_j=None, # Inequality parameters
                        halex_i=None, halex_j=None, morbidity_parameter=None): # Morbidity adjustment parameters
    # Restrict on selected ages
    S_i = S_i[age_min:age_max + 1]
    S_j = S_j[age_min:age_max + 1]
    c_i_bar = c_i_bar[age_min:age_max + 1]
    c_j_bar = c_j_bar[age_min:age_max + 1]
    ell_i_bar = ell_i_bar[age_min:age_max + 1]
    ell_j_bar = ell_j_bar[age_min:age_max + 1]
    halex_i = halex_i[age_min:age_max + 1]
    halex_j = halex_j[age_min:age_max + 1]
    Elog_of_c_i = Elog_of_c_i[age_min:age_max + 1]
    Elog_of_c_j = Elog_of_c_j[age_min:age_max + 1]
    Elog_of_c_i_nd = Elog_of_c_i_nd[age_min:age_max + 1]
    Elog_of_c_j_nd = Elog_of_c_j_nd[age_min:age_max + 1]
    c_i_bar_nd = c_i_bar_nd[age_min:age_max + 1]
    c_j_bar_nd = c_j_bar_nd[age_min:age_max + 1]
    Ev_of_ell_i = Ev_of_ell_i[age_min:age_max + 1]
    Ev_of_ell_j = Ev_of_ell_j[age_min:age_max + 1]
    S_intercept = S_intercept[age_min_intercept:age_max_intercept + 1]
    c_intercept = c_intercept[age_min_intercept:age_max_intercept + 1]
    ell_intercept = ell_intercept[age_min_intercept:age_max_intercept + 1]
    halex_intercept = halex_intercept[age_min_intercept:age_max_intercept + 1]

    # Calculate the quality of life terms from the HALex
    Q_i = morbidity_parameter + (1 - morbidity_parameter) * halex_i
    Q_j = morbidity_parameter + (1 - morbidity_parameter) * halex_j
    Q_intercept = morbidity_parameter + (1 - morbidity_parameter) * halex_intercept

    # Calculate the intercept
    u_bar = (vsl / c_nominal - dot(S_intercept * Q_intercept, log(c_intercept) + v_of_ell(ell_intercept))) / sum(S_intercept * Q_intercept)

    # Define the lower case survival rates
    s_i = S_i * Q_i / sum(S_i * Q_i)
    s_j = S_j * Q_j / sum(S_j * Q_j)
    Delta_s_EV = (S_j - S_i) * Q_j / sum(S_i * Q_i)
    Delta_s_CV = (S_j - S_i) * Q_i / sum(S_j * Q_j)
    Delta_q_EV = (Q_j - Q_i) * S_i / sum(S_i * Q_i)
    Delta_q_CV = (Q_j - Q_i) * S_j / sum(S_j * Q_j)

    # Calculate flow utility for each group
    flow_EV = u_bar + Elog_of_c_j + Ev_of_ell_j
    flow_CV = u_bar + Elog_of_c_i + Ev_of_ell_i

    # Calculate the EV and CV life expectancy terms, and average them
    LE_EV = sum(Delta_s_EV * flow_EV)
    LE_CV = sum(Delta_s_CV * flow_CV)
    LE = (LE_EV + LE_CV) / 2

    # Calculate the EV and CV morbidity terms, and average them
    M_EV = sum(Delta_q_EV * flow_EV)
    M_CV = sum(Delta_q_CV * flow_CV)
    M = (M_EV + M_CV) / 2

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
    log_lambda_EV = LE_EV + M_EV + C_EV + CI_EV + L_EV + LI_EV
    log_lambda_CV = LE_CV + M_CV + C_CV + CI_CV + L_CV + LI_CV
    log_lambda = (log_lambda_EV + log_lambda_CV) / 2

    # Store the results in a dictionary
    d = {'LE_CV':         LE_CV,
         'LE_EV':         LE_EV,
         'LE':            LE,
         'M_CV':          M_CV,
         'M_EV':          M_EV,
         'M':             M,
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

    # Return the dictionary
    return d

# Define the level consumption-equivalent welfare calculation function with the morbidity adjustment
def cew_level_incarceration(S_i=None, S_j=None, c_i_bar=None, c_j_bar=None, ell_i_bar=None, ell_j_bar=None, age_min=0, age_max=100, # Standard parameters
                            S_intercept=None, I_intercept=None, c_intercept=None, c_intercept_I=None, ell_intercept=None, ell_intercept_I=None, vsl=7.4e6, c_nominal=None, age_min_intercept=40, age_max_intercept=100, # Intercept parameters
                            c_i_bar_nd=None, c_j_bar_nd=None, Elog_of_c_i=None, Elog_of_c_j=None, Elog_of_c_i_nd=None, Elog_of_c_j_nd=None, Ev_of_ell_i=None, Ev_of_ell_j=None, # Inequality parameters
                            Elog_of_c_I=None, Ev_of_ell_I=None, I_i=None, I_j=None, incarceration_parameter=None): # Incarceration adjustment parameters
    # Restrict on selected ages
    S_i = S_i[age_min:age_max + 1]
    S_j = S_j[age_min:age_max + 1]
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
    I_i = I_i[age_min:age_max + 1]
    I_j = I_j[age_min:age_max + 1]
    S_intercept = S_intercept[age_min_intercept:age_max_intercept + 1]
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

# Define the growth consumption-equivalent welfare calculation function
def cew_growth(S_i=None, S_j=None, c_i_bar=None, c_j_bar=None, ell_i_bar=None, ell_j_bar=None, beta=1, g=0, age_min=0, age_max=100, T=None, # Standard parameters
               S_intercept=None, c_intercept=None, ell_intercept=None, vsl=7.4e6, c_nominal=None, age_min_intercept=40, age_max_intercept=100, # Intercept parameters
               inequality=False, c_i_bar_nd=None, c_j_bar_nd=None, Elog_of_c_i=None, Elog_of_c_j=None, Elog_of_c_i_nd=None, Elog_of_c_j_nd=None, Ev_of_ell_i=None, Ev_of_ell_j=None): # Inequality parameters
    # Restrict on selected ages
    S_i = S_i[age_min:age_max + 1]
    S_j = S_j[age_min:age_max + 1]
    c_i_bar = c_i_bar[age_min:age_max + 1]
    c_j_bar = c_j_bar[age_min:age_max + 1]
    ell_i_bar = ell_i_bar[age_min:age_max + 1]
    ell_j_bar = ell_j_bar[age_min:age_max + 1]
    S_intercept = S_intercept[age_min_intercept:age_max_intercept + 1]
    c_intercept = c_intercept[age_min_intercept:age_max_intercept + 1]
    ell_intercept = ell_intercept[age_min_intercept:age_max_intercept + 1]

    # Define the sequence of discount rates
    beta_age = beta**linspace(age_min, age_max, age_max - age_min + 1)
    beta_age_intercept = beta**linspace(age_min_intercept, age_max_intercept, age_max_intercept - age_min_intercept + 1)

    # Define the sequence of growth rates
    g_age = g * linspace(age_min, age_max, age_max - age_min + 1)
    g_age_intercept = g * linspace(age_min_intercept, age_max_intercept, age_max_intercept - age_min_intercept + 1)

    # Calculate the intercept
    u_bar = (vsl / c_nominal - dot(beta_age_intercept * S_intercept, log(c_intercept) + v_of_ell(ell_intercept) + g_age_intercept)) / sum(beta_age_intercept * S_intercept)

    # Define the lower case survival rates
    s_i = beta_age * S_i / sum(beta_age * S_i)
    s_j = beta_age * S_j / sum(beta_age * S_j)
    Delta_s_EV = beta_age * (S_j - S_i) / sum(beta_age * S_i)
    Delta_s_CV = beta_age * (S_j - S_i) / sum(beta_age * S_j)

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
        log_lambda_EV = LE_EV + C_EV + CI_EV + L_EV + LI_EV
        log_lambda_CV = LE_CV + C_CV + CI_CV + L_CV + LI_CV
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

        # Calculate the EV and CV consumption terms, and average them
        C_EV = (log(sum(s_i * c_j_bar)) - log(sum(s_i * c_i_bar))) / T
        C_CV = (log(sum(s_j * c_j_bar)) - log(sum(s_j * c_i_bar))) / T
        C = (C_EV + C_CV) / 2

        # Calculate the EV and CV leisure terms, and average them
        L_EV = (v_of_ell(sum(s_i * ell_j_bar)) - v_of_ell(sum(s_i * ell_i_bar))) / T
        L_CV = (v_of_ell(sum(s_j * ell_j_bar)) - v_of_ell(sum(s_j * ell_i_bar))) / T
        L = (L_EV + L_CV) / 2

        # Calculate the EV and CV consumption-equivalent welfare, and average them
        log_lambda_EV = LE_EV + C_EV + L_EV
        log_lambda_CV = LE_CV + C_CV + L_CV
        log_lambda = (log_lambda_EV + log_lambda_CV) / 2

        # Store the results in a dictionary
        d = {'LE_CV':         LE_CV,
             'LE_EV':         LE_EV,
             'LE':            LE,
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
