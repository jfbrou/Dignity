# Import libraries
from numpy import *
import pandas as pd
import itertools
import statsmodels.api as sm

# Define a weighted average function
def weighted_average(x, data=None, weights=None):
    if sum(data.loc[x[x.notna()].index, weights]) == 0:
        return nan
    else:
        return average(x[x.notna()], weights=data.loc[x[x.notna()].index, weights])

# Define a weighted standard deviation function
def weighted_sd(x, data=None, weights=None):
    if sum(data.loc[x[x.notna()].index, weights]) == 0:
        return nan
    else:
        mean = average(x[x.notna()], weights=data.loc[x[x.notna()].index, weights])
        return sqrt(average((x[x.notna()] - mean)**2, weights=data.loc[x[x.notna()].index, weights]))

# Define a function to create a data frame of the right form
def expand(dictionary):
    rows = itertools.product(*dictionary.values())
    return pd.DataFrame.from_records(rows, columns=dictionary.keys())

# Define the leisure utility function
def v_of_ℓ(x, ϵ=1.0, θ=14.2):
    return -(θ * ϵ / (1 + ϵ)) * (1 - x)**((1 + ϵ) / ϵ)

# Define the Beer population interpolation function
def beerpopulation(x):
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
def beerdeath(x):
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
def logλ_level(Sᵢ=None, Sⱼ=None, cᵢ_bar=None, cⱼ_bar=None, ℓᵢ_bar=None, ℓⱼ_bar=None, β=1, g=0, agemin=0, agemax=100, # Standard parameters
               S_u_bar=None, c_u_bar=None, ℓ_u_bar=None, vsl=7.4e6, c_nominal=None, agemin_u_bar=40, agemax_u_bar=100, # Intercept parameters
               inequality=False, cᵢ_bar_nd=None, cⱼ_bar_nd=None, Elog_of_cᵢ=None, Elog_of_cⱼ=None, Elog_of_cᵢ_nd=None, Elog_of_cⱼ_nd=None, Ev_of_ℓᵢ=None, Ev_of_ℓⱼ=None): # Inequality parameters
    # Restrict on selected ages
    Sᵢ = Sᵢ[agemin:agemax + 1]
    Sⱼ = Sⱼ[agemin:agemax + 1]
    cᵢ_bar = cᵢ_bar[agemin:agemax + 1]
    cⱼ_bar = cⱼ_bar[agemin:agemax + 1]
    ℓᵢ_bar = ℓᵢ_bar[agemin:agemax + 1]
    ℓⱼ_bar = ℓⱼ_bar[agemin:agemax + 1]
    S_u_bar = S_u_bar[agemin_u_bar:agemax_u_bar + 1]
    c_u_bar = c_u_bar[agemin_u_bar:agemax_u_bar + 1]
    ℓ_u_bar = ℓ_u_bar[agemin_u_bar:agemax_u_bar + 1]

    # Define the sequence of discount rates
    βᵃ = β**linspace(agemin, agemax, agemax - agemin + 1)
    βᵃ_u_bar = β**linspace(agemin_u_bar, agemax_u_bar, agemax_u_bar - agemin_u_bar + 1)

    # Define the sequence of growth rates
    ga = g * linspace(agemin, agemax, agemax - agemin + 1)
    ga_u_bar = g * linspace(agemin_u_bar, agemax_u_bar, agemax_u_bar - agemin_u_bar + 1)

    # Calculate the intercept
    u_bar = (vsl / c_nominal - dot(βᵃ_u_bar * S_u_bar, log(c_u_bar) + v_of_ℓ(ℓ_u_bar) + ga_u_bar)) / sum(βᵃ_u_bar * S_u_bar)

    # Define the lower case survival rates
    sᵢ = βᵃ * Sᵢ / sum(βᵃ * Sᵢ)
    sⱼ = βᵃ * Sⱼ / sum(βᵃ * Sⱼ)
    Δs_EV = βᵃ * (Sⱼ - Sᵢ) / sum(βᵃ * Sᵢ)
    Δs_CV = βᵃ * (Sⱼ - Sᵢ) / sum(βᵃ * Sⱼ)

    # Calculate consumption-equivalent welfare with the inequality terms
    if inequality:
        # Restrict on selected ages for the inequality terms
        Elog_of_cᵢ = Elog_of_cᵢ[agemin:agemax + 1]
        Elog_of_cⱼ = Elog_of_cⱼ[agemin:agemax + 1]
        Elog_of_cᵢ_nd = Elog_of_cᵢ_nd[agemin:agemax + 1]
        Elog_of_cⱼ_nd = Elog_of_cⱼ_nd[agemin:agemax + 1]
        cᵢ_bar_nd = cᵢ_bar_nd[agemin:agemax + 1]
        cⱼ_bar_nd = cⱼ_bar_nd[agemin:agemax + 1]
        Ev_of_ℓᵢ = Ev_of_ℓᵢ[agemin:agemax + 1]
        Ev_of_ℓⱼ = Ev_of_ℓⱼ[agemin:agemax + 1]

        # Calculate flow utility for each group
        flow_EV = u_bar + ga + Elog_of_cⱼ + Ev_of_ℓⱼ
        flow_CV = u_bar + ga + Elog_of_cᵢ + Ev_of_ℓᵢ

        # Calculate the EV and CV life expectancy terms, and average them
        LE_EV = sum(Δs_EV * flow_EV)
        LE_CV = sum(Δs_CV * flow_CV)
        LE = (LE_EV + LE_CV) / 2

        # Calculate the EV and CV consumption terms, and average them
        C_EV = log(sum(sᵢ * cⱼ_bar * exp(ga))) - log(sum(sᵢ * cᵢ_bar * exp(ga)))
        C_CV = log(sum(sⱼ * cⱼ_bar * exp(ga))) - log(sum(sⱼ * cᵢ_bar * exp(ga)))
        C = (C_EV + C_CV) / 2

        # Calculate the EV and CV consumption inequality terms, and average them
        CI_EV = sum(sᵢ * (Elog_of_cⱼ_nd - Elog_of_cᵢ_nd)) - (log(sum(sᵢ * cⱼ_bar_nd * exp(ga))) - log(sum(sᵢ * cᵢ_bar_nd * exp(ga))))
        CI_CV = sum(sⱼ * (Elog_of_cⱼ_nd - Elog_of_cᵢ_nd)) - (log(sum(sⱼ * cⱼ_bar_nd * exp(ga))) - log(sum(sⱼ * cᵢ_bar_nd * exp(ga))))
        CI = (CI_EV + CI_CV) / 2

        # Calculate the EV and CV leisure terms, and average them
        L_EV = v_of_ℓ(sum(sᵢ * ℓⱼ_bar)) - v_of_ℓ(sum(sᵢ * ℓᵢ_bar))
        L_CV = v_of_ℓ(sum(sⱼ * ℓⱼ_bar)) - v_of_ℓ(sum(sⱼ * ℓᵢ_bar))
        L = (L_EV + L_CV) / 2

        # Calculate the EV and CV leisure inequality terms, and average them
        LI_EV = sum(sᵢ * (Ev_of_ℓⱼ - Ev_of_ℓᵢ)) - L_EV
        LI_CV = sum(sⱼ * (Ev_of_ℓⱼ - Ev_of_ℓᵢ)) - L_CV
        LI = (LI_EV + LI_CV) / 2

        # Calculate the EV and CV consumption-equivalent welfare, and average them
        logλ_EV = LE_EV + C_EV + CI_EV + L_EV + LI_EV
        logλ_CV = LE_CV + C_CV + CI_CV + L_CV + LI_CV
        logλ = (logλ_EV + logλ_CV) / 2

        # Store the results in a dictionary
        d = {'LE_CV':   LE_CV,
             'LE_EV':   LE_EV,
             'LE':      LE,
             'C_CV':    C_CV,
             'C_EV':    C_EV,
             'C':       C,
             'CI_CV':   CI_CV,
             'CI_EV':   CI_EV,
             'CI':      CI,
             'L_CV':    L_CV,
             'L_EV':    L_EV,
             'L':       L,
             'LI_CV':   LI_CV,
             'LI_EV':   LI_EV,
             'LI':      LI,
             'logλ_CV': logλ_CV,
             'logλ_EV': logλ_EV,
             'logλ':    logλ,
             'u_bar':   u_bar}
    else:
        # Compute flow utility for each group
        flow_EV = u_bar + log(sum(sⱼ * cⱼ_bar * exp(ga))) + v_of_ℓ(sum(sⱼ * ℓⱼ_bar))
        flow_CV = u_bar + log(sum(sᵢ * cᵢ_bar * exp(ga))) + v_of_ℓ(sum(sᵢ * ℓᵢ_bar))

        # Calculate the EV and CV life expectancy terms, and average them
        LE_EV = sum(Δs_EV) * flow_EV
        LE_CV = sum(Δs_CV) * flow_CV
        LE = (LE_EV + LE_CV) / 2

        # Calculate the EV and CV consumption terms, and average them
        C_EV = log(sum(sᵢ * cⱼ_bar * exp(ga))) - log(sum(sᵢ * cᵢ_bar * exp(ga)))
        C_CV = log(sum(sⱼ * cⱼ_bar * exp(ga))) - log(sum(sⱼ * cᵢ_bar * exp(ga)))
        C = (C_EV + C_CV) / 2

        # Calculate the EV and CV leisure terms, and average them
        L_EV = v_of_ℓ(sum(sᵢ * ℓⱼ_bar)) - v_of_ℓ(sum(sᵢ * ℓᵢ_bar))
        L_CV = v_of_ℓ(sum(sⱼ * ℓⱼ_bar)) - v_of_ℓ(sum(sⱼ * ℓᵢ_bar))
        L = (L_EV + L_CV) / 2

        # Calculate the EV and CV consumption-equivalent welfare, and average them
        logλ_EV = LE_EV + C_EV + L_EV
        logλ_CV = LE_CV + C_CV + L_CV
        logλ = (logλ_EV + logλ_CV) / 2

        # Store the results in a dictionary
        d = {'LE_CV':   LE_CV,
             'LE_EV':   LE_EV,
             'LE':      LE,
             'C_CV':    C_CV,
             'C_EV':    C_EV,
             'C':       C,
             'L_CV':    L_CV,
             'L_EV':    L_EV,
             'L':       L,
             'logλ_CV': logλ_CV,
             'logλ_EV': logλ_EV,
             'logλ':    logλ,
             'u_bar':   u_bar}

    # Return the dictionary
    return d

# Define the level consumption-equivalent welfare calculation function with the morbidity adjustment
def logλ_level_morbidity(Sᵢ=None, Sⱼ=None, cᵢ_bar=None, cⱼ_bar=None, ℓᵢ_bar=None, ℓⱼ_bar=None, agemin=0, agemax=100, # Standard parameters
                         S_u_bar=None, halex_u_bar=None, c_u_bar=None, ℓ_u_bar=None, vsl=7.4e6, c_nominal=None, agemin_u_bar=40, agemax_u_bar=100, # Intercept parameters
                         cᵢ_bar_nd=None, cⱼ_bar_nd=None, Elog_of_cᵢ=None, Elog_of_cⱼ=None, Elog_of_cᵢ_nd=None, Elog_of_cⱼ_nd=None, Ev_of_ℓᵢ=None, Ev_of_ℓⱼ=None, # Inequality parameters
                         halexᵢ=None, halexⱼ=None, morbidity_parameter=None): # Morbidity adjustment parameters
    # Restrict on selected ages
    Sᵢ = Sᵢ[agemin:agemax + 1]
    Sⱼ = Sⱼ[agemin:agemax + 1]
    cᵢ_bar = cᵢ_bar[agemin:agemax + 1]
    cⱼ_bar = cⱼ_bar[agemin:agemax + 1]
    ℓᵢ_bar = ℓᵢ_bar[agemin:agemax + 1]
    ℓⱼ_bar = ℓⱼ_bar[agemin:agemax + 1]
    halexᵢ = halexᵢ[agemin:agemax + 1]
    halexⱼ = halexⱼ[agemin:agemax + 1]
    Elog_of_cᵢ = Elog_of_cᵢ[agemin:agemax + 1]
    Elog_of_cⱼ = Elog_of_cⱼ[agemin:agemax + 1]
    Elog_of_cᵢ_nd = Elog_of_cᵢ_nd[agemin:agemax + 1]
    Elog_of_cⱼ_nd = Elog_of_cⱼ_nd[agemin:agemax + 1]
    cᵢ_bar_nd = cᵢ_bar_nd[agemin:agemax + 1]
    cⱼ_bar_nd = cⱼ_bar_nd[agemin:agemax + 1]
    Ev_of_ℓᵢ = Ev_of_ℓᵢ[agemin:agemax + 1]
    Ev_of_ℓⱼ = Ev_of_ℓⱼ[agemin:agemax + 1]
    S_u_bar = S_u_bar[agemin_u_bar:agemax_u_bar + 1]
    c_u_bar = c_u_bar[agemin_u_bar:agemax_u_bar + 1]
    ℓ_u_bar = ℓ_u_bar[agemin_u_bar:agemax_u_bar + 1]
    halex_u_bar = halex_u_bar[agemin_u_bar:agemax_u_bar + 1]

    # Calculate the quality of life terms from the HALex
    Qᵢ = morbidity_parameter + (1 - morbidity_parameter) * halexᵢ
    Qⱼ = morbidity_parameter + (1 - morbidity_parameter) * halexⱼ
    Q_u_bar = morbidity_parameter + (1 - morbidity_parameter) * halex_u_bar

    # Calculate the intercept
    u_bar = (vsl / c_nominal - dot(S_u_bar * Q_u_bar, log(c_u_bar) + v_of_ℓ(ℓ_u_bar))) / sum(S_u_bar * Q_u_bar)

    # Define the lower case survival rates
    sᵢ = Sᵢ * Qᵢ / sum(Sᵢ * Qᵢ)
    sⱼ = Sⱼ * Qⱼ / sum(Sⱼ * Qⱼ)
    Δs_EV = (Sⱼ - Sᵢ) * Qⱼ / sum(Sᵢ * Qᵢ)
    Δs_CV = (Sⱼ - Sᵢ) * Qᵢ / sum(Sⱼ * Qⱼ)
    Δq_EV = (Qⱼ - Qᵢ) * Sᵢ / sum(Sᵢ * Qᵢ)
    Δq_CV = (Qⱼ - Qᵢ) * Sⱼ / sum(Sⱼ * Qⱼ)

    # Calculate flow utility for each group
    flow_EV = u_bar + Elog_of_cⱼ + Ev_of_ℓⱼ
    flow_CV = u_bar + Elog_of_cᵢ + Ev_of_ℓᵢ

    # Calculate the EV and CV life expectancy terms, and average them
    LE_EV = sum(Δs_EV * flow_EV)
    LE_CV = sum(Δs_CV * flow_CV)
    LE = (LE_EV + LE_CV) / 2

    # Calculate the EV and CV morbidity terms, and average them
    M_EV = sum(Δq_EV * flow_EV)
    M_CV = sum(Δq_CV * flow_CV)
    M = (M_EV + M_CV) / 2

    # Calculate the EV and CV consumption terms, and average them
    C_EV = log(sum(sᵢ * cⱼ_bar)) - log(sum(sᵢ * cᵢ_bar))
    C_CV = log(sum(sⱼ * cⱼ_bar)) - log(sum(sⱼ * cᵢ_bar))
    C = (C_EV + C_CV) / 2

    # Calculate the EV and CV consumption inequality terms, and average them
    CI_EV = sum(sᵢ * (Elog_of_cⱼ_nd - Elog_of_cᵢ_nd)) - (log(sum(sᵢ * cⱼ_bar_nd)) - log(sum(sᵢ * cᵢ_bar_nd)))
    CI_CV = sum(sⱼ * (Elog_of_cⱼ_nd - Elog_of_cᵢ_nd)) - (log(sum(sⱼ * cⱼ_bar_nd)) - log(sum(sⱼ * cᵢ_bar_nd)))
    CI = (CI_EV + CI_CV) / 2

    # Calculate the EV and CV leisure terms, and average them
    L_EV = v_of_ℓ(sum(sᵢ * ℓⱼ_bar)) - v_of_ℓ(sum(sᵢ * ℓᵢ_bar))
    L_CV = v_of_ℓ(sum(sⱼ * ℓⱼ_bar)) - v_of_ℓ(sum(sⱼ * ℓᵢ_bar))
    L = (L_EV + L_CV) / 2

    # Calculate the EV and CV leisure inequality terms, and average them
    LI_EV = sum(sᵢ * (Ev_of_ℓⱼ - Ev_of_ℓᵢ)) - L_EV
    LI_CV = sum(sⱼ * (Ev_of_ℓⱼ - Ev_of_ℓᵢ)) - L_CV
    LI = (LI_EV + LI_CV) / 2

    # Calculate the EV and CV consumption-equivalent welfare, and average them
    logλ_EV = LE_EV + M_EV + C_EV + CI_EV + L_EV + LI_EV
    logλ_CV = LE_CV + M_CV + C_CV + CI_CV + L_CV + LI_CV
    logλ = (logλ_EV + logλ_CV) / 2

    # Store the results in a dictionary
    d = {'LE_CV':   LE_CV,
         'LE_EV':   LE_EV,
         'LE':      LE,
         'M_CV':    M_CV,
         'M_EV':    M_EV,
         'M':       M,
         'C_CV':    C_CV,
         'C_EV':    C_EV,
         'C':       C,
         'CI_CV':   CI_CV,
         'CI_EV':   CI_EV,
         'CI':      CI,
         'L_CV':    L_CV,
         'L_EV':    L_EV,
         'L':       L,
         'LI_CV':   LI_CV,
         'LI_EV':   LI_EV,
         'LI':      LI,
         'logλ_CV': logλ_CV,
         'logλ_EV': logλ_EV,
         'logλ':    logλ,
         'u_bar':   u_bar}

    # Return the dictionary
    return d

# Define the level consumption-equivalent welfare calculation function with the morbidity adjustment
def logλ_level_incarceration(Sᵢ=None, Sⱼ=None, cᵢ_bar=None, cⱼ_bar=None, ℓᵢ_bar=None, ℓⱼ_bar=None, agemin=0, agemax=100, # Standard parameters
                             S_u_bar=None, I_u_bar=None, c_u_bar=None, c_u_barᴵ=None, ℓ_u_bar=None, ℓ_u_barᴵ=None, vsl=7.4e6, c_nominal=None, agemin_u_bar=40, agemax_u_bar=100, # Intercept parameters
                             cᵢ_bar_nd=None, cⱼ_bar_nd=None, Elog_of_cᵢ=None, Elog_of_cⱼ=None, Elog_of_cᵢ_nd=None, Elog_of_cⱼ_nd=None, Ev_of_ℓᵢ=None, Ev_of_ℓⱼ=None, # Inequality parameters
                             Elog_of_cᴵ=None, Ev_of_ℓᴵ=None, Iᵢ=None, Iⱼ=None, incarceration_parameter=None): # Incarceration adjustment parameters
    # Restrict on selected ages
    Sᵢ = Sᵢ[agemin:agemax + 1]
    Sⱼ = Sⱼ[agemin:agemax + 1]
    cᵢ_bar = cᵢ_bar[agemin:agemax + 1]
    cⱼ_bar = cⱼ_bar[agemin:agemax + 1]
    ℓᵢ_bar = ℓᵢ_bar[agemin:agemax + 1]
    ℓⱼ_bar = ℓⱼ_bar[agemin:agemax + 1]
    Elog_of_cᵢ = Elog_of_cᵢ[agemin:agemax + 1]
    Elog_of_cⱼ = Elog_of_cⱼ[agemin:agemax + 1]
    Elog_of_cᵢ_nd = Elog_of_cᵢ_nd[agemin:agemax + 1]
    Elog_of_cⱼ_nd = Elog_of_cⱼ_nd[agemin:agemax + 1]
    cᵢ_bar_nd = cᵢ_bar_nd[agemin:agemax + 1]
    cⱼ_bar_nd = cⱼ_bar_nd[agemin:agemax + 1]
    Ev_of_ℓᵢ = Ev_of_ℓᵢ[agemin:agemax + 1]
    Ev_of_ℓⱼ = Ev_of_ℓⱼ[agemin:agemax + 1]
    Elog_of_cᴵ = Elog_of_cᴵ[agemin:agemax + 1]
    Ev_of_ℓᴵ = Ev_of_ℓᴵ[agemin:agemax + 1]
    Iᵢ = Iᵢ[agemin:agemax + 1]
    Iⱼ = Iⱼ[agemin:agemax + 1]
    S_u_bar = S_u_bar[agemin_u_bar:agemax_u_bar + 1]
    I_u_bar = I_u_bar[agemin_u_bar:agemax_u_bar + 1]
    c_u_bar = c_u_bar[agemin_u_bar:agemax_u_bar + 1]
    c_u_barᴵ = c_u_barᴵ[agemin_u_bar:agemax_u_bar + 1]
    ℓ_u_bar = ℓ_u_bar[agemin_u_bar:agemax_u_bar + 1]
    ℓ_u_barᴵ = ℓ_u_barᴵ[agemin_u_bar:agemax_u_bar + 1]

    # Calculate the intercept
    u_bar = (vsl / c_nominal - dot(S_u_bar * (1 - I_u_bar), log(c_u_bar) + v_of_ℓ(ℓ_u_bar)) \
                             - incarceration_parameter * dot(S_u_bar * I_u_bar, log(c_u_barᴵ) + v_of_ℓ(ℓ_u_barᴵ))) \
                             / sum(S_u_bar * (1 - I_u_bar * (1 - incarceration_parameter)))

    # Calculate the EV and CV incarceration terms, and average them
    I_EV = incarceration_parameter * sum((u_bar + Elog_of_cᴵ + Ev_of_ℓᴵ) * (Sⱼ * Iⱼ - Sᵢ * Iᵢ)) / sum(Sᵢ * (1 - Iᵢ))
    I_CV = incarceration_parameter * sum((u_bar + Elog_of_cᴵ + Ev_of_ℓᴵ) * (Sⱼ * Iⱼ - Sᵢ * Iᵢ)) / sum(Sⱼ * (1 - Iⱼ))
    I = (I_EV + I_CV) / 2

    # Adjust the survival rates for incarceration
    Sᵢ = Sᵢ * (1 - Iᵢ)
    Sⱼ = Sⱼ * (1 - Iⱼ)

    # Define the lower case survival rates
    sᵢ = Sᵢ / sum(Sᵢ)
    sⱼ = Sⱼ / sum(Sⱼ)
    Δs_EV = (Sⱼ - Sᵢ) / sum(Sᵢ)
    Δs_CV = (Sⱼ - Sᵢ) / sum(Sⱼ)

    # Calculate flow utility for each group
    flow_EV = u_bar + Elog_of_cⱼ + Ev_of_ℓⱼ
    flow_CV = u_bar + Elog_of_cᵢ + Ev_of_ℓᵢ

    # Calculate the EV and CV life expectancy terms, and average them
    LE_EV = sum(Δs_EV * flow_EV)
    LE_CV = sum(Δs_CV * flow_CV)
    LE = (LE_EV + LE_CV) / 2

    # Calculate the EV and CV consumption terms, and average them
    C_EV = log(sum(sᵢ * cⱼ_bar)) - log(sum(sᵢ * cᵢ_bar))
    C_CV = log(sum(sⱼ * cⱼ_bar)) - log(sum(sⱼ * cᵢ_bar))
    C = (C_EV + C_CV) / 2

    # Calculate the EV and CV consumption inequality terms, and average them
    CI_EV = sum(sᵢ * (Elog_of_cⱼ_nd - Elog_of_cᵢ_nd)) - (log(sum(sᵢ * cⱼ_bar_nd)) - log(sum(sᵢ * cᵢ_bar_nd)))
    CI_CV = sum(sⱼ * (Elog_of_cⱼ_nd - Elog_of_cᵢ_nd)) - (log(sum(sⱼ * cⱼ_bar_nd)) - log(sum(sⱼ * cᵢ_bar_nd)))
    CI = (CI_EV + CI_CV) / 2

    # Calculate the EV and CV leisure terms, and average them
    L_EV = v_of_ℓ(sum(sᵢ * ℓⱼ_bar)) - v_of_ℓ(sum(sᵢ * ℓᵢ_bar))
    L_CV = v_of_ℓ(sum(sⱼ * ℓⱼ_bar)) - v_of_ℓ(sum(sⱼ * ℓᵢ_bar))
    L = (L_EV + L_CV) / 2

    # Calculate the EV and CV leisure inequality terms, and average them
    LI_EV = sum(sᵢ * (Ev_of_ℓⱼ - Ev_of_ℓᵢ)) - L_EV
    LI_CV = sum(sⱼ * (Ev_of_ℓⱼ - Ev_of_ℓᵢ)) - L_CV
    LI = (LI_EV + LI_CV) / 2

    # Calculate the EV and CV consumption-equivalent welfare, and average them
    logλ_EV = LE_EV + C_EV + CI_EV + L_EV + LI_EV + I_EV
    logλ_CV = LE_CV + C_CV + CI_CV + L_CV + LI_CV + I_CV
    logλ = (logλ_EV + logλ_CV) / 2

    # Store the results in a dictionary
    d = {'LE_CV':   LE_CV,
         'LE_EV':   LE_EV,
         'LE':      LE,
         'C_CV':    C_CV,
         'C_EV':    C_EV,
         'C':       C,
         'CI_CV':   CI_CV,
         'CI_EV':   CI_EV,
         'CI':      CI,
         'L_CV':    L_CV,
         'L_EV':    L_EV,
         'L':       L,
         'LI_CV':   LI_CV,
         'LI_EV':   LI_EV,
         'LI':      LI,
         'I_CV':    I_CV,
         'I_EV':    I_EV,
         'I':       I,
         'logλ_CV': logλ_CV,
         'logλ_EV': logλ_EV,
         'logλ':    logλ,
         'u_bar':   u_bar}

    # Return the dictionary
    return d

# Define the growth consumption-equivalent welfare calculation function
def logλ_growth(Sᵢ=None, Sⱼ=None, cᵢ_bar=None, cⱼ_bar=None, ℓᵢ_bar=None, ℓⱼ_bar=None, β=1, g=0, agemin=0, agemax=100, T=None, # Standard parameters
                S_u_bar=None, c_u_bar=None, ℓ_u_bar=None, vsl=7.4e6, c_nominal=None, agemin_u_bar=40, agemax_u_bar=100, # Intercept parameters
                inequality=False, cᵢ_bar_nd=None, cⱼ_bar_nd=None, Elog_of_cᵢ=None, Elog_of_cⱼ=None, Elog_of_cᵢ_nd=None, Elog_of_cⱼ_nd=None, Ev_of_ℓᵢ=None, Ev_of_ℓⱼ=None): # Inequality parameters
    # Restrict on selected ages
    Sᵢ = Sᵢ[agemin:agemax + 1]
    Sⱼ = Sⱼ[agemin:agemax + 1]
    cᵢ_bar = cᵢ_bar[agemin:agemax + 1]
    cⱼ_bar = cⱼ_bar[agemin:agemax + 1]
    ℓᵢ_bar = ℓᵢ_bar[agemin:agemax + 1]
    ℓⱼ_bar = ℓⱼ_bar[agemin:agemax + 1]
    S_u_bar = S_u_bar[agemin_u_bar:agemax_u_bar + 1]
    c_u_bar = c_u_bar[agemin_u_bar:agemax_u_bar + 1]
    ℓ_u_bar = ℓ_u_bar[agemin_u_bar:agemax_u_bar + 1]

    # Define the sequence of discount rates
    βᵃ = β**linspace(agemin, agemax, agemax - agemin + 1)
    βᵃ_u_bar = β**linspace(agemin_u_bar, agemax_u_bar, agemax_u_bar - agemin_u_bar + 1)

    # Define the sequence of growth rates
    ga = g * linspace(agemin, agemax, agemax - agemin + 1)
    ga_u_bar = g * linspace(agemin_u_bar, agemax_u_bar, agemax_u_bar - agemin_u_bar + 1)

    # Calculate the intercept
    u_bar = (vsl / c_nominal - dot(βᵃ_u_bar * S_u_bar, log(c_u_bar) + v_of_ℓ(ℓ_u_bar) + ga_u_bar)) / sum(βᵃ_u_bar * S_u_bar)

    # Define the lower case survival rates
    sᵢ = βᵃ * Sᵢ / sum(βᵃ * Sᵢ)
    sⱼ = βᵃ * Sⱼ / sum(βᵃ * Sⱼ)
    Δs_EV = βᵃ * (Sⱼ - Sᵢ) / sum(βᵃ * Sᵢ)
    Δs_CV = βᵃ * (Sⱼ - Sᵢ) / sum(βᵃ * Sⱼ)

    # Calculate consumption-equivalent welfare with the inequality terms
    if inequality:
        # Restrict on selected ages for the inequality terms
        Elog_of_cᵢ = Elog_of_cᵢ[agemin:agemax + 1]
        Elog_of_cⱼ = Elog_of_cⱼ[agemin:agemax + 1]
        Elog_of_cᵢ_nd = Elog_of_cᵢ_nd[agemin:agemax + 1]
        Elog_of_cⱼ_nd = Elog_of_cⱼ_nd[agemin:agemax + 1]
        cᵢ_bar_nd = cᵢ_bar_nd[agemin:agemax + 1]
        cⱼ_bar_nd = cⱼ_bar_nd[agemin:agemax + 1]
        Ev_of_ℓᵢ = Ev_of_ℓᵢ[agemin:agemax + 1]
        Ev_of_ℓⱼ = Ev_of_ℓⱼ[agemin:agemax + 1]

        # Calculate flow utility for each group
        flow_EV = u_bar + ga + Elog_of_cⱼ + Ev_of_ℓⱼ
        flow_CV = u_bar + ga + Elog_of_cᵢ + Ev_of_ℓᵢ

        # Calculate the EV and CV life expectancy terms, and average them
        LE_EV = sum(Δs_EV * flow_EV) / T
        LE_CV = sum(Δs_CV * flow_CV) / T
        LE = (LE_EV + LE_CV) / 2

        # Calculate the EV and CV consumption terms, and average them
        C_EV = (log(sum(sᵢ * cⱼ_bar)) - log(sum(sᵢ * cᵢ_bar))) / T
        C_CV = (log(sum(sⱼ * cⱼ_bar)) - log(sum(sⱼ * cᵢ_bar))) / T
        C = (C_EV + C_CV) / 2

        # Calculate the EV and CV consumption inequality terms, and average them
        CI_EV = (sum(sᵢ * (Elog_of_cⱼ_nd - Elog_of_cᵢ_nd)) - (log(sum(sᵢ * cⱼ_bar_nd)) - log(sum(sᵢ * cᵢ_bar_nd)))) / T
        CI_CV = (sum(sⱼ * (Elog_of_cⱼ_nd - Elog_of_cᵢ_nd)) - (log(sum(sⱼ * cⱼ_bar_nd)) - log(sum(sⱼ * cᵢ_bar_nd)))) / T
        CI = (CI_EV + CI_CV) / 2

        # Calculate the EV and CV leisure terms, and average them
        L_EV = (v_of_ℓ(sum(sᵢ * ℓⱼ_bar)) - v_of_ℓ(sum(sᵢ * ℓᵢ_bar))) / T
        L_CV = (v_of_ℓ(sum(sⱼ * ℓⱼ_bar)) - v_of_ℓ(sum(sⱼ * ℓᵢ_bar))) / T
        L = (L_EV + L_CV) / 2

        # Calculate the EV and CV leisure inequality terms, and average them
        LI_EV = (sum(sᵢ * (Ev_of_ℓⱼ - Ev_of_ℓᵢ)) - (v_of_ℓ(sum(sᵢ * ℓⱼ_bar)) - v_of_ℓ(sum(sᵢ * ℓᵢ_bar)))) / T
        LI_CV = (sum(sⱼ * (Ev_of_ℓⱼ - Ev_of_ℓᵢ)) - (v_of_ℓ(sum(sⱼ * ℓⱼ_bar)) - v_of_ℓ(sum(sⱼ * ℓᵢ_bar)))) / T
        LI = (LI_EV + LI_CV) / 2

        # Calculate the EV and CV consumption-equivalent welfare, and average them
        logλ_EV = LE_EV + C_EV + CI_EV + L_EV + LI_EV
        logλ_CV = LE_CV + C_CV + CI_CV + L_CV + LI_CV
        logλ = (logλ_EV + logλ_CV) / 2

        # Store the results in a dictionary
        d = {'LE_CV':   LE_CV,
             'LE_EV':   LE_EV,
             'LE':      LE,
             'C_CV':    C_CV,
             'C_EV':    C_EV,
             'C':       C,
             'CI_CV':   CI_CV,
             'CI_EV':   CI_EV,
             'CI':      CI,
             'L_CV':    L_CV,
             'L_EV':    L_EV,
             'L':       L,
             'LI_CV':   LI_CV,
             'LI_EV':   LI_EV,
             'LI':      LI,
             'logλ_CV': logλ_CV,
             'logλ_EV': logλ_EV,
             'logλ':    logλ,
             'u_bar':   u_bar}
    else:
        # Compute flow utility for each group
        flow_EV = u_bar + log(sum(sⱼ * cⱼ_bar * exp(ga))) + v_of_ℓ(sum(sⱼ * ℓⱼ_bar))
        flow_CV = u_bar + log(sum(sᵢ * cᵢ_bar * exp(ga))) + v_of_ℓ(sum(sᵢ * ℓᵢ_bar))

        # Calculate the EV and CV life expectancy terms, and average them
        LE_EV = (sum(Δs_EV) * flow_EV) / T
        LE_CV = (sum(Δs_CV) * flow_CV) / T
        LE = (LE_EV + LE_CV) / 2

        # Calculate the EV and CV consumption terms, and average them
        C_EV = (log(sum(sᵢ * cⱼ_bar)) - log(sum(sᵢ * cᵢ_bar))) / T
        C_CV = (log(sum(sⱼ * cⱼ_bar)) - log(sum(sⱼ * cᵢ_bar))) / T
        C = (C_EV + C_CV) / 2

        # Calculate the EV and CV leisure terms, and average them
        L_EV = (v_of_ℓ(sum(sᵢ * ℓⱼ_bar)) - v_of_ℓ(sum(sᵢ * ℓᵢ_bar))) / T
        L_CV = (v_of_ℓ(sum(sⱼ * ℓⱼ_bar)) - v_of_ℓ(sum(sⱼ * ℓᵢ_bar))) / T
        L = (L_EV + L_CV) / 2

        # Calculate the EV and CV consumption-equivalent welfare, and average them
        logλ_EV = LE_EV + C_EV + L_EV
        logλ_CV = LE_CV + C_CV + L_CV
        logλ = (logλ_EV + logλ_CV) / 2

        # Store the results in a dictionary
        d = {'LE_CV':   LE_CV,
             'LE_EV':   LE_EV,
             'LE':      LE,
             'C_CV':    C_CV,
             'C_EV':    C_EV,
             'C':       C,
             'L_CV':    L_CV,
             'L_EV':    L_EV,
             'L':       L,
             'logλ_CV': logλ_CV,
             'logλ_EV': logλ_EV,
             'logλ':    logλ,
             'u_bar':   u_bar}

    # Return the dictionary
    return d
