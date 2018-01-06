# File Contains: Python code containing closed-form solutions for the valuation of European Options,
# American Options, Asian Options, Spread Options, Heat Rate Options, and Implied Volatility
#
# This document demonstrates a Python implementation of some option models described in books written by Davis
# Edwards: "Energy Trading and Investing", "Risk Management in Trading", "Energy Investing Demystified".
#
# for backward compatability with Python 2.7
from __future__ import division

# import necessary libaries
import unittest
import math
import numpy as np
from scipy.stats import norm
from scipy.stats import mvn

# Developer can toggle _DEBUG to True for more messages
# normally this is set to False
_DEBUG = False


# This class contains the limits on inputs for GBS models
# It is not intended to be part of this module's public interface
class _GBS_Limits:
    # An GBS model will return an error if an out-of-bound input is input
    MAX32 = 2147483248.0

    MIN_T = 1.0 / 1000.0  # requires some time left before expiration
    MIN_X = 0.01
    MIN_FS = 0.01

    # Volatility smaller than 0.5% causes American Options calculations
    # to fail (Number to large errors).
    # GBS() should be OK with any positive number. Since vols less
    # than 0.5% are expected to be extremely rare, and most likely bad inputs,
    # _gbs() is assigned this limit too
    MIN_V = 0.005

    MAX_T = 100
    MAX_X = MAX32
    MAX_FS = MAX32

    # Asian Option limits
    # maximum TA is time to expiration for the option
    MIN_TA = 0

    # This model will work with higher values for b, r, and V. However, such values are extremely uncommon.
    # To catch some common errors, interest rates and volatility is capped to 100%
    # This reason for 1 (100%) is mostly to cause the library to throw an exceptions
    # if a value like 15% is entered as 15 rather than 0.15)
    MIN_b = -1
    MIN_r = -1

    MAX_b = 1
    MAX_r = 1
    MAX_V = 1


# ------------------------------
# This function verifies that the Call/Put indicator is correctly entered
def _test_option_type(option_type):
    if (option_type != "c") and (option_type != "p"):
        raise GBS_InputError("Invalid Input option_type ({0}). Acceptable value are: c, p".format(option_type))


# ------------------------------
# This function makes sure inputs are OK
# It throws an exception if there is a failure
def _gbs_test_inputs(option_type, fs, x, t, r, b, v):
    # -----------
    # Test inputs are reasonable
    _test_option_type(option_type)

    if (x < _GBS_Limits.MIN_X) or (x > _GBS_Limits.MAX_X):
        raise GBS_InputError(
            "Invalid Input Strike Price (X). Acceptable range for inputs is {1} to {2}".format(x, _GBS_Limits.MIN_X,
                                                                                               _GBS_Limits.MAX_X))

    if (fs < _GBS_Limits.MIN_FS) or (fs > _GBS_Limits.MAX_FS):
        raise GBS_InputError(
            "Invalid Input Forward/Spot Price (FS). Acceptable range for inputs is {1} to {2}".format(fs,
                                                                                                      _GBS_Limits.MIN_FS,
                                                                                                      _GBS_Limits.MAX_FS))

    if (t < _GBS_Limits.MIN_T) or (t > _GBS_Limits.MAX_T):
        raise GBS_InputError(
            "Invalid Input Time (T = {0}). Acceptable range for inputs is {1} to {2}".format(t, _GBS_Limits.MIN_T,
                                                                                             _GBS_Limits.MAX_T))

    if (b < _GBS_Limits.MIN_b) or (b > _GBS_Limits.MAX_b):
        raise GBS_InputError(
            "Invalid Input Cost of Carry (b = {0}). Acceptable range for inputs is {1} to {2}".format(b,
                                                                                                      _GBS_Limits.MIN_b,
                                                                                                      _GBS_Limits.MAX_b))

    if (r < _GBS_Limits.MIN_r) or (r > _GBS_Limits.MAX_r):
        raise GBS_InputError(
            "Invalid Input Risk Free Rate (r = {0}). Acceptable range for inputs is {1} to {2}".format(r,
                                                                                                       _GBS_Limits.MIN_r,
                                                                                                       _GBS_Limits.MAX_r))

    if (v < _GBS_Limits.MIN_V) or (v > _GBS_Limits.MAX_V):
        raise GBS_InputError(
            "Invalid Input Implied Volatility (V = {0}). Acceptable range for inputs is {1} to {2}".format(v,
                                                                                                           _GBS_Limits.MIN_V,
                                                                                                           _GBS_Limits.MAX_V))


# The primary class for calculating Generalized Black Scholes option prices and deltas
# It is not intended to be part of this module's public interface

# Inputs: option_type = "p" or "c", fs = price of underlying, x = strike, t = time to expiration, r = risk free rate
#         b = cost of carry, v = implied volatility
# Outputs: value, delta, gamma, theta, vega, rho
def _gbs(option_type, fs, x, t, r, b, v):
    _debug("Debugging Information: _gbs()")
    # -----------
    # Test Inputs (throwing an exception on failure)
    _gbs_test_inputs(option_type, fs, x, t, r, b, v)

    # -----------
    # Create preliminary calculations
    t__sqrt = math.sqrt(t)
    d1 = (math.log(fs / x) + (b + (v * v) / 2) * t) / (v * t__sqrt)
    d2 = d1 - v * t__sqrt

    if option_type == "c":
        # it's a call
        _debug("     Call Option")
        value = fs * math.exp((b - r) * t) * norm.cdf(d1) - x * math.exp(-r * t) * norm.cdf(d2)
        delta = math.exp((b - r) * t) * norm.cdf(d1)
        gamma = math.exp((b - r) * t) * norm.pdf(d1) / (fs * v * t__sqrt)
        theta = -(fs * v * math.exp((b - r) * t) * norm.pdf(d1)) / (2 * t__sqrt) - (b - r) * fs * math.exp(
            (b - r) * t) * norm.cdf(d1) - r * x * math.exp(-r * t) * norm.cdf(d2)
        vega = math.exp((b - r) * t) * fs * t__sqrt * norm.pdf(d1)
        rho = x * t * math.exp(-r * t) * norm.cdf(d2)
    else:
        # it's a put
        _debug("     Put Option")
        value = x * math.exp(-r * t) * norm.cdf(-d2) - (fs * math.exp((b - r) * t) * norm.cdf(-d1))
        delta = -math.exp((b - r) * t) * norm.cdf(-d1)
        gamma = math.exp((b - r) * t) * norm.pdf(d1) / (fs * v * t__sqrt)
        theta = -(fs * v * math.exp((b - r) * t) * norm.pdf(d1)) / (2 * t__sqrt) + (b - r) * fs * math.exp(
            (b - r) * t) * norm.cdf(-d1) + r * x * math.exp(-r * t) * norm.cdf(-d2)
        vega = math.exp((b - r) * t) * fs * t__sqrt * norm.pdf(d1)
        rho = -x * t * math.exp(-r * t) * norm.cdf(-d2)

    _debug("     d1= {0}\n     d2 = {1}".format(d1, d2))
    _debug("     delta = {0}\n     gamma = {1}\n     theta = {2}\n     vega = {3}\n     rho={4}".format(delta, gamma,
                                                                                                        theta, vega,
                                                                                                        rho))

    return value, delta, gamma, theta, vega, rho


# -----------
# Generalized American Option Pricer
# This is a wrapper to check inputs and route to the current "best" American option model
def _american_option(option_type, fs, x, t, r, b, v):
    # -----------
    # Test Inputs (throwing an exception on failure)
    _debug("Debugging Information: _american_option()")
    _gbs_test_inputs(option_type, fs, x, t, r, b, v)

    # -----------
    if option_type == "c":
        # Call Option
        _debug("     Call Option")
        return _bjerksund_stensland_2002(fs, x, t, r, b, v)
    else:
        # Put Option
        _debug("     Put Option")

        # Using the put-call transformation: P(X, FS, T, r, b, V) = C(FS, X, T, -b, r-b, V)
        # WARNING - When reconciling this code back to the B&S paper, the order of variables is different

        put__x = fs
        put_fs = x
        put_b = -b
        put_r = r - b

        # pass updated values into the Call Valuation formula
        return _bjerksund_stensland_2002(put_fs, put__x, t, put_r, put_b, v)



# -----------
# American Call Option (Bjerksund Stensland 1993 approximation)
# This is primarily here for testing purposes; 2002 model has superseded this one
def _bjerksund_stensland_1993(fs, x, t, r, b, v):
    # -----------
    # initialize output
    # using GBS greeks (TO DO: update greek calculations)
    my_output = _gbs("c", fs, x, t, r, b, v)

    e_value = my_output[0]
    delta = my_output[1]
    gamma = my_output[2]
    theta = my_output[3]
    vega = my_output[4]
    rho = my_output[5]

    # debugging for calculations
    _debug("-----")
    _debug("Debug Information: _Bjerksund_Stensland_1993())")

    # if b >= r, it is never optimal to exercise before maturity
    # so we can return the GBS value
    if b >= r:
        _debug("     b >= r, early exercise never optimal, returning GBS value")
        return e_value, delta, gamma, theta, vega, rho

    # Intermediate Calculations
    v2 = v ** 2
    sqrt_t = math.sqrt(t)

    beta = (0.5 - b / v2) + math.sqrt(((b / v2 - 0.5) ** 2) + 2 * r / v2)
    b_infinity = (beta / (beta - 1)) * x
    b_zero = max(x, (r / (r - b)) * x)

    h1 = -(b * t + 2 * v * sqrt_t) * (b_zero / (b_infinity - b_zero))
    i = b_zero + (b_infinity - b_zero) * (1 - math.exp(h1))
    alpha = (i - x) * (i ** (-beta))

    # debugging for calculations
    _debug("     b = {0}".format(b))
    _debug("     v2 = {0}".format(v2))
    _debug("     beta = {0}".format(beta))
    _debug("     b_infinity = {0}".format(b_infinity))
    _debug("     b_zero = {0}".format(b_zero))
    _debug("     h1 = {0}".format(h1))
    _debug("     i  = {0}".format(i))
    _debug("     alpha = {0}".format(alpha))

    # Check for immediate exercise
    if fs >= i:
        _debug("     Immediate Exercise")
        value = fs - x
    else:
        _debug("     American Exercise")
        value = (alpha * (fs ** beta)
                 - alpha * _phi(fs, t, beta, i, i, r, b, v)
                 + _phi(fs, t, 1, i, i, r, b, v)
                 - _phi(fs, t, 1, x, i, r, b, v)
                 - x * _phi(fs, t, 0, i, i, r, b, v)
                 + x * _phi(fs, t, 0, x, i, r, b, v))

    # The approximation can break down in boundary conditions
    # make sure the value is at least equal to the European value
    value = max(value, e_value)
    return value, delta, gamma, theta, vega, rho



# -----------
# American Call Option (Bjerksund Stensland 2002 approximation)
def _bjerksund_stensland_2002(fs, x, t, r, b, v):
    # -----------
    # initialize output
    # using GBS greeks (TO DO: update greek calculations)
    my_output = _gbs("c", fs, x, t, r, b, v)

    e_value = my_output[0]
    delta = my_output[1]
    gamma = my_output[2]
    theta = my_output[3]
    vega = my_output[4]
    rho = my_output[5]

    # debugging for calculations
    _debug("-----")
    _debug("Debug Information: _Bjerksund_Stensland_2002())")

    # If b >= r, it is never optimal to exercise before maturity
    # so we can return the GBS value
    if b >= r:
        _debug("     Returning GBS value")
        return e_value, delta, gamma, theta, vega, rho

    # -----------
    # Create preliminary calculations
    v2 = v ** 2
    t1 = 0.5 * (math.sqrt(5) - 1) * t
    t2 = t

    beta_inside = ((b / v2 - 0.5) ** 2) + 2 * r / v2
    # forcing the inside of the sqrt to be a positive number
    beta_inside = abs(beta_inside)
    beta = (0.5 - b / v2) + math.sqrt(beta_inside)
    b_infinity = (beta / (beta - 1)) * x
    b_zero = max(x, (r / (r - b)) * x)

    h1 = -(b * t1 + 2 * v * math.sqrt(t1)) * ((x ** 2) / ((b_infinity - b_zero) * b_zero))
    h2 = -(b * t2 + 2 * v * math.sqrt(t2)) * ((x ** 2) / ((b_infinity - b_zero) * b_zero))

    i1 = b_zero + (b_infinity - b_zero) * (1 - math.exp(h1))
    i2 = b_zero + (b_infinity - b_zero) * (1 - math.exp(h2))
    print i1, i2
    alpha1 = (i1 - x) * np.power(i1, -beta)
    alpha2 = (i2 - x) * np.power(i2,(-beta))

    # debugging for calculations
    _debug("     t1 = {0}".format(t1))
    _debug("     beta = {0}".format(beta))
    _debug("     b_infinity = {0}".format(b_infinity))
    _debug("     b_zero = {0}".format(b_zero))
    _debug("     h1 = {0}".format(h1))
    _debug("     h2 = {0}".format(h2))
    _debug("     i1 = {0}".format(i1))
    _debug("     i2 = {0}".format(i2))
    _debug("     alpha1 = {0}".format(alpha1))
    _debug("     alpha2 = {0}".format(alpha2))

    # check for immediate exercise
    if fs >= i2:
        value = fs - x
    else:
        # Perform the main calculation
        value = (alpha2 * (fs ** beta)
                 - alpha2 * _phi(fs, t1, beta, i2, i2, r, b, v)
                 + _phi(fs, t1, 1, i2, i2, r, b, v)
                 - _phi(fs, t1, 1, i1, i2, r, b, v)
                 - x * _phi(fs, t1, 0, i2, i2, r, b, v)
                 + x * _phi(fs, t1, 0, i1, i2, r, b, v)
                 + alpha1 * _phi(fs, t1, beta, i1, i2, r, b, v)
                 - alpha1 * _psi(fs, t2, beta, i1, i2, i1, t1, r, b, v)
                 + _psi(fs, t2, 1, i1, i2, i1, t1, r, b, v)
                 - _psi(fs, t2, 1, x, i2, i1, t1, r, b, v)
                 - x * _psi(fs, t2, 0, i1, i2, i1, t1, r, b, v)
                 + x * _psi(fs, t2, 0, x, i2, i1, t1, r, b, v))

    # in boundary conditions, this approximation can break down
    # Make sure option value is greater than or equal to European value
    value = max(value, e_value)

    # -----------
    # Return Data
    return value, delta, gamma, theta, vega, rho



# ---------------------------
# American Option Intermediate Calculations

# -----------
# The Psi() function used by _Bjerksund_Stensland_2002 model
def _psi(fs, t2, gamma, h, i2, i1, t1, r, b, v):
    vsqrt_t1 = v * math.sqrt(t1)
    vsqrt_t2 = v * math.sqrt(t2)

    bgamma_t1 = (b + (gamma - 0.5) * (v ** 2)) * t1
    bgamma_t2 = (b + (gamma - 0.5) * (v ** 2)) * t2

    d1 = (math.log(fs / i1) + bgamma_t1) / vsqrt_t1
    d3 = (math.log(fs / i1) - bgamma_t1) / vsqrt_t1

    d2 = (math.log((i2 ** 2) / (fs * i1)) + bgamma_t1) / vsqrt_t1
    d4 = (math.log((i2 ** 2) / (fs * i1)) - bgamma_t1) / vsqrt_t1

    e1 = (math.log(fs / h) + bgamma_t2) / vsqrt_t2
    e2 = (math.log((i2 ** 2) / (fs * h)) + bgamma_t2) / vsqrt_t2
    e3 = (math.log((i1 ** 2) / (fs * h)) + bgamma_t2) / vsqrt_t2
    e4 = (math.log((fs * (i1 ** 2)) / (h * (i2 ** 2))) + bgamma_t2) / vsqrt_t2

    tau = math.sqrt(t1 / t2)
    lambda1 = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * (v ** 2))
    kappa = (2 * b) / (v ** 2) + (2 * gamma - 1)

    psi = math.exp(lambda1 * t2) * (fs ** gamma) * (_cbnd(-d1, -e1, tau)
                                                    - ((i2 / fs) ** kappa) * _cbnd(-d2, -e2, tau)
                                                    - ((i1 / fs) ** kappa) * _cbnd(-d3, -e3, -tau)
                                                    + ((i1 / i2) ** kappa) * _cbnd(-d4, -e4, -tau))
    return psi



# -----------
# The Phi() function used by _Bjerksund_Stensland_2002 model and the _Bjerksund_Stensland_1993 model
def _phi(fs, t, gamma, h, i, r, b, v):
    d1 = -(math.log(fs / h) + (b + (gamma - 0.5) * (v ** 2)) * t) / (v * math.sqrt(t))
    d2 = d1 - 2 * math.log(i / fs) / (v * math.sqrt(t))

    lambda1 = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * (v ** 2))
    kappa = (2 * b) / (v ** 2) + (2 * gamma - 1)

    phi = math.exp(lambda1 * t) * (fs ** gamma) * (norm.cdf(d1) - ((i / fs) ** kappa) * norm.cdf(d2))

    _debug("-----")
    _debug("Debug info for: _phi()")
    _debug("    d1={0}".format(d1))
    _debug("    d2={0}".format(d2))
    _debug("    lambda={0}".format(lambda1))
    _debug("    kappa={0}".format(kappa))
    _debug("    phi={0}".format(phi))
    return phi


# -----------
# Cumulative Bivariate Normal Distribution
# Primarily called by Psi() function, part of the _Bjerksund_Stensland_2002 model
def _cbnd(a, b, rho):
    # This distribution uses the Genz multi-variate normal distribution
    # code found as part of the standard SciPy distribution
    lower = np.array([0, 0])
    upper = np.array([a, b])
    infin = np.array([0, 0])
    correl = rho
    error, value, inform = mvn.mvndst(lower, upper, infin, correl)
    return value


# ----------
# Inputs (not all functions use all inputs)
#      fs = forward/spot price
#      x = Strike
#      t = Time (in years)
#      r = risk free rate
#      b = cost of carry
#      cp = Call or Put price
#      precision = (optional) precision at stopping point
#      max_steps = (optional) maximum number of steps

# ----------
# Approximate Implied Volatility
#
# This function is used to choose a starting point for the
# search functions (Newton and bisection searches).
# Brenner & Subrahmanyam (1988), Feinstein (1988)

def _approx_implied_vol(option_type, fs, x, t, r, b, cp):
    _test_option_type(option_type)

    ebrt = math.exp((b - r) * t)
    ert = math.exp(-r * t)

    a = math.sqrt(2 * math.pi) / (fs * ebrt + x * ert)

    if option_type == "c":
        payoff = fs * ebrt - x * ert
    else:
        payoff = x * ert - fs * ebrt

    b = cp - payoff / 2
    c = (payoff ** 2) / math.pi

    v = (a * (b + math.sqrt(b ** 2 + c))) / math.sqrt(t)

    return v


# ----------
# Find the Implied Volatility of an European (GBS) Option given a price
# using Newton-Raphson method for greater speed since Vega is available

def _gbs_implied_vol(option_type, fs, x, t, r, b, cp, precision=.00001, max_steps=100):
    return _newton_implied_vol(_gbs, option_type, x, fs, t, b, r, cp, precision, max_steps)


# ----------
# Find the Implied Volatility of an American Option given a price
# Using bisection method since Vega is difficult to estimate for Americans
def _american_implied_vol(option_type, fs, x, t, r, b, cp, precision=.00001, max_steps=100):
    return _bisection_implied_vol(_american_option, option_type, fs, x, t, r, b, cp, precision, max_steps)


# ----------
# Calculate Implied Volatility with a Newton Raphson search
def _newton_implied_vol(val_fn, option_type, x, fs, t, b, r, cp, precision=.00001, max_steps=100):
    # make sure a valid option type was entered
    _test_option_type(option_type)

    # Estimate starting Vol, making sure it is allowable range
    v = _approx_implied_vol(option_type, fs, x, t, r, b, cp)
    v = max(_GBS_Limits.MIN_V, v)
    v = min(_GBS_Limits.MAX_V, v)

    # Calculate the value at the estimated vol
    value, delta, gamma, theta, vega, rho = val_fn(option_type, fs, x, t, r, b, v)
    min_diff = abs(cp - value)

    _debug("-----")
    _debug("Debug info for: _Newton_ImpliedVol()")
    _debug("    Vinitial={0}".format(v))

    # Newton-Raphson Search
    countr = 0
    while precision <= abs(cp - value) <= min_diff and countr < max_steps:

        v = v - (value - cp) / vega
        if (v > _GBS_Limits.MAX_V) or (v < _GBS_Limits.MIN_V):
            _debug("    Volatility out of bounds")
            break

        value, delta, gamma, theta, vega, rho = val_fn(option_type, fs, x, t, r, b, v)
        min_diff = min(abs(cp - value), min_diff)

        # keep track of how many loops
        countr += 1
        _debug("     IVOL STEP {0}. v={1}".format(countr, v))

    # check if function converged and return a value
    if abs(cp - value) < precision:
        # the search function converged
        return v
    else:
        # if the search function didn't converge, try a bisection search
        return _bisection_implied_vol(val_fn, option_type, fs, x, t, r, b, cp, precision, max_steps)


# ----------
# Find the Implied Volatility using a Bisection search
def _bisection_implied_vol(val_fn, option_type, fs, x, t, r, b, cp, precision=.00001, max_steps=100):
    _debug("-----")
    _debug("Debug info for: _bisection_implied_vol()")

    # Estimate Upper and Lower bounds on volatility
    # Assume American Implied vol is within +/- 50% of the GBS Implied Vol
    v_mid = _approx_implied_vol(option_type, fs, x, t, r, b, cp)

    if (v_mid <= _GBS_Limits.MIN_V) or (v_mid >= _GBS_Limits.MAX_V):
        # if the volatility estimate is out of bounds, search entire allowed vol space
        v_low = _GBS_Limits.MIN_V
        v_high = _GBS_Limits.MAX_V
        v_mid = (v_low + v_high) / 2
    else:
        # reduce the size of the vol space
        v_low = max(_GBS_Limits.MIN_V, v_mid * .5)
        v_high = min(_GBS_Limits.MAX_V, v_mid * 1.5)

    # Estimate the high/low bounds on price
    cp_mid = val_fn(option_type, fs, x, t, r, b, v_mid)[0]

    # initialize bisection loop
    current_step = 0
    diff = abs(cp - cp_mid)

    _debug("     American IVOL starting conditions: CP={0} cp_mid={1}".format(cp, cp_mid))
    _debug("     IVOL {0}. V[{1},{2},{3}]".format(current_step, v_low, v_mid, v_high))

    # Keep bisection volatility until correct price is found
    while (diff > precision) and (current_step < max_steps):
        current_step += 1

        # Cut the search area in half
        if cp_mid < cp:
            v_low = v_mid
        else:
            v_high = v_mid

        cp_low = val_fn(option_type, fs, x, t, r, b, v_low)[0]
        cp_high = val_fn(option_type, fs, x, t, r, b, v_high)[0]

        v_mid = v_low + (cp - cp_low) * (v_high - v_low) / (cp_high - cp_low)
        v_mid = max(_GBS_Limits.MIN_V, v_mid)  # enforce high/low bounds
        v_mid = min(_GBS_Limits.MAX_V, v_mid)  # enforce high/low bounds

        cp_mid = val_fn(option_type, fs, x, t, r, b, v_mid)[0]
        diff = abs(cp - cp_mid)

        _debug("     IVOL {0}. V[{1},{2},{3}]".format(current_step, v_low, v_mid, v_high))

    # return output
    if abs(cp - cp_mid) < precision:
        return v_mid
    else:
        raise GBS_CalculationError(
            "Implied Vol did not converge. Best Guess={0}, Price diff={1}, Required Precision={2}".format(v_mid, diff,
                                                                                                          precision))


# This is the public interface for European Options
# Each call does a little bit of processing and then calls the calculations located in the _gbs module

# Inputs:
#    option_type = "p" or "c"
#    fs          = price of underlying
#    x           = strike
#    t           = time to expiration
#    v           = implied volatility
#    r           = risk free rate
#    q           = dividend payment
#    b           = cost of carry
# Outputs:
#    value       = price of the option
#    delta       = first derivative of value with respect to price of underlying
#    gamma       = second derivative of value w.r.t price of underlying
#    theta       = first derivative of value w.r.t. time to expiration
#    vega        = first derivative of value w.r.t. implied volatility
#    rho         = first derivative of value w.r.t. risk free rates

# ---------------------------
# Black Scholes: stock Options (no dividend yield)
def black_scholes(option_type, fs, x, t, r, v):
    b = r
    return _gbs(option_type, fs, x, t, r, b, v)



# ---------------------------
# Merton Model: Stocks Index, stocks with a continuous dividend yields
def merton(option_type, fs, x, t, r, q, v):
    b = r - q
    return _gbs(option_type, fs, x, t, r, b, v)



# ---------------------------
# Commodities
def black_76(option_type, fs, x, t, r, v):
    b = 0
    return _gbs(option_type, fs, x, t, r, b, v)



# ---------------------------
# FX Options
def garman_kohlhagen(option_type, fs, x, t, r, rf, v):
    b = r - rf
    return _gbs(option_type, fs, x, t, r, b, v)



# ---------------------------
# Average Price option on commodities
def asian_76(option_type, fs, x, t, t_a, r, v):
    # Check that TA is reasonable
    if (t_a < _GBS_Limits.MIN_TA) or (t_a > t):
        raise GBS_InputError(
            "Invalid Input Averaging Time (TA = {0}). Acceptable range for inputs is {1} to <T".format(t_a,
                                                                                                       _GBS_Limits.MIN_TA))

    # Approximation to value Asian options on commodities
    b = 0
    if t_a == t:
        # if there is no averaging period, this is just Black Scholes
        v_a = v
    else:
        # Approximate the volatility
        m = (2 * math.exp((v ** 2) * t) - 2 * math.exp((v ** 2) * t_a) * (1 + (v ** 2) * (t - t_a))) / (
            (v ** 4) * ((t - t_a) ** 2))
        v_a = math.sqrt(math.log(m) / t)

    # Finally, have the GBS function do the calculation
    return _gbs(option_type, fs, x, t, r, b, v_a)



# ---------------------------
# Spread Option formula
def kirks_76(option_type, f1, f2, x, t, r, v1, v2, corr):
    # create the modifications to the GBS formula to handle spread options
    b = 0
    fs = f1 / (f2 + x)
    f_temp = f2 / (f2 + x)
    v = math.sqrt((v1 ** 2) + ((v2 * f_temp) ** 2) - (2 * corr * v1 * v2 * f_temp))
    my_values = _gbs(option_type, fs, 1.0, t, r, b, v)

    # Have the GBS function return a value
    return my_values[0] * (f2 + x), 0, 0, 0, 0, 0



# ---------------------------
# American Options (stock style, set q=0 for non-dividend paying options)
def american(option_type, fs, x, t, r, q, v):
    b = r - q
    return _american_option(option_type, fs, x, t, r, b, v)



# ---------------------------
# Commodities
def american_76(option_type, fs, x, t, r, v):
    b = 0
    return _american_option(option_type, fs, x, t, r, b, v)


# Inputs:
#    option_type = "p" or "c"
#    fs          = price of underlying
#    x           = strike
#    t           = time to expiration
#    v           = implied volatility
#    r           = risk free rate
#    q           = dividend payment
#    b           = cost of carry
# Outputs:
#    value       = price of the option
#    delta       = first derivative of value with respect to price of underlying
#    gamma       = second derivative of value w.r.t price of underlying
#    theta       = first derivative of value w.r.t. time to expiration
#    vega        = first derivative of value w.r.t. implied volatility
#    rho         = first derivative of value w.r.t. risk free rates

def euro_implied_vol(option_type, fs, x, t, r, q, cp):
    b = r - q
    return _gbs_implied_vol(option_type, fs, x, t, r, b, cp)


def euro_implied_vol_76(option_type, fs, x, t, r, cp):
    b = 0
    return _gbs_implied_vol(option_type, fs, x, t, r, b, cp)


def amer_implied_vol(option_type, fs, x, t, r, q, cp):
    b = r - q
    return _american_implied_vol(option_type, fs, x, t, r, b, cp)


def amer_implied_vol_76(option_type, fs, x, t, r, cp):
    b = 0
    return _american_implied_vol(option_type, fs, x, t, r, b, cp)


# ---------------------------
# Helper Function for Debugging

# Prints a message if running code from this module and _DEBUG is set to true
# otherwise, do nothing

def _debug(debug_input):
    if (__name__ is "__main__") and (_DEBUG is True):
        print(debug_input)


# This class defines the Exception that gets thrown when invalid input is placed into the GBS function
class GBS_InputError(Exception):
    def __init__(self, mismatch):
        Exception.__init__(self, mismatch)


# This class defines the Exception that gets thrown when there is a calculation error
class GBS_CalculationError(Exception):
    def __init__(self, mismatch):
        Exception.__init__(self, mismatch)


# This function tests that two floating point numbers are the same
# Numbers less than 1 million are considered the same if they are within .000001 of each other
# Numbers larger than 1 million are considered the same if they are within .0001% of each other
# User can override the default precision if necessary
def assert_close(value_a, value_b, precision=.000001):
    my_precision = precision

    if (value_a < 1000000.0) and (value_b < 1000000.0):
        my_diff = abs(value_a - value_b)
        my_diff_type = "Difference"
    else:
        my_diff = abs((value_a - value_b) / value_a)
        my_diff_type = "Percent Difference"

    _debug("Comparing {0} and {1}. Difference is {2}, Difference Type is {3}".format(value_a, value_b, my_diff,
                                                                                     my_diff_type))

    if my_diff < my_precision:
        my_result = True
    else:
        my_result = False

    if (__name__ is "__main__") and (my_result is False):
        print("  FAILED TEST. Comparing {0} and {1}. Difference is {2}, Difference Type is {3}".format(value_a, value_b,
                                                                                                       my_diff,
                                                                                                       my_diff_type))
    else:
        print(".")

    return my_result



if __name__ == "__main__":
    print ("=====================================")
    print ("American Options Intermediate Functions")
    print ("=====================================")

    # ---------------------------
    # unit tests for _psi()
    # _psi(FS, t2, gamma, H, I2, I1, t1, r, b, V):
    print("Testing _psi (American Option Intermediate Calculation)")
    assert_close(_psi(fs=120, t2=3, gamma=1, h=375, i2=375, i1=300, t1=1, r=.05, b=0.03, v=0.1), 112.87159814023171)
    assert_close(_psi(fs=125, t2=2, gamma=1, h=100, i2=100, i1=75, t1=1, r=.05, b=0.03, v=0.1), 1.7805459905819128)

    # ---------------------------
    # unit tests for _phi()
    print("Testing _phi (American Option Intermediate Calculation)")
    # _phi(FS, T, gamma, h, I, r, b, V):
    assert_close(
        _phi(fs=120, t=3, gamma=4.51339343051624, h=151.696096685711, i=151.696096685711, r=.02, b=-0.03, v=0.14),
        1102886677.05955)
    assert_close(_phi(fs=125, t=3, gamma=1, h=374.061664206768, i=374.061664206768, r=.05, b=0.03, v=0.14),
                 117.714544103477)

    # ---------------------------
    # unit tests for _CBND
    print("Testing _CBND (Cumulative Binomial Normal Distribution)")
    assert_close(_cbnd(0, 0, 0), 0.25)
    assert_close(_cbnd(0, 0, -0.5), 0.16666666666666669)
    assert_close(_cbnd(-0.5, 0, 0), 0.15426876936299347)
    assert_close(_cbnd(0, -0.5, 0), 0.15426876936299347)
    assert_close(_cbnd(0, -0.99999999, -0.99999999), 0.0)
    assert_close(_cbnd(0.000001, -0.99999999, -0.99999999), 0.0)

    assert_close(_cbnd(0, 0, 0.5), 0.3333333333333333)
    assert_close(_cbnd(0.5, 0, 0), 0.3457312306370065)
    assert_close(_cbnd(0, 0.5, 0), 0.3457312306370065)
    assert_close(_cbnd(0, 0.99999999, 0.99999999), 0.5)
    assert_close(_cbnd(0.000001, 0.99999999, 0.99999999), 0.5000003989422803)


# ---------------------------
# Testing American Options
if __name__ == "__main__":
    print("=====================================")
    print("American Options Testing")
    print("=====================================")

    print("testing _Bjerksund_Stensland_2002()")
    # _american_option(option_type, X, FS, T, b, r, V)
    assert_close(_bjerksund_stensland_2002(fs=90, x=100, t=0.5, r=0.1, b=0, v=0.15)[0], 0.8099, precision=.001)
    assert_close(_bjerksund_stensland_2002(fs=100, x=100, t=0.5, r=0.1, b=0, v=0.25)[0], 6.7661, precision=.001)
    assert_close(_bjerksund_stensland_2002(fs=110, x=100, t=0.5, r=0.1, b=0, v=0.35)[0], 15.5137, precision=.001)

    assert_close(_bjerksund_stensland_2002(fs=100, x=90, t=0.5, r=.1, b=0, v=0.15)[0], 10.5400, precision=.001)
    assert_close(_bjerksund_stensland_2002(fs=100, x=100, t=0.5, r=.1, b=0, v=0.25)[0], 6.7661, precision=.001)
    assert_close(_bjerksund_stensland_2002(fs=100, x=110, t=0.5, r=.1, b=0, v=0.35)[0], 5.8374, precision=.001)

    print("testing _Bjerksund_Stensland_1993()")
    # Prices for 1993 model slightly different than those presented in Haug's Complete Guide to Option Pricing Formulas
    # Possibly due to those results being based on older CBND calculation?
    assert_close(_bjerksund_stensland_1993(fs=90, x=100, t=0.5, r=0.1, b=0, v=0.15)[0], 0.8089, precision=.001)
    assert_close(_bjerksund_stensland_1993(fs=100, x=100, t=0.5, r=0.1, b=0, v=0.25)[0], 6.757, precision=.001)
    assert_close(_bjerksund_stensland_1993(fs=110, x=100, t=0.5, r=0.1, b=0, v=0.35)[0], 15.4998, precision=.001)

    print("testing _american_option()")
    assert_close(_american_option("p", fs=90, x=100, t=0.5, r=0.1, b=0, v=0.15)[0], 10.5400, precision=.001)
    assert_close(_american_option("p", fs=100, x=100, t=0.5, r=0.1, b=0, v=0.25)[0], 6.7661, precision=.001)
    assert_close(_american_option("p", fs=110, x=100, t=0.5, r=0.1, b=0, v=0.35)[0], 5.8374, precision=.001)

    assert_close(_american_option('c', fs=100, x=95, t=0.00273972602739726, r=0.000751040922831883, b=0, v=0.2)[0], 5.0,
                 precision=.01)
    assert_close(_american_option('c', fs=42, x=40, t=0.75, r=0.04, b=-0.04, v=0.35)[0], 5.28, precision=.01)
    assert_close(_american_option('c', fs=90, x=100, t=0.1, r=0.10, b=0, v=0.15)[0], 0.02, precision=.01)

    print("Testing that American valuation works for integer inputs")
    assert_close(_american_option('c', fs=100, x=100, t=1, r=0, b=0, v=0.35)[0], 13.892, precision=.001)
    assert_close(_american_option('p', fs=100, x=100, t=1, r=0, b=0, v=0.35)[0], 13.892, precision=.001)

    print("Testing valuation works at minimum/maximum values for T")
    assert_close(_american_option('c', 100, 100, 0.00396825396825397, 0.000771332656950173, 0, 0.15)[0], 0.3769,
                 precision=.001)
    assert_close(_american_option('p', 100, 100, 0.00396825396825397, 0.000771332656950173, 0, 0.15)[0], 0.3769,
                 precision=.001)
    assert_close(_american_option('c', 100, 100, 100, 0.042033868311581, 0, 0.15)[0], 18.61206, precision=.001)
    assert_close(_american_option('p', 100, 100, 100, 0.042033868311581, 0, 0.15)[0], 18.61206, precision=.001)

    print("Testing valuation works at minimum/maximum values for X")
    assert_close(_american_option('c', 100, 0.01, 1, 0.00330252458693489, 0, 0.15)[0], 99.99, precision=.001)
    assert_close(_american_option('p', 100, 0.01, 1, 0.00330252458693489, 0, 0.15)[0], 0, precision=.001)
    assert_close(_american_option('c', 100, 2147483248, 1, 0.00330252458693489, 0, 0.15)[0], 0, precision=.001)
    assert_close(_american_option('p', 100, 2147483248, 1, 0.00330252458693489, 0, 0.15)[0], 2147483148, precision=.001)

    print("Testing valuation works at minimum/maximum values for F/S")
    assert_close(_american_option('c', 0.01, 100, 1, 0.00330252458693489, 0, 0.15)[0], 0, precision=.001)
    assert_close(_american_option('p', 0.01, 100, 1, 0.00330252458693489, 0, 0.15)[0], 99.99, precision=.001)
    assert_close(_american_option('c', 2147483248, 100, 1, 0.00330252458693489, 0, 0.15)[0], 2147483148, precision=.001)
    assert_close(_american_option('p', 2147483248, 100, 1, 0.00330252458693489, 0, 0.15)[0], 0, precision=.001)

    print("Testing valuation works at minimum/maximum values for b")
    assert_close(_american_option('c', 100, 100, 1, 0, -1, 0.15)[0], 0.0, precision=.001)
    assert_close(_american_option('p', 100, 100, 1, 0, -1, 0.15)[0], 63.2121, precision=.001)
    assert_close(_american_option('c', 100, 100, 1, 0, 1, 0.15)[0], 171.8282, precision=.001)
    assert_close(_american_option('p', 100, 100, 1, 0, 1, 0.15)[0], 0.0, precision=.001)

    print("Testing valuation works at minimum/maximum values for r")
    assert_close(_american_option('c', 100, 100, 1, -1, 0, 0.15)[0], 16.25133, precision=.001)
    assert_close(_american_option('p', 100, 100, 1, -1, 0, 0.15)[0], 16.25133, precision=.001)
    assert_close(_american_option('c', 100, 100, 1, 1, 0, 0.15)[0], 3.6014, precision=.001)
    assert_close(_american_option('p', 100, 100, 1, 1, 0, 0.15)[0], 3.6014, precision=.001)

    print("Testing valuation works at minimum/maximum values for V")
    assert_close(_american_option('c', 100, 100, 1, 0.05, 0, 0.005)[0], 0.1916, precision=.001)
    assert_close(_american_option('p', 100, 100, 1, 0.05, 0, 0.005)[0], 0.1916, precision=.001)
    assert_close(_american_option('c', 100, 100, 1, 0.05, 0, 1)[0], 36.4860, precision=.001)
    assert_close(_american_option('p', 100, 100, 1, 0.05, 0, 1)[0], 36.4860, precision=.001)


# ---------------------------
# Testing European Options
if __name__ == "__main__":
    print("=====================================")
    print("Generalized Black Scholes (GBS) Testing")
    print("=====================================")

    print("testing GBS Premium")
    assert_close(_gbs('c', 100, 95, 0.00273972602739726, 0.000751040922831883, 0, 0.2)[0],
                 4.99998980469552)
    assert_close(
        _gbs('c', 92.45, 107.5, 0.0876712328767123, 0.00192960198828152, 0, 0.3)[0],
        0.162619795863781)
    assert_close(
        _gbs('c', 93.0766666666667, 107.75, 0.164383561643836, 0.00266390125346286, 0,
             0.2878)[0],
        0.584588840095316)
    assert_close(
        _gbs('c', 93.5333333333333, 107.75, 0.249315068493151, 0.00319934651984034, 0,
             0.2907)[0],
        1.27026849732877)
    assert_close(
        _gbs('c', 93.8733333333333, 107.75, 0.331506849315069, 0.00350934592318849, 0,
             0.2929)[0],
        1.97015685523537)
    assert_close(
        _gbs('c', 94.1166666666667, 107.75, 0.416438356164384, 0.00367360967852615, 0,
             0.2919)[0],
        2.61731599547608)
    assert_close(
        _gbs('p', 94.2666666666667, 107.75, 0.498630136986301, 0.00372609838856132, 0,
             0.2888)[0],
        16.6074587545269)
    assert_close(
        _gbs('p', 94.3666666666667, 107.75, 0.583561643835616, 0.00370681407974257, 0,
             0.2923)[0],
        17.1686196701434)
    assert_close(
        _gbs('p', 94.44, 107.75, 0.668493150684932, 0.00364163303865433, 0, 0.2908)[0],
        17.6038273793172)
    assert_close(
        _gbs('p', 94.4933333333333, 107.75, 0.750684931506849, 0.00355604221290591, 0,
             0.2919)[0],
        18.0870982577296)
    assert_close(
        _gbs('p', 94.49, 107.75, 0.835616438356164, 0.00346100468320478, 0, 0.2901)[0],
        18.5149895730975)
    assert_close(
        _gbs('p', 94.39, 107.75, 0.917808219178082, 0.00337464630758452, 0, 0.2876)[0],
        18.9397688539483)

    print("Testing that valuation works for integer inputs")
    assert_close(_gbs('c', fs=100, x=95, t=1, r=1, b=0, v=1)[0], 14.6711476484)
    assert_close(_gbs('p', fs=100, x=95, t=1, r=1, b=0, v=1)[0], 12.8317504425)

    print("Testing valuation works at minimum/maximum values for T")
    assert_close(_gbs('c', 100, 100, 0.00396825396825397, 0.000771332656950173, 0, 0.15)[0],
                 0.376962465712609)
    assert_close(_gbs('p', 100, 100, 0.00396825396825397, 0.000771332656950173, 0, 0.15)[0],
                 0.376962465712609)
    assert_close(_gbs('c', 100, 100, 100, 0.042033868311581, 0, 0.15)[0], 0.817104022604705)
    assert_close(_gbs('p', 100, 100, 100, 0.042033868311581, 0, 0.15)[0], 0.817104022604705)

    print("Testing valuation works at minimum/maximum values for X")
    assert_close(_gbs('c', 100, 0.01, 1, 0.00330252458693489, 0, 0.15)[0], 99.660325245681)
    assert_close(_gbs('p', 100, 0.01, 1, 0.00330252458693489, 0, 0.15)[0], 0)
    assert_close(_gbs('c', 100, 2147483248, 1, 0.00330252458693489, 0, 0.15)[0], 0)
    assert_close(_gbs('p', 100, 2147483248, 1, 0.00330252458693489, 0, 0.15)[0],
                 2140402730.16601)

    print("Testing valuation works at minimum/maximum values for F/S")
    assert_close(_gbs('c', 0.01, 100, 1, 0.00330252458693489, 0, 0.15)[0], 0)
    assert_close(_gbs('p', 0.01, 100, 1, 0.00330252458693489, 0, 0.15)[0], 99.660325245681)
    assert_close(_gbs('c', 2147483248, 100, 1, 0.00330252458693489, 0, 0.15)[0],
                 2140402730.16601)
    assert_close(_gbs('p', 2147483248, 100, 1, 0.00330252458693489, 0, 0.15)[0], 0)

    print("Testing valuation works at minimum/maximum values for b")
    assert_close(_gbs('c', 100, 100, 1, 0.05, -1, 0.15)[0], 1.62505648981223E-11)
    assert_close(_gbs('p', 100, 100, 1, 0.05, -1, 0.15)[0], 60.1291675389721)
    assert_close(_gbs('c', 100, 100, 1, 0.05, 1, 0.15)[0], 163.448023481557)
    assert_close(_gbs('p', 100, 100, 1, 0.05, 1, 0.15)[0], 4.4173615264761E-11)

    print("Testing valuation works at minimum/maximum values for r")
    assert_close(_gbs('c', 100, 100, 1, -1, 0, 0.15)[0], 16.2513262267156)
    assert_close(_gbs('p', 100, 100, 1, -1, 0, 0.15)[0], 16.2513262267156)
    assert_close(_gbs('c', 100, 100, 1, 1, 0, 0.15)[0], 2.19937783786316)
    assert_close(_gbs('p', 100, 100, 1, 1, 0, 0.15)[0], 2.19937783786316)

    print("Testing valuation works at minimum/maximum values for V")
    assert_close(_gbs('c', 100, 100, 1, 0.05, 0, 0.005)[0], 0.189742620249)
    assert_close(_gbs('p', 100, 100, 1, 0.05, 0, 0.005)[0], 0.189742620249)

    assert_close(_gbs('c', 100, 100, 1, 0.05, 0, 1)[0], 36.424945370234)
    assert_close(_gbs('p', 100, 100, 1, 0.05, 0, 1)[0], 36.424945370234)

    print("Checking that Greeks work for calls")
    assert_close(_gbs('c', 100, 100, 1, 0.05, 0, 0.15)[0], 5.68695251984796)
    assert_close(_gbs('c', 100, 100, 1, 0.05, 0, 0.15)[1], 0.50404947485)
    assert_close(_gbs('c', 100, 100, 1, 0.05, 0, 0.15)[2], 0.025227988795588)
    assert_close(_gbs('c', 100, 100, 1, 0.05, 0, 0.15)[3], -2.55380111351125)
    assert_close(_gbs('c', 100, 100, 2, 0.05, 0.05, 0.25)[4], 50.7636345571413)
    assert_close(_gbs('c', 100, 100, 1, 0.05, 0, 0.15)[5], 44.7179949651117)

    print("Checking that Greeks work for puts")
    assert_close(_gbs('p', 100, 100, 1, 0.05, 0, 0.15)[0], 5.68695251984796)
    assert_close(_gbs('p', 100, 100, 1, 0.05, 0, 0.15)[1], -0.447179949651)
    assert_close(_gbs('p', 100, 100, 1, 0.05, 0, 0.15)[2], 0.025227988795588)
    assert_close(_gbs('p', 100, 100, 1, 0.05, 0, 0.15)[3], -2.55380111351125)
    assert_close(_gbs('p', 100, 100, 2, 0.05, 0.05, 0.25)[4], 50.7636345571413)
    assert_close(_gbs('p', 100, 100, 1, 0.05, 0, 0.15)[5], -50.4049474849597)

# ---------------------------
# Testing Implied Volatility
if __name__ == "__main__":
    print("=====================================")
    print("Implied Volatility Testing")
    print("=====================================")
    print(
    "For options far away from ATM or those very near to expiry, volatility")
    print(
    "doesn't have a major effect on the price. When large changes in vol result in")
    print(
    "price changes less than the minimum precision, it is very difficult to test implied vol")
    print("=====================================")

    print ("testing at-the-money approximation")
    assert_close(
        _approx_implied_vol(option_type="c", fs=100,
                            x=100, t=1, r=.05, b=0,
                            cp=5), 0.131757)
    assert_close(
        _approx_implied_vol(option_type="c", fs=59,
                            x=60, t=0.25, r=.067,
                            b=0.067, cp=2.82), 0.239753)

    print("testing GBS Implied Vol")
    assert_close(_gbs_implied_vol('c', 92.45, 107.5,
                                  0.0876712328767123,
                                  0.00192960198828152,
                                  0, 0.162619795863781),
                 0.3)
    assert_close(
        _gbs_implied_vol('c', 93.0766666666667, 107.75,
                         0.164383561643836,
                         0.00266390125346286, 0,
                         0.584588840095316), 0.2878)
    assert_close(
        _gbs_implied_vol('c', 93.5333333333333, 107.75,
                         0.249315068493151,
                         0.00319934651984034, 0,
                         1.27026849732877), 0.2907)
    assert_close(
        _gbs_implied_vol('c', 93.8733333333333, 107.75,
                         0.331506849315069,
                         0.00350934592318849, 0,
                         1.97015685523537), 0.2929)
    assert_close(
        _gbs_implied_vol('c', 94.1166666666667, 107.75,
                         0.416438356164384,
                         0.00367360967852615, 0,
                         2.61731599547608), 0.2919)
    assert_close(
        _gbs_implied_vol('p', 94.2666666666667, 107.75,
                         0.498630136986301,
                         0.00372609838856132, 0,
                         16.6074587545269), 0.2888)
    assert_close(
        _gbs_implied_vol('p', 94.3666666666667, 107.75,
                         0.583561643835616,
                         0.00370681407974257, 0,
                         17.1686196701434), 0.2923)
    assert_close(_gbs_implied_vol('p', 94.44, 107.75,
                                  0.668493150684932,
                                  0.00364163303865433,
                                  0, 17.6038273793172),
                 0.2908)
    assert_close(
        _gbs_implied_vol('p', 94.4933333333333, 107.75,
                         0.750684931506849,
                         0.00355604221290591, 0,
                         18.0870982577296), 0.2919)
    assert_close(_gbs_implied_vol('p', 94.39, 107.75,
                                  0.917808219178082,
                                  0.00337464630758452,
                                  0, 18.9397688539483),
                 0.2876)

    print(
    "Testing that GBS implied vol works for integer inputs")
    assert_close(
        _gbs_implied_vol('c', fs=100, x=95, t=1, r=1,
                         b=0, cp=14.6711476484), 1)
    assert_close(
        _gbs_implied_vol('p', fs=100, x=95, t=1, r=1,
                         b=0, cp=12.8317504425), 1)

    print("testing American Option implied volatility")
    assert_close(
        _american_implied_vol("p", fs=90, x=100, t=0.5,
                              r=0.1, b=0, cp=10.54),
        0.15, precision=0.01)
    assert_close(
        _american_implied_vol("p", fs=100, x=100, t=0.5,
                              r=0.1, b=0, cp=6.7661),
        0.25, precision=0.0001)
    assert_close(
        _american_implied_vol("p", fs=110, x=100, t=0.5,
                              r=0.1, b=0, cp=5.8374),
        0.35, precision=0.0001)
    assert_close(
        _american_implied_vol('c', fs=42, x=40, t=0.75,
                              r=0.04, b=-0.04, cp=5.28),
        0.35, precision=0.01)
    assert_close(
        _american_implied_vol('c', fs=90, x=100, t=0.1,
                              r=0.10, b=0, cp=0.02),
        0.15, precision=0.01)

    print(
    "Testing that American implied volatility works for integer inputs")
    assert_close(
        _american_implied_vol('c', fs=100, x=100, t=1,
                              r=0, b=0, cp=13.892),
        0.35, precision=0.01)
    assert_close(
        _american_implied_vol('p', fs=100, x=100, t=1,
                              r=0, b=0, cp=13.892),
        0.35, precision=0.01)


# ---------------------------
# Testing the external interface
if __name__ == "__main__":
    print(
    "=====================================")
    print("External Interface Testing")
    print(
    "=====================================")

    # BlackScholes(option_type, X, FS, T, r, V)
    print("Testing: GBS.BlackScholes")
    assert_close(
        black_scholes('c', 102, 100, 2, 0.05,
                      0.25)[0], 20.02128028)
    assert_close(
        black_scholes('p', 102, 100, 2, 0.05,
                      0.25)[0], 8.50502208)

    # Merton(option_type, X, FS, T, r, q, V)
    print("Testing: GBS.Merton")
    assert_close(
        merton('c', 102, 100, 2, 0.05, 0.01,
               0.25)[0], 18.63371484)
    assert_close(
        merton('p', 102, 100, 2, 0.05, 0.01,
               0.25)[0], 9.13719197)

    # Black76(option_type, X, FS, T, r, V)
    print("Testing: GBS.Black76")
    assert_close(
        black_76('c', 102, 100, 2, 0.05, 0.25)[
            0], 13.74803567)
    assert_close(
        black_76('p', 102, 100, 2, 0.05, 0.25)[
            0], 11.93836083)

    # garman_kohlhagen(option_type, X, FS, T, b, r, rf, V)
    print("Testing: GBS.garman_kohlhagen")
    assert_close(
        garman_kohlhagen('c', 102, 100, 2, 0.05,
                         0.01, 0.25)[0],
        18.63371484)
    assert_close(
        garman_kohlhagen('p', 102, 100, 2, 0.05,
                         0.01, 0.25)[0],
        9.13719197)

    # Asian76(option_type, X, FS, T, TA, r, V):
    print("Testing: Asian76")
    assert_close(
        asian_76('c', 102, 100, 2, 1.9, 0.05,
                 0.25)[0], 13.53508930)
    assert_close(
        asian_76('p', 102, 100, 2, 1.9, 0.05,
                 0.25)[0], 11.72541446)

    # Kirks76(option_type, X, F1, F2, T, r, V1, V2, corr)
    print("Testing: Kirks")
    assert_close(kirks_76("c", f1=37.384913362,
                          f2=42.1774, x=3.0,
                          t=0.043055556, r=0,
                          v1=0.608063,
                          v2=0.608063, corr=.8)[
                     0], 0.007649192)
    assert_close(kirks_76("p", f1=37.384913362,
                          f2=42.1774, x=3.0,
                          t=0.043055556, r=0,
                          v1=0.608063,
                          v2=0.608063, corr=.8)[
                     0], 7.80013583)


# ---------------------------
# Testing the external interface
if __name__ == "__main__":
    print(
    "=====================================")
    print("External Interface Testing")
    print(
    "=====================================")

    # BlackScholes(option_type, X, FS, T, r, V)
    print("Testing: GBS.BlackScholes")
    assert_close(
        black_scholes('c', 102, 100, 2, 0.05,
                      0.25)[0], 20.02128028)
    assert_close(
        black_scholes('p', 102, 100, 2, 0.05,
                      0.25)[0], 8.50502208)

    # Merton(option_type, X, FS, T, r, q, V)
    print("Testing: GBS.Merton")
    assert_close(
        merton('c', 102, 100, 2, 0.05, 0.01,
               0.25)[0], 18.63371484)
    assert_close(
        merton('p', 102, 100, 2, 0.05, 0.01,
               0.25)[0], 9.13719197)

    # Black76(option_type, X, FS, T, r, V)
    print("Testing: GBS.Black76")
    assert_close(
        black_76('c', 102, 100, 2, 0.05, 0.25)[
            0], 13.74803567)
    assert_close(
        black_76('p', 102, 100, 2, 0.05, 0.25)[
            0], 11.93836083)

    # garman_kohlhagen(option_type, X, FS, T, b, r, rf, V)
    print("Testing: GBS.garman_kohlhagen")
    assert_close(
        garman_kohlhagen('c', 102, 100, 2, 0.05,
                         0.01, 0.25)[0],
        18.63371484)
    assert_close(
        garman_kohlhagen('p', 102, 100, 2, 0.05,
                         0.01, 0.25)[0],
        9.13719197)

    # Asian76(option_type, X, FS, T, TA, r, V):
    print("Testing: Asian76")
    assert_close(
        asian_76('c', 102, 100, 2, 1.9, 0.05,
                 0.25)[0], 13.53508930)
    assert_close(
        asian_76('p', 102, 100, 2, 1.9, 0.05,
                 0.25)[0], 11.72541446)

    # Kirks76(option_type, X, F1, F2, T, r, V1, V2, corr)
    print("Testing: Kirks")
    assert_close(
        kirks_76("c", f1=37.384913362,
                 f2=42.1774, x=3.0,
                 t=0.043055556, r=0,
                 v1=0.608063, v2=0.608063,
                 corr=.8)[0],
        0.007649192)
    assert_close(
        kirks_76("p", f1=37.384913362,
                 f2=42.1774, x=3.0,
                 t=0.043055556, r=0,
                 v1=0.608063, v2=0.608063,
                 corr=.8)[0],
        7.80013583)


# ------------------
# Benchmarking against other option models

if __name__ == "__main__":
    print(
    "=====================================")
    print(
    "Selected Comparison to 3rd party models")
    print(
    "=====================================")

    print("Testing GBS.BlackScholes")
    assert_close(
        black_scholes('c', fs=60, x=65, t=0.25,
                      r=0.08, v=0.30)[0],
        2.13336844492)

    print("Testing GBS.Merton")
    assert_close(
        merton('p', fs=100, x=95, t=0.5, r=0.10,
               q=0.05, v=0.20)[0],
        2.46478764676)

    print("Testing GBS.Black76")
    assert_close(
        black_76('c', fs=19, x=19, t=0.75,
                 r=0.10, v=0.28)[0],
        1.70105072524)

    print("Testing GBS.garman_kohlhagen")
    assert_close(
        garman_kohlhagen('c', fs=1.56, x=1.60,
                         t=0.5, r=0.06, rf=0.08,
                         v=0.12)[0],
        0.0290992531494)

    print("Testing Delta")
    assert_close(
        black_76('c', fs=105, x=100, t=0.5,
                 r=0.10, v=0.36)[1], 0.5946287)
    assert_close(
        black_76('p', fs=105, x=100, t=0.5,
                 r=0.10, v=0.36)[1], -0.356601)

    print("Testing Gamma")
    assert_close(
        black_scholes('c', fs=55, x=60, t=0.75,
                      r=0.10, v=0.30)[2],
        0.0278211604769)
    assert_close(
        black_scholes('p', fs=55, x=60, t=0.75,
                      r=0.10, v=0.30)[2],
        0.0278211604769)

    print("Testing Theta")
    assert_close(
        merton('p', fs=430, x=405, t=0.0833,
               r=0.07, q=0.05, v=0.20)[3],
        -31.1923670565)

    print("Testing Vega")
    assert_close(
        black_scholes('c', fs=55, x=60, t=0.75,
                      r=0.10, v=0.30)[4],
        18.9357773496)
    assert_close(
        black_scholes('p', fs=55, x=60, t=0.75,
                      r=0.10, v=0.30)[4],
        18.9357773496)

    print("Testing Rho")
    assert_close(
        black_scholes('c', fs=72, x=75, t=1,
                      r=0.09, v=0.19)[5],
        38.7325050173)