#
# Discrete Event Fourier Transform
# Author: Allan McInnes 
#
# The MIT License (MIT)

# Copyright (c) 2010 Allan McInnes

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from scipy import zeros, sqrt, exp, pi, sinc, floor
from scipy import arange, array, asarray, append
from numpy import complex, obj2sctype

def point_rule_deft(x, w):
    """
    Return a point-rule Fourier transform of a discrete event sequence x.
    The sequence x is assumed to define a possibly non-uniformly sampled
    time-domain signal. 
    
    See Bland, D.M., Laakso, T.I., and Tarczynski, A. "Analysis of Algorithms
    for Nonuniform-Time Discrete Fourier Transform", 1996 IEEE International 
    Symposium on Circuits and Systems, DOI 10.1109/ISCAS.1996.541744.
    
    Parameters
    ----------
    x : event array
        array to transform, where each event in the array is a 2-tuple
        (time, value)
    w : array
        array of radian frequencies over which the transform
        should be evaluated
        
    Returns
    -------
    z : complex array
        with the elements
            z(k) = sum[n=0..N-1] (x[n][1] * exp(-j*w[k]*x[n][0]))
    """
    z = zeros(len(w), dtype=complex)
    for k in range(len(z)):
        for n in range(len(x)):
            (t, v) = x[n]
            z[k] = z[k] + v * exp(-1j*w[k]*t)
    return z
    
def rectangle_rule_deft(x, T, w):
    """
    Return a rectangle-rule Fourier transform of a discrete event sequence x.
    The sequence x is assumed to define a possibly non-uniformly sampled
    time-domain signal. 
    
    The rectangle-rule FT is a numerical integration technique for finding
    the FT of a sequence. It is implicit in the discussion of numerical
    integration techniques found in Bland et al., although we use zero-order
    polynomial interpolants since those are most appropriate for discrete-state
    systems.
    
    This version of the DEFT assumes a finite signal duration,
    with the signal assumed to return to its zero state at the end of that
    duration (just as we assume that the signal starts at a zero state).
    
    See Bland, D.M., Laakso, T.I., and Tarczynski, A. "Analysis of Algorithms
    for Nonuniform-Time Discrete Fourier Transform", 1996 IEEE International 
    Symposium on Circuits and Systems, DOI 10.1109/ISCAS.1996.541744.
    
    Parameters
    ----------
    x : event array
        array to transform, where each event in the array is a 2-tuple
        (time, value)
    T : signal termination time
    w : array
        array of radian frequencies over which the transform
        should be evaluated
        
    Returns
    -------
    z : complex array
        with the elements
            z(k) = sum[n=0..N-2] 0.5 * x[n][1]
                                     * ((exp(-j*w[k]*x[n][0]) +
                                         exp(-j*w[k]*x[n+1][0])) 
                                    * (x[n+1][0] - x[n][0]))
    """
    assert T >= x[-1][0], \
        "Termination tag T = {0:g} must be greater".format(T) + \
        " than or equal to the tag of the final event in the signal, " + \
        "t = {0:g}. ".format(x[-1][0]) + \
        "However, T is {0:g} less than t.".format(x[-1][0] - T)
        
    z = zeros(len(w), dtype=complex)
    xf = append(x, [(T, 0.0)], axis=0)   # Append a terminating zero event
    for k in range(len(z)):
        for n in range(len(x)):
            (t0, v0) = xf[n]
            (t1, v1) = xf[n+1]
            z[k] = z[k] + 0.5 * v0 * (exp(-1j*w[k]*t0) + exp(-1j*w[k]*t1)) \
                * (t1 - t0)
    return z

def trapezoid_rule_deft(x, T, w):
    """
    Return a trapezoid-rule Fourier transform of a discrete event sequence x.
    The sequence x is assumed to define a possibly non-uniformly sampled
    time-domain signal. 
    
    The trapezoid-rule FT is a numerical integration technique for finding
    the FT of a sequence found in Bland et al.
    
    This version of the DEFT assumes a finite signal duration,
    with the signal assumed to return to its zero state at the end of that
    duration (just as we assume that the signal starts at a zero state).
    
    See Bland, D.M., Laakso, T.I., and Tarczynski, A. "Analysis of Algorithms
    for Nonuniform-Time Discrete Fourier Transform", 1996 IEEE International 
    Symposium on Circuits and Systems, DOI 10.1109/ISCAS.1996.541744.
    
    Parameters
    ----------
    x : event array
        array to transform, where each event in the array is a 2-tuple
        (time, value)
    T : signal termination time
    w : array
        array of radian frequencies over which the transform
        should be evaluated
        
    Returns
    -------
    z : complex array
        with the elements
            z(k) = sum[n=0..N-2] 0.5 * ((x[n][1] * exp(-j*w[k]*x[n][0]) 
                                        + (x[n+1][1] * exp(-j*w[k]*x[n+1][0])) 
                                     * (x[n+1][0] - x[n][0]))
    """
    assert T >= x[-1][0], \
        "Termination tag T = {0:g} must be greater".format(T) + \
        " than or equal to the tag of the final event in the signal, " + \
        "t = {0:g}. ".format(x[-1][0]) + \
        "However, T is {0:g} less than t.".format(x[-1][0] - T)
        
    z = zeros(len(w), dtype=complex)
    xf = append(x, [(T, 0.0)], axis=0)   # Append a terminating zero event
    for k in range(len(z)):
        for n in range(len(x)):
            (t0, v0) = xf[n]
            (t1, v1) = xf[n+1]
            z[k] = z[k] + 0.5 * (v0*exp(-1j*w[k]*t0) \
                + v1*exp(-1j*w[k]*t1)) * (t1 - t0)
    return z    
    
def sinc_deft(x, T, w):
    """
    Return a sinc-based Fourier transform of a discrete event sequence x.
    The sequence x is assumed to define a possibly non-uniformly sampled
    time-domain signal. 
    
    This approach is a semi-analytical technique that essentially works by 
    treating the DE sequence as defining a piecewise-constant signal, and 
    treating the FT as the sum of the FTs of the rectangular pulses that make up 
    that piecewise-constant signal.

    This version of the DEFT assumes a finite signal duration,
    with the signal assumed to return to its zero state at the end of that
    duration (just as we assume that the signal starts at a zero state).
        
    The rect function is defined as 
    
        rect(t/tau) = 0 if |t| > tau/2
                      1 if |t| < tau/2
                    
    and has Fourier transform
    
        F(w) = tau * sinc(w*tau/(2*pi))
    
    Given events e0 = (t0, v0) and e1 = (t1, v1) we treat them as defining
    a rectangle v0 * rect((t - ((t1 - t0)/2 + t0))/|t1 - t0|) with FT
    
        F(w) = v0 * exp(-j*w*(t0+t1)/2) * tau * sinc(w*tau/(2*pi))
        
    where tau = |t1 - t0|

    See McInnes, A. "A Discrete Event Fourier Transform", https://github.com/allanmcinnes/DEFT.
    
    Parameters
    ----------
    x : event array
        array to transform, where each event in the array is a 2-tuple
        (time, value)
    T : signal termination time
    w : array
        array of radian frequencies over which the transform
        should be evaluated
        
    Returns
    -------
    z : complex array
        z = sum[n=0..N] v0 * exp(-1j*w*(t0 + t1)/2) * tau * sinc(w*tau/(2*pi))
        where
            (t0, v0) = x[n]
            (t1, v1) = x[n+1]
            tau = t1 - t0
    """
    assert T >= x[-1][0], \
        "Termination tag T = {0:g} must be greater".format(T) + \
        " than or equal to the tag of the final event in the signal, " + \
        "t = {0:g}. ".format(x[-1][0]) + \
        "However, T is {0:g} less than t.".format(x[-1][0] - T)
        
    z = zeros(len(w), dtype=complex)
    xf = append(x, [(T, 0.0)], axis=0)   # Append a terminating zero event
    for n in range(len(x)):
        (t0, v0) = xf[n]
        (t1, v1) = xf[n+1]
        tau = abs(t1 - t0)
        z = z + (v0 * exp(-1j*w*(t0 + tau/2)) * tau * sinc(w*tau/(2*pi)))
    return asarray(zip(w,z))
    
def periodic_sinc_deft(x, max_w):
    """
    Return a sinc-based Fourier transform of a discrete event sequence x.
    The sequence x is assumed to define a possibly non-uniformly sampled
    periodic time-domain signal. 
    
    This approach is a semi-analytical technique that essentially works by 
    treating the DE sequence as defining a piecewise-constant signal, and 
    treating the FT as the sum of the FTs of the rectangular pulses that make up 
    that piecewise-constant signal.

    This version of the DEFT assumes a periodic signal, and like the standard
    DFT constructs the transform from a finite segment of the signal by
    assuming a periodic extension of the signal. The result is essentially a 
    frequency sampling of the nonperiodic DEFT at frequencies that are multiples
    of the fundamental frequency.
        
    The rect function is defined as 
    
        rect(t/tau) = 0 if |t| > tau/2
                      1 if |t| < tau/2
                    
    and has Fourier transform
    
        F(w) = tau * sinc(w*tau/(2*pi))
    
    Given events e0 = (t0, v0) and e1 = (t1, v1) we treat them as defining
    a rectangle rect((t - ((t1 - t0)/2 + t0))/|t1 - t0|) with FT
    
        F(w) = v0 * exp(-j*w*(t0+t1)/2) * tau * sinc(w*tau/(2*pi))
        
    where tau = |t1 - t0|
    
    Parameters
    ----------
    x : event array
        array to transform, where each event in the array is a 2-tuple
        (time, value)
    max_w : maximum radian frequency over which to evaluate the transform
        
    Returns
    -------
    z : complex array
        z = sum[n=0..N] v0 * exp(-1j*w*(t0 + t1)/2) * tau * sinc(w*tau/(2*pi))
        where
            (t0, v0) = x[n]
            (t1, v1) = x[n+1]
            tau = t1 - t0
            P = max(t) - min(t)
            w = 2*pi*k/P for k in -max_w*P/(2*pi) .. max_w*P/(2*pi)
    """
    ts = [t for (t,v) in x]
    P = max(ts) - min(ts)     # Period
    w0 = (2*pi)/P             # Fundamental radian frequency
    T = max(ts)               # Signal end point
    # Construct frequency sample points
    k_bound = floor(max_w/w0)
    w = w0*arange(-k_bound, k_bound+1, dtype=complex)
    z = zeros(len(w), dtype=complex)
    xf = append(x, [(T, x[0,1])], axis=0)   # manage the wraparound
    for n in range(len(x)):
        (t0, v0) = xf[n]
        (t1, v1) = xf[n+1]
        tau = abs(t1 - t0)
        z = z + (v0 * exp(-1j*w*(t0 + tau/2)) * tau * sinc(w*tau/(2*pi)))
    return asarray(zip(w,z/P))
    
