#
# DEFT examples
# Author: Allan McInnes 2010
#

from scipy import pi, floor, array, arange
from scipy import sin, arcsin, fmod, absolute, sinc, sign 
from deft import point_rule_deft, rectangle_rule_deft 
from deft import sinc_deft, periodic_sinc_deft
from matplotlib.pyplot import *
import numpy as np

from matplotlib import rc, rcParams

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# These lines are needed to get type-1 results:
# http://nerdjusttyped.blogspot.com/2010/07/type-1-fonts-and-matplotlib-figures.html
rcParams['ps.useafm'] = True
rcParams['pdf.use14corefonts'] = True
rcParams['pdf.fonttype'] = 42
rcParams['text.usetex'] = False


# Single event at 0    
test_seq_1 = array([(0.0, 1.0)])  
# Rectangular pulse centered at 0.0
test_seq_2 = array([(-0.5, 1.0), (0.5, 0.0)])  
# Rectangular pulse centered at 0.5
test_seq_3 = array([(0.0, 1.0), (1.0, 0.0)])  

# Roughly quantized triangle
test_seq_4 = array([(-1.0, 1.0), (-0.5, 2.0), (0.5, 1.0), (1.0, 0.0)])
# Rect pulse for use in periodic signal
test_seq_5 = array([(-1.0, 6.0), (-0.5,6.0),(0.5,6.0),(1.0, 0.0), (11.0,6.0)])

# Finely quantized triangle
def triangle_seq(delta = 0.01):
    t = -1.0
    v = -1.0
    slope = 2
    events = []
    while t < 0:
        v += delta
        events += [(t, v)]
        t += delta / abs(slope)       
    slope = -2
    while t <= 1.0:
        v -= delta
        t += delta / abs(slope)
        events += [(t, v)]
    return array(events)
    
# Approximate quantised sinc    
def sinc_seq(A = 1.0, tf = 1.0, delta = 0.1):
    t = -tf
    dt = 0.001
    events = []
    v_last = None
    # Note use of round() to compensate for floating-point arithmetic errors
    # that lead to inexact results.
    while t <= tf:
        v = delta * floor(round(A*sinc(t) / delta, 10))
        if v != v_last:
            events += [(t, v)]
            v_last = v          
        t += dt
    return array(events)    
    
# Quantised sine wave    
def sin_seq(A = 1.0, f = 1.0, T = 1.0, delta = 0.1):
    t = 0.0
    events = []
    # Note use of round() to compensate for floating-point arithmetic errors
    # that lead to inexact results.
    while t <= T:
        v = delta * floor(round(A*sin(2*pi*f*t) / delta,10))
        events += [(t, v)]
        tm = fmod(abs(t), 0.5/f)
        if tm < 0.25/f:
            vq = delta * (floor(round(A*sin(2*pi*f*tm) / delta,10)) + 1.0)
            dt = arcsin(vq/A)/(2*pi*f) - tm
        else:
            vq = delta * (floor(round(A*sin(2*pi*f*tm) / delta,10)) - 1.0)
            dt = (pi - arcsin(vq/A))/(2*pi*f) - tm             
        t += dt
    return array(events)
    
# Convert a sequence of events to a set of points for plotting continuous lines
# The result essentially represents the state trajectory defined by the
# events.
def state_trajectory(events):
    times = np.zeros(len(events)*2, dtype=np.float)
    values = np.zeros(len(events)*2, dtype=np.float)
    for ev in range(len(events)-1):
        times[2*ev + 1] = events[ev][0]
        times[2*ev + 2] = events[ev+1][0]
        values[2*ev + 1] = events[ev][1]
        values[2*ev + 2] = events[ev][1]

    times[0] = times[1]
    times[-1] = events[-1][0]
    values[0] = values[1]
    values[-1] = events[-1][1]

    return (times, values)

# See http://matplotlib.sourceforge.net/users/customizing.html
import matplotlib as mpl
mpl.rc('lines', linewidth=2.5, markersize=8)
mpl.rc('font', size=14)


figure()
x = test_seq_3
(t,v) = state_trajectory(x) # States
plot(t,v,'g--')
stem(x[:,0], x[:,1])
xlabel('Time (s)')
ylabel('Value')
plot([-0.1, 1.2], [0, 0], 'r')
title('Discrete event rectangular pulse')
axis([-0.1, 1.2, -0.1, 1.2])

w = 2*pi*arange(-10,10,0.2)   

figure()
z3 = sinc_deft(x, 1.0, w)
stem(z3[:,0]/(2*pi),absolute(z3[:,1]))
xlabel('Frequency (Hz)')
ylabel('Magnitude')
ax = axis()
axis([ax[0], ax[1], ax[2], 1.1*ax[3]])
title('DEFT of DE rectangular pulse')


# Sinc example
figure()
#(t1,v1) = state_trajectory(sinc_seq(delta=0.0001)) # Continuous trajectory
#plot(t1,v1,'k:')
ev = sinc_seq(delta = 0.05, tf = 30.0)
(t,v) = state_trajectory(ev) # States
plot(t,v,'g--')
stem(ev[:,0], ev[:,1])       # Events
xlabel('Time (s)')
ylabel('Value')
ax = axis()
axis([-4,4,-0.3,1.1])
title('Discrete event sinc function')


w = 2*pi*arange(-2,2,0.05)   

figure()
z3 = sinc_deft(ev, 30.0, w)
stem(z3[:,0]/(2*pi),absolute(z3[:,1]))
xlabel('Frequency (Hz)')
ylabel('Magnitude')
ax = axis()
axis([ax[0], ax[1], ax[2], 1.1*ax[3]])
title('DEFT of DE sinc function')

figure()
ev = sinc_seq(delta = 0.001, tf = 30.0)
z3 = sinc_deft(ev, 30.0, w)
stem(z3[:,0]/(2*pi),absolute(z3[:,1]))
xlabel('Frequency (Hz)')
ylabel('Magnitude')
ax = axis()
axis([ax[0], ax[1], ax[2], 1.1*ax[3]])
title('DEFT of DE sinc function')

# Sinusoid example
figure()
(t1,v1) = state_trajectory(sin_seq(delta=0.0001)) # Continuous trajectory
plot(t1,v1,'k:')
ev = sin_seq(delta=0.1)
(t,v) = state_trajectory(ev) # States
plot(t,v,'g--')
stem(ev[:,0], ev[:,1])       # Events
xlabel('Time (s)')
ylabel('Value')
ax = axis()
axis([ax[0], ax[1], 1.1*ax[2], 1.1*ax[3]])
title('Discrete event sinusoid')

figure()
 # Look at the first 25 Hz
z4 = periodic_sinc_deft(ev, 2*pi*25)
stem(z4[:,0]/(2*pi),abs(z4[:,1]))
xlabel('Frequency (Hz)')
ylabel('Magnitude')
ax = axis()
axis([ax[0], ax[1], ax[2], 1.1*ax[3]])
title('Periodic DEFT (1 period of DE sinusoid)')

# More periods give better approximation
figure()
ev = sin_seq(delta=0.1, T=5)
z6 = periodic_sinc_deft(ev, 2*pi*25)
stem(z6[:,0]/(2*pi),abs(z6[:,1]))
xlabel('Frequency (Hz)')
ylabel('Magnitude')
ax = axis()
axis([ax[0], ax[1], ax[2], 1.1*ax[3]])
title('Periodic DEFT (5 periods of DE sinusoid)')

show()



