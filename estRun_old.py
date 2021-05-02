import numpy as np
import scipy as sp
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

V = 1
L = 0

def calc_xp(xm):
  x1 = xm[0]+xm[1]
  x2 = xm[1]-1/10*xm[1]+0
  return np.array([x1, x2]).T

def calc_Pp(A,Pm,L,V):
  return A@Pm@A+L*V*L

def calc_A(xm):
  return np.array([[1, 1],[0, 1-3/10*xm[1]**2]])



# Extended Kalman Filter

def estRun(time, dt, internalStateIn, steeringAngle, pedalSpeed, measurement):
    # In this function you implement your estimator. The function arguments
    # are:
    #  time: current time in [s] 
    #  dt: current time step [s]
    #  internalStateIn: the estimator internal state, definition up to you. 
    #  steeringAngle: the steering angle of the bike, gamma, [rad] 
    #  pedalSpeed: the rotational speed of the pedal, omega, [rad/s] 
    #  measurement: the position measurement valid at the current time step
    #
    # Note: the measurement is a 2D vector, of x-y position measurement.
    #  The measurement sensor may fail to return data, in which case the
    #  measurement is given as NaN (not a number).
    #
    # The function has four outputs:
    #  x: your current best estimate for the bicycle's x-position
    #  y: your current best estimate for the bicycle's y-position
    #  theta: your current best estimate for the bicycle's rotation theta
    #  internalState: the estimator's internal state, in a format that can be understood by the next call to this function

    # Example code only, you'll want to heavily modify this.
    # this internal state needs to correspond to your init function:

    xm0 = np.array([1,2]).T
    Pm0 = np.diag([4,9])

    A = calc_A(xm0)

    x1 = calc_xp(xm0)
    Pp1 = calc_Pp(A,Pm0,L,V)

    x = internalStateIn[0]
    y = internalStateIn[1]
    theta = internalStateIn[2]
    myColor = internalStateIn[3]

    x = x + pedalSpeed
    y = y + pedalSpeed

    if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
        # have a valid measurement
        x = measurement[0]
        y = measurement[1]
        theta = theta + 1
        

    #we're unreliable about our favourite colour: 
    if myColor == 'green':
        myColor = 'red'
    else:
        myColor = 'green'


    #### OUTPUTS ####
    # Update the internal state (will be passed as an argument to the function
    # at next run), must obviously be compatible with the format of
    # internalStateIn:
    internalStateOut = [x,
                     y,
                     theta, 
                     myColor
                     ]

    # DO NOT MODIFY THE OUTPUT FORMAT:
    return x, y, theta, internalStateOut 


