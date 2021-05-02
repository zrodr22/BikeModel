import numpy as np
import scipy as sp
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

def estInitialize():
    # Fill in whatever initialization you'd like here. This function generates
    # the internal state of the estimator at time 0. You may do whatever you
    # like here, but you must return something that is in the format as may be
    # used by your estRun() function as the first returned variable.
    #
    # The second returned variable must be a list of student names.
    # 
    # The third return variable must be a string with the estimator type

    #we make the internal state a list, with the first three elements the position
    # x, y; the angle theta; and our favorite color. 
    x = 0
    y = 0
    theta = np.pi/4
    # variance = 1/9*np.diag([1,1,(np.pi/4)**2])
    variance = 1/9*np.diag([0.1,0.1,(0.1)**2])

    Np = 100
    x_samples = np.random.normal(x,variance[0,0],Np)
    y_samples = np.random.normal(y,variance[1,1],Np)
    theta_samples = np.random.normal(theta,variance[2,2],Np)
    xm_particles = np.array([x_samples, y_samples, theta_samples])

    # color = 'green' 
    # note that there is *absolutely no prescribed format* for this internal state.
    # You can put in it whatever you like. Probably, you'll want to keep the position
    # and angle, and probably you'll remove the color.
    internalState = [x,
                     y,
                     theta, 
                     variance,
                     xm_particles
                    #  color
                     ]

    # replace these names with yours. Delete the second name if you are working alone.
    studentNames = ['Brett Bussell',
                    'Zachariah Rodriquez',
                    'Miles Luhn']
    
    # replace this with the estimator type. Use one of the following options:
    #  'EKF' for Extended Kalman Filter
    #  'UKF' for Unscented Kalman Filter
    #  'PF' for Particle Filter
    #  'OTHER: XXX' if you're using something else, in which case please
    #                 replace "XXX" with a (very short) description
    estimatorType = 'EKF'  
    
    return internalState, studentNames, estimatorType

