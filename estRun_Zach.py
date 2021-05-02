import numpy as np
import scipy as sp
#NO OTHER IMPORTS ALLOWED (However, you're allowed to import e.g. scipy.linalg)

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

    # estimator = 'EKF'
    estimator = 'PF'
    r = 0.425
    B = 0.8

    V = np.eye(3)
    W = np.diag([1,3])

    # Example code only, you'll want to heavily modify this.
    # this internal state needs to correspond to your init function:
    x = internalStateIn[0]
    y = internalStateIn[1]
    theta = internalStateIn[2]
    variance = internalStateIn[3]

    state = np.array([[x],[y],[theta]])
    measurement = np.array([[measurement[0]],[measurement[1]]])

    if estimator == 'EKF':

        # Prior update
        A = np.array([[ 1, 0, -5*r*pedalSpeed*np.sin(theta)*dt ],
                      [ 0, 1,  5*r*pedalSpeed*np.cos(theta)*dt ],
                      [ 0, 0,  1                               ] ])
        
        L = np.eye(3)

        xp = x + 5*r*pedalSpeed*np.cos(theta)*dt
        yp = y + 5*r*pedalSpeed*np.sin(theta)*dt
        thetap = theta + dt*(5*r*pedalSpeed/B)*np.tan(steeringAngle)
        variancep = A @ variance @ A.T + L @ V @ L.T

    #x = x + pedalSpeed
    #y = y + pedalSpeed

        # Measurement update
        if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
            # have a valid measurement
            H = np.array([[ 1, 0, -0.5*B*np.sin(theta) ],
                          [ 0, 1,  0.5*B*np.cos(theta) ] ])
            M = np.eye(2)

            K = variancep @ H.T @ np.linalg.inv( H @ variancep @ H.T + M @ W @ M.T )
            measurementp = np.array([[ xp + 0.5*B*np.cos(thetap) ],
                                     [ yp + 0.5*B*np.sin(thetap) ] ])
                                     
            statem = state + K @ ( measurement - measurementp )
            variancem = (np.eye(3) - K @ H ) @ variancep

            x = statem[0].item()
            y = statem[1].item()
            theta = statem[2].item()
            variance = variancem

        else:
            x = xp
            y = yp
            theta = thetap
            variance = variancep
        
########################################################################
    if estimator == 'PF':
        # Particle Filter

        # Initial Particles
        Nps = [1,10,100,1000]
        # Number of iterations
        num_times = 100
        avgs = {1: [], 10: [], 100: [], 1000: []}

        def normal(x,sigma,mu):
            return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((x-mu)/sigma)**2)

        def fzx(z,xp):
            return normal(z-xp,1,0)

        # for Np in Nps:
        #     zeros = np.zeros((3,Np))
        for s in range(num_times):
            x_samples = np.random.normal(0,1,(4,Np))
            k = 1
            for z in zs:
                v_samples = np.random.normal(0,1,Np)
                v_samples = np.vstack((zeros, v_samples))

                xp = x_samples + v_samples 
                
                fzxs = np.array([fzx(z,x) for x in xp[0]])
                alpha = 1/np.sum(fzxs)
                
                beta_n = alpha*fzxs
                
                n_bar = []
            
                sum = np.cumsum(beta_n)
                for n in range(Np):
                    r = np.random.uniform(0,1)
                    m = np.where(r <= sum)[0][0]
                    n_bar.append(m)
                
                xm_poster = xp[:,n_bar]
                x_samples = xm_poster

                if k == 5:
                    avg = np.mean(xm_poster,axis=1)
                    avgs[Np].append(avg)
                
                k += 1
    
#############################################################################            

    #we're unreliable about our favourite colour: 
    # if myColor == 'green':
    #     myColor = 'red'
    # else:
    #     myColor = 'green'


    #### OUTPUTS ####
    # Update the internal state (will be passed as an argument to the function
    # at next run), must obviously be compatible with the format of
    # internalStateIn:
    internalStateOut = [x,
                        y,
                        theta, 
                        variance
                        ]

    # DO NOT MODIFY THE OUTPUT FORMAT:
    return x, y, theta, internalStateOut 


