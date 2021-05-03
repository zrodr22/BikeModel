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
        
    # Particle Filter
    if estimator == 'PF':

        # Probability distribution for W 
        def normal(x,sigma,mu):
            return 1/(np.sqrt(np.linalg.det(sigma)*(2*np.pi)**2))*np.exp(-1/2*(x-mu).T@np.linalg.inv(sigma)@(x-mu))
        def fzx(z,h):
            return normal(z-h,np.sqrt(W),np.zeros((2,1)))

        # Obtain particles from internal state
        particles = internalStateIn[4]
        x_samples = particles[0]
        y_samples = particles[1]
        theta_samples = particles[2]
        B_samples = particles[3]
        r_samples = particles[4]

        # Calculate V samples for process noise
        Np = len(x_samples)
        vval = 1/3*0.1
        v1_samples = np.random.normal(0,vval,Np)
        v2_samples = np.random.normal(0,vval,Np)
        v3_samples = np.random.normal(0,vval,Np)

        # Pass simulated particles into Dynamics
        xp = x_samples + 5*r_samples*pedalSpeed*np.cos(theta_samples)*dt + v1_samples
        yp = y_samples + 5*r_samples*pedalSpeed*np.sin(theta_samples)*dt + v2_samples
        thetap = theta_samples + dt*(5*r_samples*pedalSpeed/B_samples)*np.tan(steeringAngle) + v3_samples
        Bp = B_samples
        rp = r_samples

        # Calculate position measurement
        h = np.array([[ xp + 0.5*Bp*np.cos(thetap) ],
                    [ yp + 0.5*Bp*np.sin(thetap) ]])

        h = np.vstack((h[0][0],h[1][0]))
        fzxs = []

        if not (np.isnan(measurement[0]) or np.isnan(measurement[1])):
            # Access measurements (z)
            x_meas = measurement[0][0]
            y_meas = measurement[1][0]
            z = np.array([[x_meas],[y_meas]])

            # Posteriori Update
            for i in range(len(h[0])):
                h_val = np.reshape(h[:,i],(2,1))
                fzxs.append(fzx(z,h_val))

            fzxs = np.array(fzxs)
            alpha = 1/np.sum(fzxs)
            beta_n = alpha*fzxs
            
            # Resample from the predicted state
            n_bar = []
            beta_cumsum = np.cumsum(beta_n)
            for n in range(Np):
                r = np.random.uniform(0,1)
                m = np.where(r <= beta_cumsum)[0][0]
                # m = np.min(np.where(r < beta_cumsum))
                n_bar.append(m)
            
            xm_poster = xp[n_bar]
            ym_poster = yp[n_bar]
            thetam_poster = thetap[n_bar]
            B_poster = Bp[n_bar]
            r_poster = rp[n_bar]

            # Find averages
            x = np.average(xm_poster)
            y = np.average(ym_poster)
            theta = np.average(thetam_poster)
            B = np.average(B_poster)
            r = np.average(r_poster)

            # Update internal state
            variance = np.diag([np.var(xm_poster), np.var(ym_poster), np.var(thetam_poster)])
            xm_particles = np.array([xm_poster, ym_poster, thetam_poster, B_poster, r_poster])

        else:
            x = np.average(xp)
            y = np.average(yp)
            theta = np.average(thetap)
            B = np.average(Bp)
            r = np.average(rp)
            variance = np.diag([np.var(x), np.var(y), np.var(theta)])
            xm_particles = np.array([xp, yp, thetap, Bp, rp])

    #### OUTPUTS ####
    # Update the internal state (will be passed as an argument to the function
    # at next run), must obviously be compatible with the format of
    # internalStateIn:
    internalStateOut = [x,
                        y,
                        theta, 
                        variance,
                        xm_particles
                        ]

    # DO NOT MODIFY THE OUTPUT FORMAT:
    return x, y, theta, internalStateOut 


