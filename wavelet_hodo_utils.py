from wavelet_functions import wavelet, wave_signif
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Data_Methods:

    def __init__(self):
        self.order = 2

    def _remove_background(self, array, time):
        ''' Removes a polynomial fit from whatever time series 
            'array' is.
        '''

        order = self.order
        fit = np.poly1d(np.polyfit(time, array, order))
        return array - fit(time)


    def _get_data(self, data_file):
        self.df = pd.read_table(data_file, sep = '\t', skiprows = 18, encoding = 'latin1', skipfooter = 10, engine = 'python')
        self.df = self.df.drop([0]) # drops the row that holds units of measured quantities
        self.df.columns = self.df.columns.str.replace(' ', '')
        alt = np.asarray(pd.to_numeric(self.df['Alt']))
        max_alt_index = list(alt).index(max(alt))
        alt = alt[:max_alt_index]
        time = np.asarray(pd.to_numeric(self.df['Time'])[:max_alt_index])
        ws = np.asarray(pd.to_numeric(self.df['Ws'])[:max_alt_index])
        wd = np.asarray(pd.to_numeric(self.df['Wd'])[:max_alt_index])
        u = ws*np.cos(np.deg2rad(wd))
        v = ws*np.sin(np.deg2rad(wd))
        temp = np.asarray(pd.to_numeric(self.df['T'])[:max_alt_index])
        t = self._remove_background(temp, time)
        u = self._remove_background(u, time)
        v = self._remove_background(v, time)
        return alt, time, u, v, temp
    

    @staticmethod
    def eta(u, temp, alt):
        ''' Comutes 'eta' over an altitude range.
            From Marlton, Williams, and Nicoll, 2015 
        '''
        dtdz = np.diff(temp)/np.diff(alt)
        eta = np.mean(u[:-1]*dtdz)
        return eta


    @staticmethod
    def _get_altitude_window(arange, data_array, alt_array):
        data_windowed = [data_array[i] for i, a in enumerate(alt_array) if a > arange[0] and a < arange[1]]
        return np.asarray(data_windowed)


class Gravity_Wave_Data:

    ''' 
    Gravity wave parameters:
    w/f
    propagation direction
    altitude of maximum power in the wavelet power spectrum
    '''
    def __init__(self):
        self.row_index = 0
        self.columns = ['launch', 'wf_stokes', 'wf_hodograph', 'prop_dir_stokes', 'prop_dir_hodograph', 'altitude', 'bottom_scale', 'top_scale', 'bottom_alt', 'top_alt']
        self.data_array = []

    def add_row_of_data(self, launch, wf_stokes, wf_hodograph, prop_dir_stokes, prop_dir_hodograph, altitude, bottom_scale, top_scale, bottom_alt, top_alt):
        data_row = np.array([launch, wf_stokes, wf_hodograph, prop_dir_stokes, prop_dir_hodograph, altitude, bottom_scale, top_scale, bottom_alt, top_alt])
        self.data_array.append(data_row)

    
    def dump_data(self, out_file):
        try:
            df = pd.DataFrame(self.data_array, columns = self.columns)
            df.to_pickle(out_file)
        
        except Exception as e:
            print(e)


class Wavelet(Data_Methods):

    def __init__(self, data_file):
        super(Wavelet, self).__init__()
        self.data_file = data_file
        self.mother = 'MORLET' # wavelet choice to use.
        self.pad = 1
        self.lag1 = 0.72
        self.alt, self.time, u, v, self.temp = self._get_data(data_file)
        self.dj = 0.01
        self.dt = np.diff(self.time)[0]
        self.s0 = 2*self.dt
        self.j1 = np.log2(len(u)*self.dt/self.s0)/self.dj
        self.u = self._remove_background(u, self.time)
        self.v = self._remove_background(v, self.time)
        self.u_wavelet, self.fourier_period, self.scale, self.ucoi = \
        wavelet(u, self.dt, self.pad, self.dj, self.s0, self.j1, self.mother)
        self.v_wavelet, self.fourier_period, self.scale, self.vcoi = \
        wavelet(v, self.dt, self.pad, self.dj, self.s0, self.j1, self.mother)

        self.power_spectrum = abs(self.u_wavelet)**2 + abs(self.v_wavelet)**2


    def invert_wavelet_coefficients(self, wavelet_coeff_array, scale_index_1, scale_index_2, alt_index_1, alt_index_2):
        ''' Takes an MxN array of wavelet coefficients and inverts it
            within the bounding box specified by scale and alt indices.
            Assumes rows are wavelet scale values, and columns the altitudes.
            Algorithm taken from Torrence and Compo (1998).
        '''
        scale_index_1 = int(scale_index_1); scale_index_2 = int(scale_index_2)
        alt_index_1 = int(alt_index_1); alt_index_2 = int(alt_index_2)
        j = np.arange(0, self.j1+1)

        a = self.dj*np.sqrt(self.dt)/(0.776) # magic number from Torrence and Compo
        b = a/(np.pi**(-1/4))

        scales = self.s0 * 2. ** (j * self.dj)
        scales_windowed = scales[scale_index_1:scale_index_2]
        wavelet_coeff_array_windowed = wavelet_coeff_array[scale_index_1:scale_index_2,alt_index_1:alt_index_2]
       
        x = []
        for col in range(wavelet_coeff_array_windowed.shape[1]):
            xn = np.sum(((wavelet_coeff_array_windowed[:, col]))/(np.sqrt(scales_windowed)))
            x.append(xn*b)

        return np.asarray(x)

    def plot_power_surface(self, title = False, levels = False):
        fig, ax = plt.subplots()
        ax.plot(self.alt, self.ucoi[:-1], 'k')
        ax.invert_yaxis()
        ax.set_ylim(ymax = self.s0)
        ax.semilogy(basey = 2)
        if title:
            ax.set_title(self.data_file)
        cf = ax.contourf(self.alt, self.fourier_period[1:], self.power_spectrum)
        fig.colorbar(cf)

 
    def windowed_data(self, arange):
        u_windowed = self._get_altitude_window(arange, self.u, self.alt)
        v_windowed = self._get_altitude_window(arange, self.v, self.alt)
        alt_windowed = self._get_altitude_window(arange, self.alt, self.alt)
        return u_windowed, v_windowed, alt_windowed

    def return_windowed_arrays(self, s1, s2, a1, a2):
        sidx1 = find_nearest_index(self.fourier_period, s1)
        sidx2 = find_nearest_index(self.fourier_period, s2)
        aidx1 = find_nearest_index(self.alt, a1)
        aidx2 = find_nearest_index(self.alt, a2)
        return self.alt[aidx1:aidx2], self.fourier_period[sidx1:sidx2], self.power_spectrum[sidx1:sidx2, aidx1:aidx2]

    def return_window_indices(self, s1, s2, a1, a2):
        sidx1 = find_nearest_index(self.fourier_period, s1)
        sidx2 = find_nearest_index(self.fourier_period, s2)
        aidx1 = find_nearest_index(self.alt, a1)
        aidx2 = find_nearest_index(self.alt, a2)
        return sidx1, sidx2, aidx1, aidx2

    @staticmethod
    def direction_and_frequency_from_stokes_params(u_wavelet_inverted, v_wavelet_inverted):
        '''Calculates Stokes' parameters from de-transformed wind data
        '''
        u = u_wavelet_inverted; v = v_wavelet_inverted
        
        I = np.mean(u.real**2) + np.mean(u.imag**2) + np.mean(v.real**2) + np.mean(v.imag**2) 
        
        D = np.mean(u.real**2) + np.mean(u.imag**2) - np.mean(v.real**2) - np.mean(v.imag**2) 
        
        P = 2*(np.mean(u.real*v.real) + np.mean(u.imag*v.imag))
        
        Q = 2*(np.mean(u.real*v.imag) - np.mean(u.imag*v.real))
        
        phi = 0.5*np.arctan2(P, D)

        d = np.sqrt(D**2 + P**2 + Q**2)
        
        arg = Q/(d)
        
        xi = np.arcsin(arg)/2
        
        omega_over_f = (np.tan(xi))

        return phi, omega_over_f




def find_nearest_index(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx




class Hodograph(Data_Methods):

    def __init__(self, data_file):
        super(Hodograph, self).__init__()
        self.alt, self.time, self.u, self.v, self.temp = self._get_data(data_file)

    def plot_hodograph(self, arange):
        ''' 
            Plots and shows hodograph in desired altitude range.
        '''
        u_windowed = self._get_altitude_window(arange, self.u, self.alt)
        v_windowed = self._get_altitude_window(arange, self.v, self.alt)
        alt_windowed = self._get_altitude_window(arange, self.alt, self.alt)

        fig, ax = plt.subplots()
        ax.annotate(alt_windowed[0],(u_windowed[0], v_windowed[0]))    
        ax.annotate(alt_windowed[-1],(u_windowed[-1],v_windowed[-1]))
        ax.plot(u_windowed, v_windowed, 'rx', ms = 1)
        plt.show()

    def window_data(self, arange):
        u_windowed = self._get_altitude_window(arange, self.u, self.alt)
        v_windowed = self._get_altitude_window(arange, self.v, self.alt)
        alt_windowed = self._get_altitude_window(arange, self.alt, self.alt)
        return u_windowed, v_windowed, alt_windowed


    def fit_and_plot_ellipses(self, arange):

        u_windowed = self._get_altitude_window(arange, self.u, self.alt)
        v_windowed = self._get_altitude_window(arange, self.v, self.alt)
        alt_windowed = self._get_altitude_window(arange, self.alt, self.alt)
        temp_windowed = self._get_altitude_window(arange, self.temp, self.alt)

        # Method 1

        z, self.a1, self.b1, self.phi1 = fitellipse_m1(np.mat([u_windowed,v_windowed]))
        theta = np.linspace(0, 2*np.pi, 100)
        x_ellipse_m1 = self.a1*np.cos(theta)*np.cos(self.phi1) - self.b1*np.sin(theta)*np.sin(self.phi1) + z[0]
        y_ellipse_m1 = self.a1*np.cos(theta)*np.sin(self.phi1) + self.b1*np.sin(theta)*np.cos(self.phi1) + z[1]

        fig, ax = plt.subplots()
        ax.set_title(r"Ellipse fit, method 1, \eta =", self.eta(u_windowed, temp_windowed, alt_windowed))
        ax.plot(u_windowed, v_windowed, 'rx', u_windowed, v_windowed, 'r.-', alpha = 0.5)
        ax.plot(x_ellipse_m1, y_ellipse_m1, 'k')

        # Method 2

        self.a2, self.b2, self.phi2, x_ellipse_m2, y_ellipse_m2 = fitellipse_m2(u_windowed,v_windowed)
        
        fig, ax = plt.subplots()
        ax.set_title(r"Ellipse fit, method 2, \eta =", self.eta(u_windowed, temp_windowed, alt_windowed))
        ax.plot(u_windowed, v_windowed, 'rx', u_windowed, v_windowed, 'r.-', alpha = 0.5)
        ax.plot(x_ellipse_m2, y_ellipse_m2, 'k')

        plt.show()


def ascol( arr ):
    '''
    If the dimensionality of 'arr' is 1, reshapes it to be a column matrix (N,1).
    '''
    if len( arr.shape ) == 1: arr = arr.reshape( ( arr.shape[0], 1 ) )
    return arr


def asrow( arr ):
    '''
    If the dimensionality of 'arr' is 1, reshapes it to be a row matrix (1,N).
    '''
    if len( arr.shape ) == 1: arr = arr.reshape( ( 1, arr.shape[0] ) )
    return arr


def fitggk(x):
    '''
    function [z, a, b, alpha] = fitggk(x)
    % Linear least squares with the Euclidean-invariant constraint Trace(A) = 1
    '''
    
    ## Convenience variables
    m  = x.shape[1]
    x1 = x[0, :].reshape((1,m)).T
    x2 = x[1, :].reshape((1,m)).T
    
    ## Coefficient matrix
    B = np.hstack([ np.multiply( 2 * x1, x2 ), np.power( x2, 2 ) - np.power( x1, 2 ), x1, x2, np.ones((m, 1)) ])
    
    v = np.linalg.lstsq( B, -np.power( x1, 2 ) )[0].ravel()
    
    ## For clarity, fill in the quadratic form variables
    A        = np.zeros((2,2))
    A[0,0]   = 1 - v[1]
    A.ravel()[1:3] = v[0]
    A[1,1]   = v[1]
    bv       = v[2:4]
    c        = v[4]
    
    ## find parameters
    z, a, b, alpha = conic2parametric(A, bv, c)
    
    return z, a, b, alpha


def fitnonlinear(x, z0, a0, b0, alpha0, **params):
    '''
    function [z, a, b, alpha, fConverged] = fitnonlinear(x, z0, a0, b0, alpha0, params)
    % Gauss-Newton least squares ellipse fit minimising geometric distance 
    '''
    
    ## Get initial rotation matrix
    Q0 = np.array( [[ np.cos(alpha0), -np.sin(alpha0) ], [ np.sin(alpha0), np.cos(alpha0) ]] )
    m = x.shape[1]
    
    ## Get initial phase estimates
    phi0 = np.angle( np.dot( np.dot( np.array([1, 1j]), Q0.T ), x - z0.reshape((2,1)) ) ).T
    u = np.hstack( [ phi0, alpha0, a0, b0, z0 ] ).T
    
    
    def sys(u):
        '''
        function [f, J] = sys(u)
        % SYS : Define the system of nonlinear equations and Jacobian. Nested
        % function accesses X (but changeth it not)
        % from the FITELLIPSE workspace
        '''
        
        ## Tolerance for whether it is a circle
        circTol = 1e-5
        
        ## Unpack parameters from u
        phi   = u[:-5]
        alpha = u[-5]
        a     = u[-4]
        b     = u[-3]
        z     = u[-2:]
        
        ## If it is a circle, the Jacobian will be singular, and the
        ## Gauss-Newton step won't work. 
        ##TODO: This can be fixed by switching to a Levenberg-Marquardt
        ##solver
        if abs(a - b) / (a + b) < circTol:
            print('fitellipse:CircleFound', 'Ellipse is near-circular - nonlinear fit may not succeed')
        
        ## Convenience trig variables
        c = np.cos(phi)
        s = np.sin(phi)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        ## Rotation matrices
        Q    = np.array( [[ca, -sa],[sa, ca]] )
        Qdot = np.array( [[-sa, -ca],[ca, -sa]] )
        
        ## Preallocate function and Jacobian variables
        f = np.zeros(2 * m)
        J = np.zeros((2 * m, m + 5))
        for i in range( m ):
            rows = range( (2*i), (2*i)+2 )
            ## Equation system - vector difference between point on ellipse
            ## and data point
            f[ rows ] = x[:, i] - z - np.dot( Q, np.array([ a * np.cos(phi[i]), b * np.sin(phi[i]) ]) )
            
            ## Jacobian
            J[ rows, i ] = np.dot( -Q, np.array([ -a * s[i], b * c[i] ]) )
            J[ rows, -5: ] = \
                np.hstack([ ascol( np.dot( -Qdot, np.array([ a * c[i], b * s[i] ]) ) ), ascol( np.dot( -Q, np.array([ c[i], 0 ]) ) ), ascol( np.dot( -Q, np.array([ 0, s[i] ]) ) ), np.array([[-1, 0],[0, -1]]) ])
        
        return f,J
    
    
    ## Iterate using Gauss Newton
    fConverged = False
    for nIts in range( params['maxits'] ):
        ## Find the function and Jacobian
        f, J = sys(u)
        
        ## Solve for the step and update u
        #h = linalg.solve( -J, f )
        h = np.linalg.lstsq( -J, f )[0]
        u = u + h
        
        ## Check for convergence
        delta = np.linalg.norm(h, np.inf) / np.linalg.norm(u, np.inf)
        if delta < params['tol']:
            fConverged = True
            break
    
    alpha = u[-5]
    a     = u[-4]
    b     = u[-3]
    z     = u[-2:]
    
    return z, a, b, alpha, fConverged


def conic2parametric(A, bv, c):
    '''
    function [z, a, b, alpha] = conic2parametric(A, bv, c)
    '''
    ## Diagonalise A - find Q, D such at A = Q' * D * Q
    D, Q = np.linalg.eig(A)
    Q = Q.T
    
    ## If the determinant < 0, it's not an ellipse
    if np.prod(D) <= 0:
        raise RuntimeError('fitellipse:NotEllipse Linear fit did not produce an ellipse')
    
    ## We have b_h' = 2 * t' * A + b'
    t = -0.5 * np.linalg.solve(A, bv)
    
    c_h = np.dot( np.dot( t.T, A ), t ) + np.dot( bv.T, t ) + c
    
    z = t
    a = np.sqrt(-c_h / D[0])
    b = np.sqrt(-c_h / D[1])
    alpha = np.arctan2(Q[0,1], Q[0,0])
    
    return z, a, b, alpha

def fitbookstein(x):
    '''
    function [z, a, b, alpha] = fitbookstein(x)
    %FITBOOKSTEIN   Linear ellipse fit using bookstein constraint
    %   lambda_1^2 + lambda_2^2 = 1, where lambda_i are the eigenvalues of A
    '''
    
    ## Convenience variables
    m  = x.shape[1]
    x1 = x[0, :].reshape((1,m)).T
    x2 = x[1, :].reshape((1,m)).T
    
    ## Define the coefficient matrix B, such that we solve the system
    ## B *[v; w] = 0, with the constraint norm(w) == 1
    B = np.hstack([ x1, x2, np.ones((m, 1)), np.power( x1, 2 ), np.multiply( np.sqrt(2) * x1, x2 ), np.power( x2, 2 ) ])
    
    ## To enforce the constraint, we need to take the QR decomposition
    Q, R = np.linalg.qr(B)
    
    ## Decompose R into blocks
    R11 = R[0:3, 0:3]
    R12 = R[0:3, 3:6]
    R22 = R[3:6, 3:6]
    
    ## Solve R22 * w = 0 subject to norm(w) == 1
    U, S, V = np.linalg.svd(R22)
    V = V.T
    w = V[:, 2]
    
    ## Solve for the remaining variables
    v = np.dot( np.linalg.solve( -R11, R12 ), w )
    
    ## Fill in the quadratic form
    A        = np.zeros((2,2))
    A.ravel()[0]     = w.ravel()[0]
    A.ravel()[1:3] = 1 / np.sqrt(2) * w.ravel()[1]
    A.ravel()[3]     = w.ravel()[2]
    bv       = v[0:2]
    c        = v[2]
    
    ## Find the parameters
    z, a, b, alpha = conic2parametric(A, bv, c)
    
    return z, a, b, alpha


def fitellipse_m1(x, opt = 'nonlinear', **kwargs ):
    '''
    function [z, a, b, alpha] = fitellipse(x, varargin) ::: returns alpha counter-clockwise from N.
    %FITELLIPSE   least squares fit of ellipse to 2D data
    %
    %   [Z, A, B, ALPHA] = FITELLIPSE(X)
    %       Fit an ellipse to the 2D points in the 2xN array X. The ellipse is
    %       returned in parametric form such that the equation of the ellipse
    %       parameterised by 0 <= theta < 2*pi is:
    %           X = Z + Q(ALPHA) * [A * cos(theta); B * sin(theta)]
    %       where Q(ALPHA) is the rotation matrix
    %           Q(ALPHA) = [cos(ALPHA), -sin(ALPHA); 
    %                       sin(ALPHA), cos(ALPHA)]
    %
    %       Fitting is performed by nonlinear least squares, optimising the
    %       squared sum of orthogonal distances from the points to the fitted
    %       ellipse. The initial guess is calculated by a linear least squares
    %       routine, by default using the Bookstein constraint (see below)
    %
    %   [...]            = FITELLIPSE(X, 'linear')
    %       Fit an ellipse using linear least squares. The conic to be fitted
    %       is of the form
    %           x'Ax + b'x + c = 0
    %       and the algebraic error is minimised by least squares with the
    %       Bookstein constraint (lambda_1^2 + lambda_2^2 = 1, where 
    %       lambda_i are the eigenvalues of A)
    %
    %   [...]            = FITELLIPSE(..., 'Property', 'value', ...)
    %       Specify property/value pairs to change problem parameters
    %          Property                  Values
    %          =================================
    %          'constraint'              {|'bookstein'|, 'trace'}
    %                                    For the linear fit, the following
    %                                    quadratic form is considered
    %                                    x'Ax + b'x + c = 0. Different
    %                                    constraints on the parameters yield
    %                                    different fits. Both 'bookstein' and
    %                                    'trace' are Euclidean-invariant
    %                                    constraints on the eigenvalues of A,
    %                                    meaning the fit will be invariant
    %                                    under Euclidean transformations
    %                                    'bookstein': lambda1^2 + lambda2^2 = 1
    %                                    'trace'    : lambda1 + lambda2     = 1
    %
    %           Nonlinear Fit Property   Values
    %           ===============================
    %           'maxits'                 positive integer, default 200
    %                                    Maximum number of iterations for the
    %                                    Gauss Newton step
    %
    %           'tol'                    positive real, default 1e-5
    %                                    Relative step size tolerance
    %   Example:
    %       % A set of points
    %       x = [1 2 5 7 9 6 3 8; 
    %            7 6 8 7 5 7 2 4];
    % 
    %       % Fit an ellipse using the Bookstein constraint
    %       [zb, ab, bb, alphab] = fitellipse(x, 'linear');
    %
    %       % Find the least squares geometric estimate       
    %       [zg, ag, bg, alphag] = fitellipse(x);
    %       
    %       % Plot the results
    %       plot(x(1,:), x(2,:), 'ro')
    %       hold on
    %       % plotellipse(zb, ab, bb, alphab, 'b--')
    %       % plotellipse(zg, ag, bg, alphag, 'k')
    % 
    %   See also PLOTELLIPSE
    
    % Copyright Richard Brown, this code can be freely used and modified so
    % long as this line is retained
    '''
    #error(nargchk(1, 5, nargin, 'struct'))
    
    x = np.asarray( x )
    
    ## Parse inputs
    # ...
    ## Default parameters
    kwargs[ 'fNonlinear' ] = opt is not 'linear'
    kwargs.setdefault( 'constraint', 'bookstein' )
    kwargs.setdefault( 'maxits', 200 )
    kwargs.setdefault( 'tol', 1e-5 )
    if x.shape[1] == 2:
        x = x.T
    if x.shape[1] < 6:
        raise RuntimeError('fitellipse:InsufficientPoints At least 6 points required to compute fit')
    
    ## Constraints are Euclidean-invariant, so improve conditioning by removing
    ## centroid
    centroid = np.mean(x, 1)
    x        = x - centroid.reshape((2,1))
    
    ## Obtain a linear estimate
    if kwargs['constraint'] == 'bookstein':
        ## Bookstein constraint : lambda_1^2 + lambda_2^2 = 1
        z, a, b, alpha = fitbookstein(x)
    
    elif kwargs['constraint'] == 'trace':
        ## 'trace' constraint, lambda1 + lambda2 = trace(A) = 1
        z, a, b, alpha = fitggk(x)
    
    ## Minimise geometric error using nonlinear least squares if required
    if kwargs['fNonlinear']:
        ## Initial conditions
        z0     = z
        a0     = a
        b0     = b
        alpha0 = alpha
        
        ## Apply the fit
        z, a, b, alpha, fConverged = fitnonlinear(x, z0, a0, b0, alpha0, **kwargs)
        
        ## Return linear estimate if GN doesn't converge
        if not fConverged:
            print('fitellipse:FailureToConverge', 'Gauss-Newton did not converge, returning linear estimate')
            z = z0
            a = a0
            b = b0
            alpha = alpha0
    
    ## Add the centroid back on
    z = z + centroid
    
    return z, a, b, alpha


def fitellipse_m2(x,y):

    a = Ellipse(x,y)
    center = ellipseCenter(a)
    axes = ellipseAxisLength(a)
    phi = ellipseAngleOfRotation(a) 
    R = np.linspace(0,2*np.pi,50)
    a, b = axes #semi-major, semi-minor axes
    xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
    yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
    return a,b,phi,xx,yy #semi-major, semi-minor, angle of rotation, x array, y array


def Ellipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a


def ellipseCenter(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def ellipseAxisLength(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])


def ellipseAngleOfRotation(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2









