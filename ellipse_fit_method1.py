from math import *
from numpy import *



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
    B = hstack([ multiply( 2 * x1, x2 ), power( x2, 2 ) - power( x1, 2 ), x1, x2, ones((m, 1)) ])
    
    v = linalg.lstsq( B, -power( x1, 2 ) )[0].ravel()
    
    ## For clarity, fill in the quadratic form variables
    A        = zeros((2,2))
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
    Q0 = array( [[ cos(alpha0), -sin(alpha0) ], [ sin(alpha0), cos(alpha0) ]] )
    m = x.shape[1]
    
    ## Get initial phase estimates
    phi0 = angle( dot( dot( array([1, 1j]), Q0.T ), x - z0.reshape((2,1)) ) ).T
    u = hstack( [ phi0, alpha0, a0, b0, z0 ] ).T
    
    
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
        c = cos(phi)
        s = sin(phi)
        ca = cos(alpha)
        sa = sin(alpha)
        
        ## Rotation matrices
        Q    = array( [[ca, -sa],[sa, ca]] )
        Qdot = array( [[-sa, -ca],[ca, -sa]] )
        
        ## Preallocate function and Jacobian variables
        f = zeros(2 * m)
        J = zeros((2 * m, m + 5))
        for i in range( m ):
            rows = range( (2*i), (2*i)+2 )
            ## Equation system - vector difference between point on ellipse
            ## and data point
            f[ rows ] = x[:, i] - z - dot( Q, array([ a * cos(phi[i]), b * sin(phi[i]) ]) )
            
            ## Jacobian
            J[ rows, i ] = dot( -Q, array([ -a * s[i], b * c[i] ]) )
            J[ rows, -5: ] = \
                hstack([ ascol( dot( -Qdot, array([ a * c[i], b * s[i] ]) ) ), ascol( dot( -Q, array([ c[i], 0 ]) ) ), ascol( dot( -Q, array([ 0, s[i] ]) ) ), array([[-1, 0],[0, -1]]) ])
        
        return f,J
    
    
    ## Iterate using Gauss Newton
    fConverged = False
    for nIts in range( params['maxits'] ):
        ## Find the function and Jacobian
        f, J = sys(u)
        
        ## Solve for the step and update u
        #h = linalg.solve( -J, f )
        h = linalg.lstsq( -J, f )[0]
        u = u + h
        
        ## Check for convergence
        delta = linalg.norm(h, inf) / linalg.norm(u, inf)
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
    D, Q = linalg.eig(A)
    Q = Q.T
    
    ## If the determinant < 0, it's not an ellipse
    if prod(D) <= 0:
        raise RuntimeError('fitellipse:NotEllipse Linear fit did not produce an ellipse')
    
    ## We have b_h' = 2 * t' * A + b'
    t = -0.5 * linalg.solve(A, bv)
    
    c_h = dot( dot( t.T, A ), t ) + dot( bv.T, t ) + c
    
    z = t
    a = sqrt(-c_h / D[0])
    b = sqrt(-c_h / D[1])
    alpha = atan2(Q[0,1], Q[0,0])
    
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
    B = hstack([ x1, x2, ones((m, 1)), power( x1, 2 ), multiply( sqrt(2) * x1, x2 ), power( x2, 2 ) ])
    
    ## To enforce the constraint, we need to take the QR decomposition
    Q, R = linalg.qr(B)
    
    ## Decompose R into blocks
    R11 = R[0:3, 0:3]
    R12 = R[0:3, 3:6]
    R22 = R[3:6, 3:6]
    
    ## Solve R22 * w = 0 subject to norm(w) == 1
    U, S, V = linalg.svd(R22)
    V = V.T
    w = V[:, 2]
    
    ## Solve for the remaining variables
    v = dot( linalg.solve( -R11, R12 ), w )
    
    ## Fill in the quadratic form
    A        = zeros((2,2))
    A.ravel()[0]     = w.ravel()[0]
    A.ravel()[1:3] = 1 / sqrt(2) * w.ravel()[1]
    A.ravel()[3]     = w.ravel()[2]
    bv       = v[0:2]
    c        = v[2]
    
    ## Find the parameters
    z, a, b, alpha = conic2parametric(A, bv, c)
    
    return z, a, b, alpha

def fitellipse( x, opt = 'nonlinear', **kwargs ):
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
	
	x = asarray( x )
	
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
	centroid = mean(x, 1)
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






