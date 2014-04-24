__author__ = ('Thomas Rueckstiess, ruecksti@in.tum.de'
              'Justin Bayer, bayer.justin@googlemail.com'
              'Xin Chen, xin.chen@nexd.co (Conjugate Gradient Descent)')


from scipy import zeros, asarray, sign, array, cov, dot, clip, ndarray
from scipy.linalg import inv
from scipy import sqrt, isnan, isinf, isreal
from scipy import finfo


class GradientDescent(object):

    def __init__(self):
        """ initialize algorithms with standard parameters (typical values given in parentheses)"""

        # --- BackProp parameters ---
        # learning rate (0.1-0.001, down to 1e-7 for RNNs)
        self.alpha = 0.1

        # alpha decay (0.999; 1.0 = disabled)
        self.alphadecay = 1.0

        # momentum parameters (0.1 or 0.9)
        self.momentum = 0.0
        self.momentumvector = None

        # --- RProp parameters ---
        self.rprop = False
        # maximum step width (1 - 20)
        self.deltamax = 5.0
        # minimum step width (0.01 - 1e-6)
        self.deltamin = 0.01
        # the remaining parameters do not normally need to be changed
        self.deltanull = 0.1
        self.etaplus = 1.2
        self.etaminus = 0.5
        self.lastgradient = None

    def init(self, values):
        """ call this to initialize data structures *after* algorithm to use
        has been selected

        :arg values: the list (or array) of parameters to perform gradient descent on
                       (will be copied, original not modified)
        """
        assert isinstance(values, ndarray)
        self.values = values.copy()
        if self.rprop:
            self.lastgradient = zeros(len(values), dtype='float64')
            self.rprop_theta = self.lastgradient + self.deltanull
            self.momentumvector = None
        else:
            self.lastgradient = None
            self.momentumvector = zeros(len(values))

    def __call__(self, gradient, error=None):
        """ calculates parameter change based on given gradient and returns updated parameters """
        # check if gradient has correct dimensionality, then make array """
        assert len(gradient) == len(self.values)
        gradient_arr = asarray(gradient)

        if self.rprop:
            rprop_theta = self.rprop_theta

            # update parameters
            self.values += sign(gradient_arr) * rprop_theta

            # update rprop meta parameters
            dirSwitch = self.lastgradient * gradient_arr
            rprop_theta[dirSwitch > 0] *= self.etaplus
            idx =  dirSwitch < 0
            rprop_theta[idx] *= self.etaminus
            gradient_arr[idx] = 0

            # upper and lower bound for both matrices
            rprop_theta = rprop_theta.clip(min=self.deltamin, max=self.deltamax)

            # save current gradients to compare with in next time step
            self.lastgradient = gradient_arr.copy()

            self.rprop_theta = rprop_theta

        else:
            # update momentum vector (momentum = 0 clears it)
            self.momentumvector *= self.momentum

            # update parameters (including momentum)
            self.momentumvector += self.alpha * gradient_arr
            self.alpha *= self.alphadecay

            # update parameters
            self.values += self.momentumvector

        return self.values

    descent = __call__


class NaturalGradient(object):

    def __init__(self, samplesize):
        # Counter after how many samples a new gradient estimate will be
        # returned.
        self.samplesize = samplesize
        # Samples of the gradient are held in this datastructure.
        self.samples = []

    def init(self, values):
        self.values = values.copy()

    def __call__(self, gradient, error=None):
        # Append a copy to make sure this one is not changed after by the
        # client.
        self.samples.append(array(gradient))
        # Return None if no new estimate is being given.
        if len(self.samples) < self.samplesize:
            return None
        # After all the samples have been put into a single array, we can
        # delete them.
        gradientarray = array(self.samples).T
        inv_covar = inv(cov(gradientarray))
        self.values += dot(inv_covar, gradientarray.sum(axis=1))
        return self.values


class IRpropPlus(object):

    def __init__(self, upfactor=1.1, downfactor=0.9, bound=0.5):
        self.upfactor = upfactor
        self.downfactor = downfactor
        if not bound > 0:
            raise ValueError("bound greater than 0 needed.")

    def init(self, values):
        self.values = values.copy()
        self.prev_values = values.copy()
        self.more_prev_values = values.copy()
        self.previous_gradient = zeros(values.shape)
        self.step = zeros(values.shape)
        self.previous_error = float("-inf")

    def __call__(self, gradient, error):
        products = self.previous_gradient * gradient
        signs = sign(gradient)

        # For positive gradient parts.
        positive = (products > 0).astype('int8')
        pos_step = self.step * self.upfactor * positive
        clip(pos_step, -self.bound, self.bound)
        pos_update = self.values - signs * pos_step

        # For negative gradient parts.
        negative = (products < 0).astype('int8')
        neg_step = self.step * self.downfactor * negative
        clip(neg_step, -self.bound, self.bound)
        if error <= self.previous_error:
            # If the error has decreased, do nothing.
            neg_update = zeros(gradient.shape)
        else:
            # If it has increased, move back 2 steps.
            neg_update = self.more_prev_values
        # Set all negative gradients to zero for the next step.
        gradient *= positive

        # Bookkeeping.
        self.previous_gradient = gradient
        self.more_prev_values = self.prev_values
        self.prev_values = self.values.copy()
        self.previous_error = error

        # Updates.
        self.step[:] = pos_step + neg_step
        self.values[:] = positive * pos_update + negative * neg_update

        return self.values



class PRConjugateGradientDescent(object):

    """ A conjugate gradient descent algorithm class. The Polack-Rebiere flavour
    of conjugate gradients is used to calculate search directions, and a line
    search using quadratic and cubic polynomial approximations and the 
    Wolfe-Powell stopping criteria is used topgether with the slope ratio metho
    for guessing initial step sizes. Additionally a bunch of checks are made to 
    make sure that exploration is taking place and that extrapolation will not
    be unboundedly large. 
    Different from the GradientDescent class above, the usage of this class is
    more C/MATLAB style rather than pybrain-style. The cost function handler
    will be passed to this class and all the optimization step are within this
    class. """
    
    def __init__(self, RHO=0.0001, SIG=0.5, reEvaluate=0.1, extrapolate=3.0, 
                 maxEvaluate=20, maxSlope=100.0):
        """ Setup a bunch of constants for the algorithm. RHO and SIG are the
        constants in the Wolfe-Powell conditions. Do not re-evaluate within
        [limit-reEvaluate, limit+reEvaluate] of the current bracket. Extrapolate 
        maximum 3 times the current bracket. Max 20 function evaluations per 
        line search. maxSlope determines the maximum allowed slope ratio. """
        self.RHO = RHO
        self.SIG = SIG
        self.reEvaluate = reEvaluate
        self.extrapolate = extrapolate
        self.maxSlope = maxSlope
        self.maxEvaluate = maxEvaluate
    
    
    def __str__(self):
        """ Return the string representation of this class """
        s = "Polack-Ribiere Conjugate Gradient Descent {\n"
        s += "\tRHO: {:f}\n".format(self.RHO)
        s += "\tSIG: {:f}\n".format(self.SIG)
        s += "\tINT: {:f}\n".format(self.reEvaluate)
        s += "\tEXT: {:f}\n".format(self.extrapolate)
        s += "\tMAX: {:f}\n".format(self.maxEvaluate)
        s += "\tSLP: {:f}\n".format(self.maxSlope)
        return s
    
    
    @staticmethod
    def quadraticFit(z3, d3, f2, f3):
        """ The quadratic polynomial approximation """
        return z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3)


    @staticmethod
    def cubicFit(d2, d3, f2, f3, z3):
        """ The cubic polynomial approximation """
        A = 6.0 * (f2 - f3) / z3 + 3.0 * (d2 + d3)
        B = 3.0 * (f3 - f2) - z3 * (d3 + 2.0 * d2)
        z2 = (sqrt(B * B - A * d2 * z3 * z3) - B) / A
        return A, B, z2

    
    @staticmethod
    def cubicExtrapolate(d2, d3, f2, f3, z3):
        """ The cubic extrapolation. line 122-124 in fmincg.m """
        A = 6.0 * (f2 - f3) / z3 + 3.0 * (d2 + d3)
        B = 3.0 * (f3 - f2) - z3 * (d3 + 2.0 * d2)
        z2 = -d2 * z3 * z3 / (B + sqrt(B * B - A * d2 * z3 * z3))
        return A, B, z2
    
    
    def __call__(self, X, evalfunc, maxEpochs=100, verbose=True, convergenceThreshold=None):
        """ Run the conjugate gradient descent. evalfunc is a function handler
        which has only one input vairble and X is the input module. X should
        have a property named params and implement a copy method. Generally,
        X.params should return a numpy/scipy array. 
        
        An minimal example is given below showing how to use this class:
        from scipy import array, zeros
        class T(object):
            def __init__(self):
                self._params = array([0.0, 0.0])
            @property
            def params(self):
                return self._params
            @params.setter
            def params(self, value):
                self._params = value
            def copy(self):
                c = T()
                c.params = self.params.copy()
                return c
        class F(object):
            def __init__(self):
                self.A = array([[1.0, 1.0], 
                                [1.0, 2.0], 
                                [1.0, 3.0], 
                                [1.0, 4.0], 
                                [1.0, 5.0]])
                self.b = array([2.0, 3.0, 4.0, 5.0, 6.0])
            def cost(self, module):
                errors = 0.0
                G = array([0.0, 0.0])
                for i in range(self.A.shape[0]):
                    dx = dot(self.A[i], module.params) - self.b[i]
                    errors += 0.5 * dx * dx
                    G += dx * self.A[i]
                return errors, G
        sample_X = T()
        sample_CostFunction = F()
        PRConjugateG = PRConjugateGradientDescent(verbose=True)
        PRConjugateG(sample_X, sample_CostFunction.cost)
        print sample_X.params
        """
        
        # The module object must have a property whose name is params and also
        # implemented the copy function.
        assert hasattr(X, "params"), "X should have a property named params."
        assert hasattr(X, "copy"), "X should have a copy method."
        
        # Print the optimizer parameters
        print(self)
        
        epoch = 0
        lineSearchFailed = False
        strfmt = "Epoch {" + ":{:d}".format(len(str(maxEpochs))) + "d} | error: {:.12f}"
        realmin = finfo(float).tiny
        reason = ""
        
        # Get function value and gradient
        f1, g1 = evalfunc(X)
        fX = []
        
        # Search direction is the steepest
        s = -g1
        
        # This is the slope
        d1 = -dot(s, s)
        
        # Initial step is 1.0 / (1.0 - d1)
        z1 = 1.0 / (1.0 - d1)
        
        # Setup the convergence threhold
        minDelta = 1e-8
        minRatio = 1e-6
        if convergenceThreshold is not None:
            deltaThreshold = convergenceThreshold[0]
            ratioThreshold = convergenceThreshold[1]
        
        while epoch < maxEpochs:
            epoch += 1
            
            # make a copy of current values
            X0 = X.copy()
            f0 = f1
            g0 = g1.copy()
            
            # begin line search
            X.params[:] += z1 * s            
            f2, g2 = evalfunc(X)            
            d2 = dot(g2, s)
            
            # initialize point 3 equal to point 1
            f3 = f1
            d3 = d1
            z3 = -z1
            M = self.maxEvaluate
            success = False
            limit = -1.0
            
            # declare some global variables
            A = 0.0
            B = 0.0
            z2 = 0.0
            while True:    
                while (f2 > f1 + z1 * self.RHO * d1 or d2 > -self.SIG * d1) and M > 0:
                    # tighten the bracket
                    limit = z1
                    if f2 > f1:
                        z2 = self.quadraticFit(z3, d3, f2, f3)                        
                    else:
                        A, B, z2 = self.cubicFit(d2, d3, f2, f3, z3)                    

                    # if we have a numerical problem then bisect
                    if isnan(z2) or isinf(z2):
                        z2 = z3 * 0.5
                    # do not accept too close to limits
                    z2 = min(z2, self.reEvaluate * z3)
                    z2 = max(z2, (1.0 - self.reEvaluate) * z3)
                    # update the step
                    z1 += z2;
                    X.params[:] += z2 * s;
                    f2, g2 = evalfunc(X)
                    M -= 1; 
                    d2 = dot(g2, s)
                    # z3 is now relative to the location of z2
                    z3 -= z2
                pass
                
                # this is a failure
                if f2 > f1 + z1 * self.RHO * d1 or d2 > -self.SIG * d1:
                    break
                elif d2 > self.SIG * d1:
                    success = True
                    break
                # this is also a failure
                elif M == 0:
                    break
                
                # make cubic extrapolation
                A, B, z2 = self.cubicExtrapolate(d2, d3, f2, f3, z3)
                
                # numeric problem or wrong sign
                if not isreal(z2) or isnan(z2) or isinf(z2) or z2 < 0.0:
                    if limit < -0.5:
                        z2 = z1 * (self.extrapolate - 1.0)
                    else:
                        z2 = (limit - z1) * 0.5
                # extrapolation beyond max
                elif limit > -0.5 and (z2 + z1) > limit:
                    z2 = (limit - z1) * 0.5
                # extrapolation beyond limit
                elif limit < -0.5 and (z2 + z1) > z1 * self.extrapolate:
                    z2 = z1 * (self.extrapolate - 1.0)
                elif z2 < -z3 * self.reEvaluate:
                    z2 = -z3 * self.reEvaluate
                # too close to limit
                elif limit > -0.5 and z2 < (limit - z1) * (1.0 - self.reEvaluate):
                    z2 = (limit - z1) * (1.0 - self.reEvaluate)
                # set point 3 equal to point 2
                f3 = f2
                d3 = d2
                z3 = -z2
                # update current estimate
                z1 += z2
                X.params[:] += z2 * s
                f2, g2 = evalfunc(X)
                M -= 1
                d2 = dot(g2, s)
            pass
            if success == True:
                fX.append(f2)
                if verbose:
                    print(strfmt.format(epoch, f2))
                # The error decrease rate is below the convergence criteria, so
                # the optimization is done successfully.
                if (f1 - f2) < deltaThreshold or (f1 - f2) / f1 < ratioThreshold:
                    break
                f1 = f2
                # Polack-Ribiere direction
                s = (dot(g2, g2) - dot(g1, g2)) / dot(g1, g1) * s - g2
                # swap derivatives
                tmp = g1
                g1 = g2
                g2 = tmp
                d2 = dot(g1, s)
                # new slope must be positive, otherwise use steepest direction
                if d2 > 0.0:
                    s = -g1
                    d2 = -dot(s, s)
                z1 *= min(self.maxSlope, d1 / (d2 - realmin))
                d1 = d2
                lineSearchFailed = False
            else:
                # restore point from the point before line search
                X = X0.copy()
                f1 = f0
                g1 = g0
                # line search failed twice in a row or we ran out of time
                if lineSearchFailed:
                    reason = "Line search failed twice in a row!"
                    break
                if epoch > maxEpochs:
                    reason = "Maximum number of epochs reached!"
                    break
                # swap derivatives
                tmp = g1
                g1 = g2
                g2 = tmp
                # try steepest
                s = -g1
                d1 = -dot(s, s)
                z1 = 1.0 / (1.0 - d1)
                lineSearchFailed = True
        if success:    
            print("PRCG converged with {:d} epochs. Final error: {:f}".format(epoch, f1))
        else:
            print("PRCG failed to converge: {:s}".format(reason))
        
        return fX
