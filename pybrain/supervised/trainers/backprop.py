__author__ = ('Daan Wierstra and Tom Schaul', 'Xin Chen, xin.chen@nexd.co')

from scipy import dot, argmax, absolute, isnan, isinf, isreal, sqrt, finfo
from random import shuffle
from pybrain.supervised.trainers.trainer import Trainer
from pybrain.utilities import fListToString
from pybrain.auxiliary import GradientDescent
from pybrain.auxiliary import PRConjugateGradientDescent


class BackpropTrainer(Trainer):
    """Trainer that trains the parameters of a module according to a
    supervised dataset (potentially sequential) by backpropagating the errors
    (through time)."""

    def __init__(self, module, dataset=None, learningrate=0.01, lrdecay=1.0,
                 momentum=0., verbose=False, batchlearning=False,
                 weightdecay=0.):
        """Create a BackpropTrainer to train the specified `module` on the
        specified `dataset`.

        The learning rate gives the ratio of which parameters are changed into
        the direction of the gradient. The learning rate decreases by `lrdecay`,
        which is used to to multiply the learning rate after each training
        step. The parameters are also adjusted with respect to `momentum`, which
        is the ratio by which the gradient of the last timestep is used.

        If `batchlearning` is set, the parameters are updated only at the end of
        each epoch. Default is False.

        `weightdecay` corresponds to the weightdecay rate, where 0 is no weight
        decay at all.
        """
        Trainer.__init__(self, module)
        self.setData(dataset)
        self.verbose = verbose
        self.batchlearning = batchlearning
        self.weightdecay = weightdecay
        self.epoch = 0
        self.totalepochs = 0
        # set up gradient descender
        self.descent = GradientDescent()
        self.descent.alpha = learningrate
        self.descent.momentum = momentum
        self.descent.alphadecay = lrdecay
        self.descent.init(module.params)


    def train(self):
        """Train the associated module for one epoch."""
        assert len(self.ds) > 0, "Dataset cannot be empty."
        self.module.resetDerivatives()
        errors = 0
        ponderation = 0.
        shuffledSequences = []
        for seq in self.ds._provideSequences():
            shuffledSequences.append(seq)
        shuffle(shuffledSequences)
        for seq in shuffledSequences:
            e, p = self._calcDerivs(seq)
            errors += e
            ponderation += p
            if not self.batchlearning:
                gradient = self.module.derivs - self.weightdecay * self.module.params
                new = self.descent(gradient, errors)
                if new is not None:
                    self.module.params[:] = new
                self.module.resetDerivatives()

        if self.verbose:
            print("Epoch:", self.epoch, "| Total error:", errors / ponderation)
        if self.batchlearning:
            self.module._setParameters(self.descent(self.module.derivs))
        self.epoch += 1
        self.totalepochs += 1
        return errors / ponderation


    def _calcDerivs(self, seq):
        """Calculate error function and backpropagate output errors to yield
        the gradient."""
        self.module.reset()
        for sample in seq:
            self.module.activate(sample[0])
        error = 0
        ponderation = 0.
        for offset, sample in reversed(list(enumerate(seq))):
            # need to make a distinction here between datasets containing
            # importance, and others
            target = sample[1]
            outerr = target - self.module.outputbuffer[offset]
            if len(sample) > 2:
                importance = sample[2]
                error += 0.5 * dot(importance, outerr ** 2)
                ponderation += sum(importance)
                self.module.backActivate(outerr * importance)
            else:
                error += 0.5 * sum(outerr ** 2)
                ponderation += len(target)
                # FIXME: the next line keeps arac from producing NaNs. I don't
                # know why that is, but somehow the __str__ method of the
                # ndarray class fixes something,
                str(outerr)
                self.module.backActivate(outerr)

        return error, ponderation


    def _checkGradient(self, dataset=None, silent=False):
        """Numeric check of the computed gradient for debugging purposes."""
        if dataset:
            self.setData(dataset)
        res = []
        for seq in self.ds._provideSequences():
            self.module.resetDerivatives()
            self._calcDerivs(seq)
            e = 1e-6
            analyticalDerivs = self.module.derivs.copy()
            numericalDerivs = []
            for p in range(self.module.paramdim):
                storedoldval = self.module.params[p]
                self.module.params[p] += e
                righterror, dummy = self._calcDerivs(seq)
                self.module.params[p] -= 2 * e
                lefterror, dummy = self._calcDerivs(seq)
                approxderiv = (righterror - lefterror) / (2 * e)
                self.module.params[p] = storedoldval
                numericalDerivs.append(approxderiv)
            r = zip(analyticalDerivs, numericalDerivs)
            res.append(r)
            if not silent:
                print(r)
        return res


    def testOnData(self, dataset=None, verbose=False):
        """Compute the MSE of the module performance on the given dataset.

        If no dataset is supplied, the one passed upon Trainer initialization is
        used."""
        if dataset == None:
            dataset = self.ds
        dataset.reset()
        if verbose:
            print('\nTesting on data:')
        errors = []
        importances = []
        ponderatedErrors = []
        for seq in dataset._provideSequences():
            self.module.reset()
            e, i = dataset._evaluateSequence(self.module.activate, seq, verbose)
            importances.append(i)
            errors.append(e)
            ponderatedErrors.append(e / i)
        if verbose:
            print('All errors:', ponderatedErrors)
        assert sum(importances) > 0
        avgErr = sum(errors) / sum(importances)
        if verbose:
            print('Average error:', avgErr)
            print(('Max error:', max(ponderatedErrors), 'Median error:',
                   sorted(ponderatedErrors)[len(errors) / 2]))
        return avgErr


    def testOnClassData(self, dataset=None, verbose=False,
                        return_targets=False):
        """Return winner-takes-all classification output on a given dataset.

        If no dataset is given, the dataset passed during Trainer
        initialization is used. If return_targets is set, also return
        corresponding target classes.
        """
        if dataset == None:
            dataset = self.ds
        dataset.reset()
        out = []
        targ = []
        for seq in dataset._provideSequences():
            self.module.reset()
            for input, target in seq:
                res = self.module.activate(input)
                out.append(argmax(res))
                targ.append(argmax(target))
        if return_targets:
            return out, targ
        else:
            return out

    def trainUntilConvergence(self, dataset=None, maxEpochs=None, verbose=None,
                              continueEpochs=10, validationProportion=0.25,
                              trainingData=None, validationData=None,
                              convergence_threshold=10):
        """Train the module on the dataset until it converges.

        Return the module with the parameters that gave the minimal validation
        error.

        If no dataset is given, the dataset passed during Trainer
        initialization is used. validationProportion is the ratio of the dataset
        that is used for the validation dataset.
        
        If the training and validation data is already set, the splitPropotion is ignored

        If maxEpochs is given, at most that many epochs
        are trained. Each time validation error hits a minimum, try for
        continueEpochs epochs to find a better one."""
        epochs = 0
        if dataset is None:
            dataset = self.ds
        if verbose is None:
            verbose = self.verbose
        if trainingData is None or validationData is None:
            # Split the dataset randomly: validationProportion of the samples
            # for validation.
            trainingData, validationData = (
                dataset.splitWithProportion(1 - validationProportion))
        if not (len(trainingData) > 0 and len(validationData)):
            raise ValueError("Provided dataset too small to be split into training " +
                             "and validation sets with proportion " + str(validationProportion))
        self.ds = trainingData
        bestweights = self.module.params.copy()
        bestverr = self.testOnData(validationData)
        bestepoch = 0
        self.trainingErrors = []
        self.validationErrors = [bestverr]
        while True:
            trainingError = self.train()
            validationError = self.testOnData(validationData)
            if isnan(trainingError) or isnan(validationError):
                raise Exception("Training produced NaN results")
            self.trainingErrors.append(trainingError)
            self.validationErrors.append(validationError)
            if epochs == 0 or self.validationErrors[-1] < bestverr:
                # one update is always done
                bestverr = self.validationErrors[-1]
                bestweights = self.module.params.copy()
                bestepoch = epochs

            if maxEpochs != None and epochs >= maxEpochs:
                self.module.params[:] = bestweights
                break
            epochs += 1

            if len(self.validationErrors) >= continueEpochs * 2:
                # have the validation errors started going up again?
                # compare the average of the last few to the previous few
                old = self.validationErrors[-continueEpochs * 2:-continueEpochs]
                new = self.validationErrors[-continueEpochs:]
                if min(new) > max(old):
                    self.module.params[:] = bestweights
                    break
                elif reduce(lambda x, y: x + (y - round(new[-1], convergence_threshold)), [round(y, convergence_threshold) for y in new]) == 0:
                    self.module.params[:] = bestweights
                    break
        #self.trainingErrors.append(self.testOnData(trainingData))
        self.ds = dataset
        if verbose:
            print('train-errors:', fListToString(self.trainingErrors, 6))
            print('valid-errors:', fListToString(self.validationErrors, 6))
        return self.trainingErrors[:bestepoch], self.validationErrors[:1 + bestepoch]

        

class ScaledConjugateGradientDescent(object):
    
    def __init__(self):
        pass
    
    
    def __call__(self, X, evalfunc, maxEpoch=100, verbose=False, convergenceThreshold=None):
        """ Run the scaled conjugate gradient descent """
        eps = finfo(float).eps
        sigma0 = 1.0e-4
        epoch = 0
        
        # Initial function value and gradient
        fold, gradnew = evalfunc(X)
        fnew = fold
        gradold = gradnew.copy()
            
        # Initial search direction
        d = -gradnew.copy()
        
        # Force calculation of ddirectional derivs
        success = True
        
        # nsuccess counts number of successes
        nsuccess = 0
        
        # Initial scale parameter
        beta = 1.0
        theta = 0.0
        
        # Lower and upper bound on scale
        betamin = 1.0e-15
        betamax = 1.0e100
        
        # Set the convergence threshold
        deltaThreshold = 1.0e-8
        ratioThreshold = 1.0e-6
        if convergenceThreshold is not None:
            deltaThreshold = convergenceThreshold[0]
            ratioThreshold = convergenceThreshold[1]
        pass
        
        # Main optimization loop
        epoch = 1
        while epoch < maxEpoch:
            
            # Calculate first and second directional derivatives
            if success:
                mu = dot(d, gradnew)
                if mu >= 0:
                    d = -gradnew.copy()
                    mu = dot(d, gradnew)
                pass
                kappa = dot(d, d)
                if kappa < eps:
                    print("kappa: ", kappa, " eps: ", eps)
                    print("Terminated due to kappa < eps.")
                    return
                pass
                sigma = sigma0 / sqrt(kappa)
                Xplus = X.copy()
                Xplus.params[:] = X.params + sigma * d
                _, gplus = evalfunc(Xplus)
                theta = dot(d, gplus - gradnew) / sigma
            pass
            
            # Increase effective curvature and evaluate step size alpha
            delta = theta + beta * kappa
            if delta <= 0:
                delta = beta * kappa
                beta = beta - theta / kappa
            pass
            alpha = -mu/delta
            
            # Calculate the comparison ratio
            Xnew = X.copy()
            Xnew.params[:] = X.params + alpha * d
            fnew, _ = evalfunc(Xnew)
            
            Delta = 2.0 * (fnew - fold) / (alpha * mu)
            if Delta >= 0.0:
                success = True
                nsuccess += 1
                X = Xnew.copy()
                fnow = fnew
            else:
                success = False
                fnow = fold
            pass
            
            if verbose:
                print("Epoch {:d} Error {:f} Scale {:f}".format(epoch, fnow, beta))
            
            if success:
                # Test for termination
                if absolute(alpha * d).max() < ratioThreshold:
                    if absolute(fnew - fold) < deltaThreshold:
                        return
                    pass
                else:
                    # Update variables for new position
                    fold = fnew
                    gradold = gradnew.copy()
                    _, gradnew = evalfunc(X)
                    
                    # If the gradient is zero, then we are done.
                    if dot(gradnew, gradnew) == 0:
                        print("Optimization converges. Final error: ", fnew)
                        return
                    pass
                pass
            pass
            
            # Adjust beta according to comparison ratio
            if Delta < 0.25:
                beta = min(4.0 * beta, betamax)
            pass
            if Delta > 0.75:
                beta = max(0.5 * beta, betamin)
            pass
            
            # Update search direction using Polack-Ribiere formula, or restart
            # in direction of negative gradient after nparams steps
            if nsuccess == X.params.shape[0]:
                d = -gradnew.copy()
                nsuccess = 0
            else:
                if success:
                    gamma = dot(gradold - gradnew, gradnew) / mu
                    d = dot(gamma, d) - gradnew
                pass
            pass
            epoch += 1
        pass
        # If we get here, then we haven't terminated in the given number of
        # iterations.
        print("Warning: Maximum number of iterations has been exceeded")
        

class BackpropCGTrainer(Trainer):
    """ Trainer that trains the parameters of a module according to a
    supervised dataset (potentially sequential) by backpropagating the errors
    (through time) using Polack-Ribiere conjugate gradient descent algorithm.
    This will achieve a faster and better performance compared with the normal
    BackpropTrainer. 
    The PRCG algorithm is implemented based on the MATLAB version of fmincg
    written by Carl Edward Rasmussen. """
    
    def __init__(self, module, dataset, RHO=0.0001, SIG=0.5, reEvaluate=0.1, 
                 extrapolate=3.0, maxEvaluate=20, maxSlope=100.0):
        """ Initialization method. Unlike the normal BackpropTrainer, the
        training dataset must be set during the initialization procedure when
        using conjugate gradient descent because the dataset will be used in
        the cost function and can not be passed as a variable explicitly. 
        """
        assert dataset is not None, "The training dataset can not be empty."
        Trainer.__init__(self, module)
        self.setData(dataset)
        self.epoch = 0
        self.descent = PRConjugateGradientDescent(RHO=RHO, SIG=SIG, reEvaluate=reEvaluate, 
            extrapolate=extrapolate, maxEvaluate=maxEvaluate, maxSlope=maxSlope)
        
    
    def _calcDerivs(self, module, seq):
        """ Calculate error function and backpropagate output errors to yield
        the gradient. """
        module.reset()
        for sample in seq:
            module.activate(sample[0])
        error = 0
        ponderation = 0.
        for offset, sample in reversed(list(enumerate(seq))):
            # need to make a distinction here between datasets containing
            # importance, and others
            target = sample[1]
            outerr = target - module.outputbuffer[offset]
            if len(sample) > 2:
                importance = sample[2]
                error += 0.5 * dot(importance, outerr ** 2)
                ponderation += sum(importance)
                module.backActivate(outerr * importance)
            else:
                error += 0.5 * sum(outerr ** 2)
                ponderation += len(target)
                # FIXME: the next line keeps arac from producing NaNs. I don't
                # know why that is, but somehow the __str__ method of the
                # ndarray class fixes something,
                str(outerr)
                module.backActivate(outerr)

        return error, ponderation
    
    
    def cost(self, module):
        """ Return the cost and gradient of the dataset given the network model.
        This function will be used as a handler passed to the PRCG. """
        module.resetDerivatives()
        error = 0.0
        ponderation = 0.0
        shuffledSequences = []
        for seq in self.ds._provideSequences():
            shuffledSequences.append(seq)
        shuffle(shuffledSequences)
        for seq in shuffledSequences:
            e, p = self._calcDerivs(module, seq)
            error += e
            ponderation += p
        grad = module.derivs.copy()
        error /= ponderation
        # Calculate the scaling parameter of the gradient to make the elements
        # sum to 1.0.
        norm = absolute(grad).sum()
        if norm > 10.0:
            scale = 1.0 / norm
        else:
            scale = 1.0
        return error, -grad * scale

    
    def testOnData(self, dataset, verbose=False):
        """Compute the MSE of the module performance on the given dataset. """
        dataset.reset()
        if verbose:
            print('\nTesting on data:')
        errors = []
        importances = []
        ponderatedErrors = []
        for seq in dataset._provideSequences():
            self.module.reset()
            e, i = dataset._evaluateSequence(self.module.activate, seq)
            importances.append(i)
            errors.append(e)
            ponderatedErrors.append(e / i)
        if verbose:
            print('All errors:', ponderatedErrors)
        assert sum(importances) > 0
        avgErr = sum(errors) / sum(importances)
        if verbose:
            print('Average error:', avgErr)
            print(('Max error:', max(ponderatedErrors), 'Median error:',
                   sorted(ponderatedErrors)[len(errors) / 2]))
        return avgErr


    def testOnClassData(self, dataset, verbose=False, return_targets=False):
        """Return winner-takes-all classification output on a given dataset.

        If no dataset is given, the dataset passed during Trainer
        initialization is used. If return_targets is set, also return
        corresponding target classes.
        """
        dataset.reset()
        out = []
        targ = []
        for seq in dataset._provideSequences():
            self.module.reset()
            for input, target in seq:
                res = self.module.activate(input)
                out.append(argmax(res))
                targ.append(argmax(target))
        if return_targets:
            return out, targ
        else:
            return out
            
    
    def trainUntilConvergence(self, maxEpochs=100, validationProportion=0.25, 
                              verbose=False, validationData=None, return_errors=False, 
                              convergenceThreshold=[1e-8, 1e-6]):
        """ Start the backpropagation training using the conjugate gradient
        descent algorithm. """
        # Make a copy of the origin dataset
        dataset = self.ds
        
        # Make a training dataset and a validation dataset
        if validationData is not None:
            trainingData = self.ds
        else:    
            trainingData, validationData = self.ds.splitWithProportion(1.0 - validationProportion)
        
        if not (len(trainingData) > 0 and len(validationData)):
            raise ValueError("Provided dataset too small to be split into training " +
                             "and validation sets with proportion " + str(validationProportion))
        
        # Run the conjugate descent algorithm
        self.ds = trainingData
        trainingHistory = self.descent(self.module, 
                                       self.cost, 
                                       maxEpochs=maxEpochs, 
                                       verbose=verbose, 
                                       convergenceThreshold=convergenceThreshold)
        
        # Validate the training results
        avgError = self.testOnData(validationData)
        classError = self.testOnClassData(validationData)
        print("Average error: ", str(avgError), "Class error: ", str(classError))
        
        # Restore the dataset of the trainer
        self.ds = dataset
        
        if return_errors:
            return trainingHistory
    
    



