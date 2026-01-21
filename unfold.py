""" this script will be used to unfold neutron spectra using various methods """

import numpy as np
import scipy
import matplotlib.pyplot as plt

# define class containing different unfolding methods
class Unfolding:

    def __init__(self, response_matrix, measured_spectrum):
        self.R = response_matrix  # response matrix shape (i,j)
        self.N = measured_spectrum  # measured spectrum shape (i, )
        self.prior = None  # prior spectrum shape (j, )

    def gravel(self, prior, iterations=100):
        self.prior = prior
        """Gravel unfolding method"""
        phi = self.prior.astype(float)          # shape (j, )
        N = self.N.astype(float)                # shape (i, ) 
        R = self.R.astype(float)                # shape (i,j)
        # assume Poissonian error for now
        sigma = np.sqrt(np.clip(N,1e-30, None) )# shape (i, ) 
        # clipping ensures we don't get any 0s in the error term later (no nans from division)
        """ print some of this out """
        print(f"Prior = {prior}")
        print(f"Measurement = {N}")
        print(f"Response matrix = {R}")
        print(f"Current phi = {phi}")


        """ start iterating """
        for i in range(iterations):

            # calc chi_2
            # r_pred is predicted response in each detector channel, i, 
            # based on the current iteration of phi (the forward model)
            r_pred = R @ phi                        # shape (i,)
            r_pred = np.clip(r_pred, 1e-30, None)
            # also denominator in weight matrix
            denom = r_pred[:, None]
            print(f"Predicted response in each detector channel \n based on current phi = {r_pred}")
            # can use this to calculate chi-squared through comparison of r_pred with N
            chi_2_red = np.mean((r_pred - N)**2/(sigma)**2)
            # check how close to consistent chi_2 value
            if abs(chi_2_red - 1) <= 0.1: 
                break
            
            # calculate weight matrix 

            # have denom now calculate the numerator
            # this is the contribution of each energy bin j to each detector channel i
            # like splitting the predicted response (above) into energy bin pieces
            # i.e r_pred = SUM_j(contrib_ij)
            contrib = R * phi[None, :]              # shape (i,j)
            num = contrib
            print(f"Contribution of each energy bin, j, (columns), \n to each detector channel, i, (rows) \n {contrib}")

            # now calculate the fractional term in the weighting matrix expression using this
            # this is, of what channel i measures, what fraction came from energy bin j
            frac_contrib = num / denom
            print(f"Fractional contribution of each energy bin, j, \n to each detector channel, i, \n {frac_contrib}")

            # now include error terms
            # tells us how "important each channel, i, response is" based on counts/errors in measurement, N
            error_term = (N**2 / sigma**2)[:, None] # ensures column vector so multiplied correctly
            print(f"Error term is: {error_term}")
            # apply this importance to get final weighting matrix
            W = frac_contrib * error_term
            print(f"Final Weights matrix is: {W}")

            # now move on to update equation itself
            # exponential term (applied to previous spectrum)
            log_term = np.log(np.clip((N/ r_pred), 1e-30, None))
            print(f"Residual vector (log term): ln(N_i/R_i) = {log_term}")
            # apply this to the weighting matrix down the columns
            exp_num = np.sum(W * log_term[:, None], axis = 0)
            print(f"Numerator of exp. term: SUM_i(W_ijln(N_i/R_i)) = {exp_num}")
            # now need to calc the denominator ready to normalise per energy bin
            exp_denom = np.clip(np.sum(W,axis = 0), 1e-30, None)
            print(f"Denominator of exp. term: SUM_i(W_ij) = {exp_denom}")
            # normalise
            # gives the amount to change each energy bin by on this iteration
            norm_exp = exp_num / exp_denom
            print(f"Normalised entries of the exp term = {norm_exp}")
            # can then get final update to be applied multiplicatively to the previous energy spec.
            update = np.exp(norm_exp)
            print(f"Update array = {update}") # determines how much each of the energy bins in the spec changes
            new_phi = phi * update
            print(f"New spectrum = {new_phi}")
            phi = new_phi
            print(f"Chi-squared is: {chi_2_red}")
            print(f"Number of iterations is: {i+1}")
        self.phi = phi
        return phi
