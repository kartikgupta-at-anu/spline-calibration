from __future__ import absolute_import, division, print_function, unicode_literals

import functools
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
# from scipy.interpolate import splev, splrep

import utilities as utils

#-------------------------------------------------------------------------------

class Spline () :

   # Initializer
   def __init__ (self, x, y, kx, runout='parabolic') :

      # This calculates and initializes the spline

      # Store the values of the knot points
      self.kx = kx
      self.delta = kx[1] - kx[0]
      self.nknots = len(kx)
      self.runout = runout

      # Now, compute the other matrices
      m_from_ky  = self.ky_to_M ()     # Computes second derivatives from knots
      my_from_ky = np.concatenate ([m_from_ky, np.eye(len(kx))], axis=0)
      y_from_my  = self.my_to_y (x)
      y_from_ky  = y_from_my @ my_from_ky

      #print (f"\nmain:"
      #      f"\ny_from_my  = \n{utils.str(y_from_my)}"
      #      f"\nm_from_ky = \n{utils.str(m_from_ky)}"
      #      f"\nmy_from_ky = \n{utils.str(my_from_ky)}"
      #      f"\ny_from_ky = \n{utils.str(y_from_ky)}"
      #     )

      # Now find the least squares solution
      ky = np.linalg.lstsq (y_from_ky, y, rcond=-1)[0]

      # Return my
      self.ky = ky
      self.my = my_from_ky @ ky

   def my_to_y (self, vecx) :
      # Makes a matrix that computes y from M
      # The matrix will have one row for each value of x

      # Make matrices of the right size
      ndata = len(vecx)
      nknots = self.nknots
      delta = self.delta

      mM = np.zeros ((ndata, nknots))
      my = np.zeros ((ndata, nknots)) 

      for i, xx in enumerate(vecx) :
         # First work out which knots it falls between
         j = int(np.floor((xx-self.kx[0]) / delta))
         if j >= self.nknots-1: j = self.nknots - 2
         if j < 0 : j = 0
         x = xx - j * delta

         # Fill in the values in the matrices
         mM[i, j]   = -x**3 / (6.0*delta) + x**2 / 2.0 - 2.0*delta*x / 6.0
         mM[i, j+1] =  x**3 / (6.0*delta) - delta*x / 6.0
         my[i, j]   = -x/delta + 1.0
         my[i, j+1] =  x/delta

      # Now, put them together
      M = np.concatenate ([mM, my], axis=1)

      return M

   #-------------------------------------------------------------------------------

   def my_to_dy (self, vecx) :
      # Makes a matrix that computes y from M for a sequence of values x
      # The matrix will have one row for each value of x in vecx
      # Knots are at evenly spaced positions kx

      # Make matrices of the right size
      ndata = len(vecx)
      h = self.delta

      mM = np.zeros ((ndata, self.nknots))
      my = np.zeros ((ndata, self.nknots)) 

      for i, xx in enumerate(vecx) :
         # First work out which knots it falls between
         j = int(np.floor((xx-self.kx[0]) / h))
         if j >= self.nknots-1: j = self.nknots - 2
         if j < 0 : j = 0
         x = xx - j * h

         mM[i, j]   = -x**2 / (2.0*h) + x - 2.0*h / 6.0
         mM[i, j+1] =  x**2 / (2.0*h) - h / 6.0
         my[i, j]   = -1.0/h
         my[i, j+1] =  1.0/h

      # Now, put them together
      M = np.concatenate ([mM, my], axis=1)

      return M

   #-------------------------------------------------------------------------------

   def ky_to_M (self) :

      # Make a matrix that computes the 
      A = 4.0 * np.eye (self.nknots-2)
      b = np.zeros(self.nknots-2)
      for i in range (1, self.nknots-2) :
         A[i-1, i] = 1.0
         A[i, i-1] = 1.0

      # For parabolic run-out spline
      if self.runout == 'parabolic':
         A[0,0] = 5.0
         A[-1,-1] = 5.0

      # For cubic run-out spline
      if self.runout == 'cubic':
         A[0,0] = 6.0
         A[0,1] = 0.0
         A[-1,-1] = 6.0
         A[-1,-2] = 0.0

      # The goal
      delta = self.delta
      B = np.zeros ((self.nknots-2, self.nknots))
      for i in range (0, self.nknots-2) :
         B[i,i]    = 1.0
         B[i,i+1]  = -2.0
         B[i, i+2] = 1.0

      B = B * (6 / delta**2)

      # Now, solve
      Ainv = np.linalg.inv(A)
      AinvB = Ainv @ B

      # Now, add rows of zeros for M[0] and M[n-1]

      # This depends on the type of spline
      if (self.runout == 'natural') :
         z0 = np.zeros((1, self.nknots))    # for natural spline
         z1 = np.zeros((1, self.nknots))    # for natural spline

      if (self.runout == 'parabolic') :
         # For parabolic runout spline
         z0 = AinvB[0] 
         z1 = AinvB[-1] 

      if (self.runout == 'cubic') :
         # For cubic runout spline

         # First and last two rows
         z0  = AinvB[0]
         z1  = AinvB[1]
         zm1 = AinvB[-1]
         zm2 = AinvB[-2]

         z0 = 2.0*z0 - z1
         z1 = 2.0*zm1 - zm2

      #print (f"ky_to_M:"
      #       f"\nz0 = {utils.str(z0)}"
      #       f"\nz1 = {utils.str(z1)}"
      #       f"\nAinvB = {utils.str(AinvB)}"
      #      )
      
      # Reshape to (1, n) matrices
      z0 = z0.reshape((1,-1))
      z1 = z1.reshape((1, -1))

      AinvB = np.concatenate ([z0, AinvB, z1], axis=0)

      #print (f"\ncompute_spline: "
      #       f"\n A     = \n{utils.str(A)}"
      #       f"\n B     = \n{utils.str(B)}"
      #       f"\n Ainv  = \n{utils.str(Ainv)}"
      #       f"\n AinvB = \n{utils.str(AinvB)}"
      #      )

      return AinvB

   #-------------------------------------------------------------------------------

   def evaluate  (self, x) :
      # Evaluates the spline at a vector of values
      y = self.my_to_y (x) @ self.my
      return y

   #-------------------------------------------------------------------------------

   def evaluate_deriv  (self, x) :

      # Evaluates the spline at a vector (or single) point
      y = self.my_to_dy (x) @ self.my
      return y

#===============================================================================

def main (argv) :

   # Random seed
   utils.set_seed ()

   # First, get the arguments
   argspec = {
      '+gc': utils.str_to_list_of_int,     # List of classes to plot
      '+d' : utils.str_to_list_of_str      # Data files
      }

   # Get the arguments
   argvals = utils.parseargs (argv, argspec)

   # Now, try a little test
   npoints = 100
   low = 0.0
   high = 1.0
   nknots  = 7
   stdev = 0.05
   x = np.linspace (low, high, npoints)
   y = np.sin(7.0*x + 0.8)
   #y = x + x**3
   y+= np.random.normal (0.0, stdev, size=npoints)

   print (f"\nmain:"
         f"\nx = {utils.str(x)}"
         f"\ny = {utils.str(y)}"
        )

   # Now, compute the spline
   kx = np.linspace (low, high, nknots)
   spline = Spline (x, y, kx, runout='parabolic')

   # Print the error
   yint = spline.evaluate (x)
   yd   = spline.evaluate_deriv (x)
   err  = yint - y
   rms = np.sqrt(np.mean (err*err))
   sumsq = np.sum (err*err)
   print ("main:"
         f"\ny    = {utils.str(y)}"
         f"\nyint = {utils.str(yint)}"
         f"\nerr  = {utils.str(err)}"
         f"\nrms  = {utils.str(rms)}"
         f"\nnknots = {nknots}, sumsq  = {utils.str(sumsq)}"
        )

   # Also, print out the values of the coefficients for each segment
   M = spline.my[0:nknots]
   delta = 1.0 / (nknots - 1)
   ky = spline.ky
   for i in range (0, nknots-1) :
      a = (M[i+1] - M[i]) / (6.0 * delta)
      b = M[i] / 2.0
      c = (ky[i+1] - ky[i])/delta - delta * (M[i+1] + 2.0*M[i]) / 6.0
      d = ky[i]

      print (f"s[{i}] = "
            f"{utils.str(a)}*x^3 + "
            f"{utils.str(b)}*x^2 + "
            f"{utils.str(c)}*x + "
            f"{utils.str(d)}"
           )

   # Try plotting them
   fig, ax = plt.subplots (nrows=1, ncols=1)
   fig.suptitle ("Spline fit")
   ax.plot (x, yint, color='b')
   ax.plot (x, yd, color='g')
   ax.scatter (x, y, color='r')

   # Also, put the theoretical derivative
   #ax.plot (x, 1.0+3.0*x**2, color='r')
   ax.plot (x, 7.0*np.cos(7.0*x + 0.8), color='r')
   plt.show()

# Main routing
if __name__ == '__main__' :
  main (sys.argv)
