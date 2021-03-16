import numpy as np

#==============================================================================

"""
For formatting arrays properly to print
"""

def is_numpy_object (x) :
  return type(x).__module__ == np.__name__


def str (A,
         form    = "{:6.3f}",
         iform   = "{:3d}",
         sep     = '  ',
         mbegin  = '  [',
         linesep = ',\n   ',
         mend    = ']',
         vbegin  = '[',
         vend    = ']',
         end     = '',
         nvals   = -1
         ) :
  # Prints a tensorflow or numpy vector nicely
  #
  # List
  if isinstance (A, list) :
    sstr = '[' + '\n'
    for i in A :
      sstr = sstr + str(i) + '\n'
    sstr = sstr + ']'
    return sstr

  elif isinstance (A, tuple) :
    sstr = '('
    for i in A :
      sstr = sstr + str(i) + ', '
    sstr = sstr + ')'
    return sstr

  # Scalar types and None
  elif A is None : return "None"
  elif isinstance (A, float) : return form.format(A)
  elif isinstance (A, int) : return iform.format(A)

  # Othewise, try to see if it is a numpy array, or can be converted to one
  elif isinstance (A, np.ndarray) :
    if A.ndim == 0 :

      sstr = form.format(A)
      return sstr

    elif A.ndim == 1 :

      sstr = vbegin

      count = 0
      for val in A :

        # Break once enough values have been written
        if count == nvals :
          sstr = sstr + sep + "..."
          break

        if count > 0 :
          sstr = sstr + sep
        sstr = sstr + form.format(val)
        count += 1

      sstr = sstr + vend
      return sstr

    elif A.ndim == 2 :

      # Before anything else
      sstr = mbegin

      count = 0
      for i, j in rh.index2D(A) :
        # First value in new line
        if j == 0 :
          if i ==  0 :
            sstr = sstr + vbegin
          else :
            sstr = sstr + vend + linesep + vbegin

        else :
          sstr = sstr + sep

        # Print out the value
        sstr = sstr + form.format (A[i][j])

        # Break once enough values have been written
        if count == nvals :
          sstr = sstr + sep + "..."
          break
        count += 1

      # At the end
      sstr = sstr + vend + mend

      # We return the string
      return sstr

    else :
      sstr = '['
      for var in A :
        if var.ndim == 2 :
          sstr = sstr + '\n'
        sstr = sstr + str(var)
        if var.ndim == 2 :
          sstr = sstr + '\n'
      sstr = sstr + ']'
      return sstr

  # Now, try things that can be converted to numpy array
  else :
    try :
      temp = np.array (A)
      return str(temp,
                 form    = form,
                 sep     = sep,
                 mbegin  = mbegin,
                 linesep = linesep,
                 mend    = mend,
                 vbegin  = vbegin,
                 vend    = vend,
                 end     = end,
                 nvals   = nvals
                 )

    except :
      return f"{A}"

#------------------------------------------------------------------------------
def len0(x) :
  # Proper len function that REALLY works.
  # It gives the number of indices in first dimension

  # Lists and tuples
  if isinstance (x, list) :
    return len(x)

  if isinstance (x, tuple) :
    return len(x)

  # Numpy array
  if isinstance (x, np.ndarray) :
    return x.shape[0]

  # Other numpy objects have length zero
  if is_numpy_object (x) :
    return 0

  # Unindexable objects have length 0
  if x is None :
    return 0
  if isinstance (x, int) :
    return 0
  if isinstance (x, float) :
    return 0

  # Do not count strings
  if type (x) == type("a") :
    return 0

  return 0

#------------------------------------------------------------------------------

def set_seed (n=100) :
  # Set seed
  #random.seed(n)
  np.random.seed (n)
