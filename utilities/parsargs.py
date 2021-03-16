"""
Parsing command line arguments.
"""

# Standard imports
import sys
import numpy as np
#-------------------------------------------------------------------------------
def str_to_list (x, typ=int): return [typ(s) for s in x.replace(',',' ').split()]
def str_to_list_of_int (x) : return str_to_list (x, typ=int)
def str_to_list_of_float (x) : return str_to_list (x, typ=float)
def str_to_list_of_str (x) : return [s for s in x.replace(',',' ').split()]


def parseargs (argv, argspec) :

  # Strip out empty arguments
  argv = [arg for arg in argv if len(arg) > 0]

  # Create a dictionary for arguments
  argvals = {}
  for arg in argspec :
    argvals[arg] = None

  # How many arguments
  argc = len(argv)

  # Run through arguments
  argp = 1
  while True:
    # Stop loop here
    if argp >= argc : break

    # Is this a known argument
    arg = argv[argp]
    if arg in argspec :
      vtypes = argspec[arg]
      nargs = 1 if not isinstance (vtypes, list) else len(vtypes)
      theargs = argv[argp+1:argp+nargs+1]

      if isinstance (vtypes, list):
        listargs = [conv(aa) for conv, aa in zip(vtypes, theargs)]
        if len(listargs) == 0: listargs = True    # Set true for ones without args
      else :
        listargs = vtypes(theargs[0])

      # Put them in the array argvals
      argvals[arg] = listargs

      # Skip the next arguments
      argp += nargs

    argp += 1

  return argvals

#-------------------------------------------------------------------------------

def main (argv) :

  argspec = {
            # This one takes a quoted string with commas or blanks and makes a list
            '-f' : str_to_list_of_int,
            '-d' : int,
            '-c' : [int, float],
            '-x' : []     # No arguments
            }

  # Parse into a dictionary
  argvals = parseargs (argv, argspec)

  print("Arguments:")
  print (argvals)

  print (f"argvals -f = {argvals['-f']}")
  if argvals['-f'] is not None:
    print (f"What are we doing here?")
    nums = np.array(argvals['-f'])
    squares = nums*nums

  if argvals['-f'] and argvals['-d'] :
    print (f"argvals['-d'] = {argvals['-d']}")
    dval = argvals['-d']
    print (f"squares[{dval}] = {squares[dval]}")

# Run main if a script
if __name__ == '__main__':
  main (sys.argv)

#==============================================================================

