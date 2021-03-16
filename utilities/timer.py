import time
import sys


class TimingEvent :

  def __init__ (self, event_name) :
    # Constructor
    self.name_ = event_name
    self.depth_ = 0
    self.numcalls_ = 0
    self.event_processor_time_ = 0.0
    self.event_wall_time_ = 0.0
    self.total_processor_time_ = 0.0
    self.total_wall_time_ = 0.0


class Timer :

  # Static members
  timing_on = True
  look_up = {}

  def __init__ (self, context) :
    # Start a timer

    # Only do if timing is on
    if not Timer.timing_on : return

    # Find an event in the table that is linked to this name
    event = Timer.look_up.get(context)

    # Make a record if it does not exist
    if (event == None) :

      # Event does not exist, so make a new event and put in table
      event = TimingEvent (context)
      Timer.look_up[context] = event

    # Update the record
    event.depth_ += 1
    event.numcalls_ += 1

    # Only record time, first time in
    if event.depth_ == 1 :
      event.event_processor_time_ = time.process_time()
      event.event_wall_time_ = time.time()

    # Store the event 
    self.event = event

  #-----------------------------------------------------

  def free (self) :

   # Only do if timing is on
   if not Timer.timing_on : return

   # Get the processor time
   ptime = time.process_time()
   wtime = time.time()

   # get an alias for this event
   event = self.event

   # End event
   event.depth_ -= 1

   # If depth is zero, then store elapsed time
   if event.depth_ == 0 :

      # Compute the used time
      elapsed_processor_time = ptime - event.event_processor_time_
      elapsed_wall_time = wtime - event.event_wall_time_

      # Store the time
      event.total_processor_time_ += elapsed_processor_time
      event.total_wall_time_ += elapsed_wall_time

   # But if it is less than zero, there is a mistake
   if event.depth_ < 0 :
      error_message ("Timer: Too many ENDS for event \"%s\"\n", context)
      return

  #-----------------------------------------------------

  @classmethod
  def print_results(cls) :
   # Now, print out the table

   # Do not do anything if timing is not on
   if not Timer.timing_on : return

   print ("\n-----------------------------------------------------------------------------")
   print ("Timing results from Timer")

   for ev in Timer.look_up :
     total_processor_time = Timer.look_up[ev].total_processor_time_
     total_wall_time = Timer.look_up[ev].total_wall_time_
     ncalls = Timer.look_up[ev].numcalls_
     avg_time = total_processor_time / ncalls
     print ("calls {0:7d} : total {1:8.3f} (wall: {2:6.3f}) : avg {3:8.2e} : {4}".format(
          ncalls, total_processor_time, total_wall_time, avg_time, Timer.look_up[ev].name_))

   print ("-----------------------------------------------------------------------------")

#---------------------------------------------------------------

def main (argv) :

  # Make a few timers
  t0 = Timer ('Outside loop')

  for i in range (20) :
    t1 = Timer('Loop')
    for j in range (1000) :
      t3 = Timer ('mult')
      total = j+j
      t3.free()
    t1.free()

  t0.free ()

  Timer.print_results()


#---------------------------------------------------------------

if __name__ == '__main__' :
  main (sys.argv)
