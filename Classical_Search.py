import time


runTime:float
# Searches a genetic string for a target mutation
# Returns location of mutation, iterations taken, and time taken to find the target.
def linear_search(sequence, target):
  
   startTime:float= time.time()
   steps:int =0
   target_dna_length = len(target)
  
   for i in range(0, len(sequence)-target_dna_length+1, 3):
       steps +=1
       substring =sequence[i:i+target_dna_length]


       if substring== target:
           endTime:float= time.time()
           runTime:float = endTime - startTime
           print(f"Found {target} ")
           return True, steps, runTime
           # Returns location of mutation, iterations taken, and time taken to find the target.
          
   endTime:float= time.time()
   runTime:float = endTime - startTime
   print("target not present")
   return False, steps, runTime
