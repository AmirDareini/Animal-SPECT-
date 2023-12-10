Animal SPECT image reconstruction project 
The whole process: 
  1) Acquire data using SPECT machine
  2) Store the data in sinograms
  3) develope forward modeling algorithms
  4) incldue resolution recovery in system modeling 
  5) reconstrcut image using system matrices in iterative process
  6) Visulize the data

-> Acquired data from SPECT machine: 
output matrix from the machine is a 1x3 dimensional matrix -> [phi_bin , slice, rbin]
phi-bin : 32 for a single head 
slice: 32 for a single head
rbin: 64 for a single head 

