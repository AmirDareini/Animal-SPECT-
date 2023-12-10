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


-> iterative process in image reconstruction : 
    measured data is defined
    initial estimation is defined
    for i in rage (number of iteration): 
      error term = measured_data / initial estimation 
      correction factor = backproject (error term )\
      initial estimation *= correction factor 
in this process we make our initial estimation of image better and better in each itertion untill get to convergence. 
2 main iterative algorithms we use in nuclear medicine: 
  Maximujm Likelihood expectatoin maximization (MLEM) - Ordering subset expectation maximization (OSEM)
  


