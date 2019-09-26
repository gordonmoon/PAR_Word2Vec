# PAR-Word2Vec

The C++ and CUDA implementations of **P**arallel **A**trraction-**R**epulsion based **Word2Vec** described in the paper titled, "Parallel Data-Local Training for Optimizing Word2Vec Embeddings for Word and Graph Embeddings".

The data pre-processing and data loading parts of code are based on the pWord2Vec implementation from [Intel](https://github.com/IntelLabs/pWord2Vec).

## Dependencies
- Intel Compiler (The C++ code is optimized on Intel CPUs)
- CUDA Compiler (The CUDA code is optimized on NVIDIA Tesla P100 PCIE GPU)
- OpenMP (No separated installation is needed once Intel compiler is installed)
- MKL (The latest version "16.0.0 or higher" is preferred as it has been improved significantly in recent years)
- Python (tested on 3.6)
- NumPy (1.13.3)
- scikit-learn
- Keras (2.1.1)
- _Keras backend; default Tensorflow_
- A few other miscellaneous libraries packaged up in the `dependencies` directory for model quality evaluation
  
## Prepare seven datasets
1. Download seven datasets: `./data.sh`

## Compile the codes
1. Compile all the codes, including our implementations and other implementions: `./compile.sh`

## Validate the results in the Experimental Evaluation section of the paper
1. The directory `SC19_AE_test_cases` contains BASH test scripts for validating all the results in our SC19 submission. A pretrained word embedding text file is used for each individual evaluation task. Each test script validates each Table presented in the Experimental Evaluation section of the paper. The name of each test script corresponds to the Table number in the paper it validates.
  + Validate the results in Table5: `./Table5.sh`
  + Validate the results in Table6: `./Table6.sh`
  + Validate the results in Table7: `./Table7.sh`
  + Validate the results in Table8: `./Table8.sh`
