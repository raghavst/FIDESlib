# Logistic Regression Workload

This project aims to implement a fully functional and performant 
logistic regression workload using FIDESlib. It is implemented both 
the training process and inference process using OpenFHE and 
FIDESlib. As validation, we also implement the same algoritms
on plaintext data vectors.

## Compilation

> [!IMPORTANT]
> Requirements:
>  -  Nvidia CUDA toolkit instaled.
>  -  A FIDESlib working installation. 
>  -  A OpenFHE working installation.

In order to be able to compile the project, one must follow these steps:

  - Generate the makefiles with CMake.
  ```bash
  cmake -B $PATH_TO_BUILD_DIR -S $PATH_TO_THIS_REPO --fresh 
  -DCMAKE_BUILD_TYPE="Release"
  ```
  - Build the project.
  ```bash
  cmake --build $PATH_TO_BUILD_DIR -j
  ```
The build process produces the following artifacts.

  - logistic_regression: Executable to train and run the logistic regression process.

## Execution.

The produced executable different modes of execution: train, inference and perf.

> [!WARNING]
> It is recomended to run the executable inside a build directory directly under the root directory of this project, as the paths
> to the files that contain or will contain data, weights or times are hardcoded into the code.

```bash
./logistic_regression perf [naive/cpu/gpu] [random/mnist] [iterations]
./logistic_regression train [naive/cpu/gpu] [random/mnist] [iterations] [accelerated/normal] [boot1/boot2]
./logistic_regression inference [naive/cpu/gpu] [random/mnist]
```

Argument list:
  - 1st. perf/train/inference. Do training, inference or run a performance benchmark of both operations.
  - 2nd. naive/cpu/gpu. Use the naive algorithm (plaintext), use OpenFHE (CPU) or use FIDESlib (GPU)
  - 3rd. random/mnist. Use a random generated dataset or the MNIST dataset for digit diferenciation.
  - 4th. iterations. Only on perf/train. Unsigned with the desired number of training iterations.
  - 5th. accelerated/normal. Only on train. Use Nesterov Accelerated Gradient (NAG) or the usual gradient descent.
  - 6th. boot1/boot2. Only on train. Bootstrap every 1 or 2 iterations.

Training and inference mode will read the data they need from the data directory (see next section). Training mode will produce a weight file that should be stored on the weights directory. Inference mode will read this weights and use them for the inference process.

Performance mode will measure times and will store them on times directory. This mode measures the times and accuracy for each iteration count both using NAG and normal gradient descent and bootstrapping every 2 iterations and every iteration. It also measures the bootstrapping time when it takes place.

## Data generation.

The data directory contains the scripts to generate CSV files with training and validation data.

- mnist.py: Generates training and validation data from the MNIST dataset to diferenciate between 2 digits specified inside the script. Images are scaled down to 14x14.
- random.py: Generates a test dataset with training and validation samples of clustered n-dimensional datapoints. Usefull for algorithm validation.

## Time statistics.

The times derectory contains a statistics_gen.py script that computes relevant statistics from the CSV files inside the times directory. It also dumps the generated data as a new CSV file.
