# FIDESlib

[!CAUTION] This branch is WIP, most things are working but tests won't pass and you might find some bugs or software
incompatibilites.

## New stuff:

> [!TIP]
> - Optimized **Bootstrapping** precision and performance.
> - Full single-node **Multi-GPU** support.
> - **BERT-Tiny** prototype implementation with JKLS with optimized CCMM and PCMM blocks.
> - **Logistic Regression** optimized with reduced level consumption and higher performance.
> - **ResNet20** prototype implementation with on-the-fly weight loading for reduced memory consumption.

> [!WARNING]
> Multi-GPU performance is highly dependent on parameters and system architecture, we
> recommend $N \in 2^{\{16,17\}}$.

## Features:

- Full CKKS implementation: Add, AddPt, AddScalar, Mult, MultPt, MultScalar, Square, Rotate, RotateHoisted, Bootstrap.
- OpenFHE interoperability for FIXEDMANUAL, FIXEDAUTO, FLEXIBLEAUTO and FLEXIBLEAUTOEXT.
- Hardware acceleration with Nvidia CUDA.
- High-performance NTT/INTT implementation.
- Hybrid Key-Switching.
- **NEW** Multi-GPU support via Intra-Primitive parallelism

## Compilation:

> [!IMPORTANT]
> Requirements:
>  - Nvidia CUDA version 12 or greater.
>  - GNU GCC Compiler version 10 or greater.
>  - CMake version 3.25.2 or greater.
>  - **New** OpenMP
>  - **New** NCCL

> [!NOTE]
> **Removed** Intel Thread Building Blocks for faster context creation.

> [!NOTE]
> Some dependencies will be automatically downloaded if needed:
> - GoogleTest: used by our test suite.
> - GoogleBenchmark: used by our benchmark suite.

In order to be able to compile the project, one must follow these steps:

- Clone this repository.
- Generate the makefiles with CMake.
  ```bash
  cmake -B $PATH_TO_BUILD_DIR -S $PATH_TO_THIS_REPO --fresh 
  -DFIDESLIB_BUILD_TYPE="Release" -DFIDESLIB_INSTALL_OPENFHE=ON
  ```
- Build the project.
  ```bash
  cmake --build $PATH_TO_BUILD_DIR -j
  ```

FIDESlib needs a patched version of OpenFHE in order to be able to access some internals needed for the interoperation.
This patched version can be automatically installed by defining FIDESLIB_INSTALL_OPENFHE=ON CMake variable. By default
this variable is set OFF.

> [!WARNING]
> Currently custom installation paths for patched OpenFHE are not supported. OpenFHE will be installed on the default
> path specified in their build files and you will probably need to run the build files generation command with
> administrator priviledges.

The build process produces the following artifacts:

- fideslib.a: The FIDESlib library to be statically linked to any client application.
- fideslib-test: The test suite executable.
- fideslib-bench: The benchark suite executable.
- gpu-test: A dummy executable to search for the CUDA capable devices on the machine.
- dummy: Another dummy executable.

> [!WARNING]
> Compiling FIDESlib sometimes gives out TLS related errors. This issue can be adressed by re-compiling OpenFHE in debug
> mode. In this case, you should:
> - Manually clone [OpenFHE](https://github.com/openfheorg/openfhe-development) and, with git, apply openfhe-hook.patch
    and openfhe-base.patch.
> - Generate the build files with CMake using Debug as build type.
> - Compile and install OpenFHE on the machine.

## Instalation

Installing the library is as easy as running the following command:

```bash
cmake --build $PATH_TO_BUILD_DIR --target install -j
```

FIDESlib is currently ready to be consumed as a CMake library. The template project on the examples directory shows how
to build and run a FIDESlib client application and contains examples of usage of most of the functionality provided by
FIDESlib. Currently client applications consuming FIDESlib should use the CUDA compiler every time they include a
FIDESlib header.

> [!NOTE]
> As the default installation prefix is /usr/local. All installed headers should be located under
> /usr/local/include/FIDESlib, the CMake package files under /usr/local/share/FIDESlib and the compiled library under
> /usr/local/lib.

> [!WARNING]
> FIDESlib currently does not support custom installation paths. One should run the installation command with
> administrator priviledges.

## Usage:

Check examples/template for a demo project that uses FIDESlib.
  
