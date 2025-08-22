
# OpenFHE modifications

FIDESlib needs a modified version of OpenFHE to be able to function properly.

This modified version only changes the visibility of some class members to be able to access them from FIDESlib.
Moreover, for debugging purposes, we have created a modified version of OpenFHE with a hook component that 
allows to record any data that is being processed by the library and its consumers.

This modifications are stored in this directory in the form of git patches.

We also provide 2 scripts to ease the process of applying these patches to OpenFHE and installing the modified version of the library.

#### Patch script

patch.sh is a bash script that given the path to the OpenFHE directory, applies the necessary changes to make it compatible with FIDESlib.

```bash
./patch.sh <path-to-OpenFHE-source>
```

#### Installation script

install.sh is a bash script that given the path to the OpenFHE directory, and optionally the absolute path to the desired installation directory, builds and installs the the version of OpenFHE inside the given path.
```bash
./install.sh <path-to-OpenFHE-source> <absolute-path-to-installation-location>
```

## FIDESlib compatibility patch

Provides the necessary changes to OpenFHE to make it compatible with FIDESlib.

To apply this patch manually, you need to execute the following command in the OpenFHE source directory:

```bash
git am <path-to-this-directory>/openfhe-base.patch
```


## Hook patch

Provides the necessary changes to OpenFHE to add a hook component that allows to record data.

To apply this patch manually, you need to execute the following command in the OpenFHE directory:

```bash
git am <path-to-this-directory>/openfhe-hook.patch
```

### Hook component usage

To use the hook component in OpenFHE, you need to add the following code to the OpenFHE header or source file that you want to debug:

```cpp
#include "hook/hook.h"
```

Data storage is managed using indexed records unique to each data type. Data management is done using the following macros:

```cpp
RECORD_SET(data, index); // Store
RECORD_GET(decltype(data), index); // Retrieve
RECORD_PRINT(decltype(data), index); // Print
```

You need to take into account that you will need to recompile OpenFHE to materialize the hook usage inside the library.

It is also possible to access this macros from a consumer of OpenFHE by including the hook header file as it was done before. For this,
you will need to install the modified version of OpenFHE with the hook component in your system first. Then you will need to include 
OpenFHE on your proyect with CMake using the template CMakeLists.Patch.txt provided in this directory.

This template adds only the following line to the original template provided by OpenFHE:

```cmake
include_directories( ${OpenFHE_INCLUDE}/hook )
```

After all this, you will be able to use the hook component in your proyect, and even access recorded data inside OpenFHE if the compiled 
version of the library was using the hook component to record data.

## Note

The provided patch files and scripts are tested agaist OpenFHE 1.2.0 (concrety repository tag: v1.2.0). It is not guaranteed that they will work with other versions of the OpenFHE.
