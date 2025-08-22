cmake_minimum_required(VERSION 3.29)

set(CMAKE_CUDA_ARCHITECTURES "70-real;80-real;86-real;89-real")

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.6)
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)

project(logistic_reg LANGUAGES CUDA CXX)
find_package(FIDESlib REQUIRED CONFIG)

enable_language(CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


enable_language(CUDA)
set(CMAKE_RELOCATABLE_DEVICE_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)

find_package(CUDAToolkit REQUIRED)
option(BUILD_STATIC "Set to ON to include static versions of the library" ON)

find_package(OpenFHE CONFIG REQUIRED)
if (OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
else ()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif ()

set(CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS})

include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/third-party/include)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)
include_directories(${OpenFHE_INCLUDE}/hook)
### add directories for other OpenFHE modules as needed for your project

link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})
if (BUILD_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    link_libraries(${OpenFHE_STATIC_LIBRARIES})
else ()
    set(CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS})
    link_libraries(${OpenFHE_SHARED_LIBRARIES})
endif ()

find_package(OpenMP)

set (SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
file (GLOB_RECURSE CXX_SOURCE_FILES ${SOURCE_DIR}/*.cpp ${SOURCE_DIR}/*.hpp)
file (GLOB_RECURSE CUDA_SOURCE_FILES ${SOURCE_DIR}/*.cu ${SOURCE_DIR}/*.cuh)

set_source_files_properties(${CXX_SOURCE_FILES} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${CUDA_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)

add_executable (resnet20)
target_sources(resnet20 PRIVATE ${CXX_SOURCE_FILES} ${CUDA_SOURCE_FILES})

if (OpenMP_CXX_FOUND)
    target_link_libraries(resnet20 PUBLIC OpenMP::OpenMP_CXX)
endif ()

if (FIDESlib_FOUND)
    message(STATUS "FOUND PACKAGE FIDESlib")
    message(STATUS "FIDESlib Version: ${FIDESlib_VERSION}")
    message(STATUS "FIDESlib header files location: ${FIDESLIB_INCLUDE_PATH}")
    message(STATUS "FIDESlib lib files location: ${FIDESLIB_LIBRARIES_PATH}")
    message(STATUS "FIDESlib binary files location: ${FIDESLIB_BINARY_PATH}")
else ()
    message(FATAL_ERROR "PACKAGE FIDESlib NOT FOUND")
endif ()

target_link_libraries(resnet20 PRIVATE FIDESlib::fideslib)

set_target_properties(resnet20 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)