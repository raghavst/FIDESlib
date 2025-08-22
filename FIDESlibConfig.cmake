############### Configuration ###################

get_filename_component(FIDESLIB_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
if(NOT FIDESLIB_BINARY_DIR)
  include("${FIDESLIB_CMAKE_DIR}/FIDESlibTargets.cmake")
endif()

############### Macros ###################

include(CMakeFindDependencyMacro)

############### CUDA ###################

find_dependency(CUDAToolkit REQUIRED)

############### Exports ###################

set(FIDESLIB_INCLUDE_PATH /usr/local/include/FIDESlib)
set(FIDESLIB_LIBRARIES_PATH /usr/local/lib64)
set(FIDESLIB_BINARY_PATH /usr/local/bin)
