cmake_minimum_required(VERSION 2.8)
#free standing module to do the GPU stuff that always needs to be able to be cross compiled
project(GPU_Builds_LaserSynthesiser)
set(CURR_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CURR_SRC_DIR}/CMake/Modules")
#windows definitions to prevent silliness
add_definitions(-D_WIN32_WINNT=0x0601)
add_definitions(-DWIN32_LEAN_AND_MEAN)
add_definitions(-D_MBCS)

#OpenCV search
find_package(OpenCV)
#check for boost
FIND_PACKAGE(Boost REQUIRED)


option(NO_CUDA "set this if the machine doesn't even have a working cuda installation" OFF)

if(${NO_CUDA})

set(LOCAL_SOURCES  ${CURR_SRC_DIR}/DidoLidar/DidoLidar_CUDA.cpp)
set(LOCAL_HEADERS ${CURR_SRC_DIR}/DidoLidar/DidoLidar_CUDA.h)

include_directories(${CURR_SRC_DIR}/utilities ${Boost_INCLUDE_DIR} ${CURR_SRC_DIR}/DidoLidar} ${OpenCV_INCLUDE} ${CURR_SRC_DIR}/user_interface/server/ovlib/include)

add_library(DidoLidar ${LOCAL_SOURCES} ${LOCAL_HEADERS})

else()
find_package(CUDA QUIET REQUIRED)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Pass options to NVCC
set( CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
	# > 1080 is arch_61
    -arch=sm_61
 #   -O3
     -std=c++11 
  # -Xcompiler -D__CORRECT_ISO_CPP11_MATH_H_PROTO
    )

#if we're in windows and don't have a GPU, we need to set the appropriate preprocessor definition
option(NO_GPU "set this if you don't have a GPU on the machine but you do have CUDA" OFF)
if(${NO_GPU})
   set(CUDA_NVCC_FLAGS  ${CUDA_NVCC_FLAGS}; -DDIDOLIDAR_NOGPU=1)
endif()


set(LOCAL_SOURCES  ${CURR_SRC_DIR}/DidoLidar/DidoLidar_CUDA.cu)
set(LOCAL_HEADERS ${CURR_SRC_DIR}/DidoLidar/DidoLidar_CUDA.h)

include_directories(${CURR_SRC_DIR}/utilities ${Boost_INCLUDE_DIR} ${CURR_SRC_DIR}/DidoLidar} ${OpenCV_INCLUDE})

CUDA_ADD_LIBRARY(DidoLidar SHARED ${LOCAL_SOURCES} ${LOCAL_HEADERS})

#protect it from the C++14 requirement

SET_TARGET_PROPERTIES(DidoLidar PROPERTIES
		CXX_STANDARD 11 )

endif()

target_link_libraries(DidoLidar ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})

# To fix compilation problem: relocation R_X86_64_32 against `a local symbol' can not be
# used when making a shared object; recompile with -fPIC
# See http://www.cmake.org/pipermail/cmake/2007-May/014350.html
#
IF( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )
  SET_TARGET_PROPERTIES(DidoLidar PROPERTIES COMPILE_FLAGS "-fPIC")
ENDIF( CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" )

#add the library to the parent scope if needed
get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
set(GATHERED_LIBRARIES ${GATHERED_LIBRARIES} ${Boost_LIBRARIES} DidoLidar PARENT_SCOPE)
set(GATHERED_INCLUDES ${GATHERED_INCLUDES} ${CURR_SRC_DIR}/DidoLidar  ${CURR_SRC_DIR} ${OpenCV_INCLUDE} ${Boost_INCLUDE_DIR} PARENT_SCOPE)
endif()
