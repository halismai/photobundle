cmake_minimum_required(VERSION 2.8.11 FATAL_ERROR)

project(photobundle CXX)

set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  ${CMAKE_SOURCE_DIR}/cmake
  ${CMAKE_SOURCE_DIR}/cmake/modules)

option(BUILD_STATIC   "Build static libraries"  ON)

option(WITH_TBB      "with Intel TBB"          ON)
option(WITH_OPENMP   "with openmp"             ON)
option(WITH_OPENCV   "with opencv"             ON)

option(WITH_MATLAB   "compile with Matlab interface" OFF)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -Wparentheses")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -fpeel-loops")

if(BUILD_STATIC)
  set(LIBRARY_TYPE STATIC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
else()
  set(LIBRARY_TYPE SHARED)
endif()

find_package(Eigen  REQUIRED)
find_package(Ceres  REQUIRED)

include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src)

list(APPEND MY_LIBRARIES ${CERES_LIBRARIES})

find_package(TBB QUIET)
find_package(OpenMP QUIET)

if(TBB_FOUND AND WITH_TBB)
  include_directories(${TBB_INCLUDE_DIRS})
  list(APPEND MY_LIBRARIES ${TBB_LIBRARIES})
  add_definitions(-DWITH_TBB)
endif()


if(OPENMP_FOUND)
  add_definitions(-DWITH_OPENMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(Boost_USE_MULTITHREAD ON)
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
find_package(Boost 1.59.0 REQUIRED COMPONENTS log system filesystem thread timer chrono program_options)
add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})
add_definitions(-DWITH_BOOST)
include_directories(${Boost_INCLUDE_DIRS})
list(APPEND MY_LIBRARIES ${Boost_LIBRARIES})

add_subdirectory(src)

